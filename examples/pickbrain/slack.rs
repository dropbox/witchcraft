use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use witchcraft::DB;

const SLACK_NAMESPACE: Uuid = Uuid::from_bytes([
    0x95, 0xdc, 0xad, 0xe9, 0x5d, 0xaa, 0x4d, 0xe2, 0x91, 0x3b, 0xe5, 0xac, 0xfd, 0x9d, 0x3c,
    0x07,
]);

const DM_MAX_GAP_MS: i64 = 90 * 60_000;
const CHANNEL_MAX_GAP_MS: i64 = 30 * 60_000;
const MAX_SESSION_SPAN_MS: i64 = 24 * 3_600_000;
const SYSTEM_SUBTYPES: &[&str] = &[
    "channel_join",
    "channel_joined",
    "group_join",
    "group_joined",
    "channel_leave",
    "group_leave",
    "channel_archive",
    "group_archive",
    "channel_unarchive",
    "group_unarchive",
    "channel_name",
    "group_name",
    "channel_purpose",
    "group_purpose",
    "channel_topic",
    "group_topic",
    "pinned_item",
    "unpinned_item",
    "message_changed",
    "message_deleted",
];

struct ParsedMessage {
    ts: String,
    ts_ms: i64,
    thread_ts: Option<String>,
    channel_id: String,
    channel_name: String,
    is_dm: bool,
    sender_name: String,
    normalized_text: String,
    client_msg_id: Option<String>,
    is_broadcast: bool,
    team_id: String,
}

struct Conversation {
    key: String,
    channel_id: String,
    channel_name: String,
    is_dm: bool,
    team_id: String,
    root_ref_ts: String,
    first_ts_ms: i64,
    latest_ts_ms: i64,
    has_unread: bool,
    unread_count: usize,
    messages: Vec<ConvMessage>,
}

struct ConvMessage {
    ts_ms: i64,
    labeled: String,
}

struct OpenSession {
    key: String,
    last_ts_ms: i64,
    first_ts_ms: i64,
    root_ref_ts: String,
}

fn find_slack_blob_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let idb_dir = PathBuf::from(home).join("Library/Application Support/Slack/IndexedDB");
    if !idb_dir.is_dir() {
        return None;
    }
    for entry in std::fs::read_dir(&idb_dir).ok()?.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.ends_with(".indexeddb.blob") && entry.path().is_dir() {
            return Some(entry.path());
        }
    }
    None
}

fn blob_dir_mtime(dir: &Path) -> i64 {
    let mut max_ms: i64 = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&path) {
                    for f in sub.flatten() {
                        if let Ok(meta) = f.metadata() {
                            if let Ok(t) = meta.modified() {
                                let ms = t
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as i64;
                                max_ms = max_ms.max(ms);
                            }
                        }
                    }
                }
            }
        }
    }
    max_ms
}

fn watermark_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(".pickbrain")
        .join("slack.watermark")
}

fn self_user_cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(".pickbrain")
        .join("slack_self_user")
}

fn cache_self_user(id: &str, name: &str) {
    let p = self_user_cache_path();
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(p, format!("{id}\n{name}")).ok();
}

pub fn cached_self_user() -> Option<(String, String)> {
    let s = std::fs::read_to_string(self_user_cache_path()).ok()?;
    let mut lines = s.lines();
    let id = lines.next()?.trim().to_string();
    let name = lines.next()?.trim().to_string();
    if id.is_empty() || name.is_empty() { None } else { Some((id, name)) }
}

fn read_watermark() -> i64 {
    std::fs::read_to_string(watermark_path())
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

pub fn remove_watermark() {
    std::fs::remove_file(watermark_path()).ok();
}

fn write_watermark(ms: i64) {
    let p = watermark_path();
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(p, ms.to_string()).ok();
}

fn extract_redux_state(blobs: &[Value]) -> Option<&Value> {
    for blob in blobs {
        if let Some(val) = blob.get("value") {
            if val.is_object() && val.get("messages").is_some() {
                return Some(val);
            }
        }
    }
    None
}

fn should_skip_message(msg: &Value, members: &Value) -> bool {
    if let Some(user_id) = msg.get("user").and_then(|v| v.as_str()) {
        if members
            .get(user_id)
            .and_then(|m| m.get("is_bot"))
            .and_then(|b| b.as_bool())
            == Some(true)
        {
            return true;
        }
        if user_id == "USLACKBOT" {
            return true;
        }
    }
    if let Some(up) = msg.get("user_profile") {
        if up.get("is_bot").and_then(|b| b.as_bool()) == Some(true) {
            return true;
        }
    }
    if let Some(username) = msg.get("username").and_then(|v| v.as_str()) {
        if username.to_lowercase().contains("slackbot") {
            return true;
        }
    }
    let subtype = msg.get("subtype").and_then(|v| v.as_str()).unwrap_or("");
    if subtype == "bot_message" || subtype == "slackbot_response" {
        return true;
    }
    if SYSTEM_SUBTYPES.contains(&subtype) {
        return true;
    }
    if msg.get("bot_id").and_then(|v| v.as_str()).is_some()
        || msg.get("bot_profile").is_some()
        || msg.get("app_id").and_then(|v| v.as_str()).is_some()
    {
        return true;
    }
    if msg.get("hidden").and_then(|v| v.as_bool()) == Some(true) {
        return true;
    }
    let text = msg.get("text").and_then(|v| v.as_str()).unwrap_or("");
    text.is_empty()
}

fn strip_code(s: &str) -> String {
    let re_fenced = Regex::new(r"```[\s\S]*?```").unwrap();
    let re_fenced_open = Regex::new(r"```[\s\S]*$").unwrap();
    let re_inline = Regex::new(r"`[^`]*`").unwrap();
    let s = re_fenced.replace_all(s, " ");
    let s = re_fenced_open.replace_all(&s, " ");
    re_inline.replace_all(&s, " ").into_owned()
}

fn decode_html_entities(s: &str) -> String {
    let re_hex = Regex::new(r"&#x([0-9a-fA-F]+);").unwrap();
    let re_dec = Regex::new(r"&#(\d+);").unwrap();
    let re_named = Regex::new(r"&([a-zA-Z]+);").unwrap();

    let s = re_hex
        .replace_all(s, |caps: &regex::Captures| {
            u32::from_str_radix(&caps[1], 16)
                .ok()
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_default()
        })
        .into_owned();
    let s = re_dec
        .replace_all(&s, |caps: &regex::Captures| {
            caps[1]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_default()
        })
        .into_owned();
    re_named
        .replace_all(&s, |caps: &regex::Captures| {
            match caps[1].to_lowercase().as_str() {
                "amp" => "&",
                "lt" => "<",
                "gt" => ">",
                "quot" => "\"",
                "apos" => "'",
                "nbsp" => "\u{00A0}",
                "mdash" => "\u{2014}",
                "ndash" => "\u{2013}",
                "hellip" => "\u{2026}",
                _ => return caps[0].to_string(),
            }
            .to_string()
        })
        .into_owned()
}

fn expand_mentions(s: &str, members: &Value) -> String {
    let re = Regex::new(r"<@([A-Z0-9]+)>").unwrap();
    re.replace_all(s, |caps: &regex::Captures| {
        members
            .get(&caps[1])
            .and_then(|m| m.get("real_name"))
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string()
    })
    .into_owned()
}

fn resolve_channels(s: &str, channels: &Value, channel_id: &str, channel_name: &str) -> String {
    let re = Regex::new(r"[(\[]?<#([A-Z0-9]+)(?:\|([^>]*))?>[)\]]?").unwrap();
    re.replace_all(s, |caps: &regex::Captures| {
        let ref_id = &caps[1];
        let ref_name = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let name = if !ref_name.is_empty() {
            ref_name.to_string()
        } else if let Some(ch) = channels.get(ref_id) {
            ch.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string()
        } else if ref_id == channel_id {
            channel_name.to_string()
        } else {
            String::new()
        };
        if name.is_empty() {
            String::new()
        } else {
            format!("#{name}")
        }
    })
    .into_owned()
}

fn unwrap_slack_links(s: &str) -> String {
    let re_labeled = Regex::new(r"<((?:https?://)[^>|]+)\|([^>]+)>").unwrap();
    let re_bare = Regex::new(r"<((?:https?://)[^>]+)>").unwrap();
    let s = re_labeled.replace_all(s, "$2");
    re_bare.replace_all(&s, "$1").into_owned()
}

fn strip_leading_label(s: &str) -> String {
    let re = Regex::new(r"^\[[^\]]+\]\s*").unwrap();
    re.replace(s, "").into_owned()
}

fn resolve_emoji(s: &str) -> String {
    let re = Regex::new(r":([a-zA-Z0-9_+-]+):").unwrap();
    re.replace_all(s, |caps: &regex::Captures| {
        match caps[1].to_lowercase().as_str() {
            "smile" | "slightly_smiling_face" | "smiley" => "\u{1F642}",
            "thumbsup" | "+1" => "\u{1F44D}",
            "thumbsdown" | "-1" => "\u{1F44E}",
            "heart" => "\u{2764}\u{FE0F}",
            "tada" | "party_popper" => "\u{1F389}",
            "wave" => "\u{1F44B}",
            "thinking_face" | "thinking" => "\u{1F914}",
            "eyes" => "\u{1F440}",
            "fire" => "\u{1F525}",
            "rocket" => "\u{1F680}",
            "100" => "\u{1F4AF}",
            "white_check_mark" => "\u{2705}",
            "x" => "\u{274C}",
            "warning" => "\u{26A0}\u{FE0F}",
            "pray" => "\u{1F64F}",
            "clap" => "\u{1F44F}",
            "raised_hands" => "\u{1F64C}",
            "point_up" => "\u{261D}\u{FE0F}",
            "muscle" => "\u{1F4AA}",
            "star" => "\u{2B50}",
            "sparkles" => "\u{2728}",
            "bulb" | "light_bulb" => "\u{1F4A1}",
            "memo" => "\u{1F4DD}",
            "speech_balloon" => "\u{1F4AC}",
            "rotating_light" => "\u{1F6A8}",
            "heavy_check_mark" => "\u{2714}\u{FE0F}",
            "question" => "\u{2753}",
            "exclamation" => "\u{2757}",
            "laughing" | "joy" | "rofl" => "\u{1F602}",
            "cry" | "sob" => "\u{1F622}",
            "angry" | "rage" => "\u{1F621}",
            "sweat_smile" => "\u{1F605}",
            "sunglasses" => "\u{1F60E}",
            "wink" => "\u{1F609}",
            "blush" => "\u{1F60A}",
            "grimacing" => "\u{1F62C}",
            "skull" => "\u{1F480}",
            _ => "",
        }
        .to_string()
    })
    .into_owned()
}

fn normalize(
    msg: &Value,
    members: &Value,
    channels: &Value,
    channel_id: &str,
    channel_name: &str,
) -> String {
    let raw = msg.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let s = strip_code(raw);
    let s = decode_html_entities(&s);
    let s = expand_mentions(&s, members);
    let s = resolve_channels(&s, channels, channel_id, channel_name);
    let s = unwrap_slack_links(&s);
    let re_ws = Regex::new(r"\s{2,}").unwrap();
    let s = re_ws.replace_all(&s, " ").into_owned();
    let s = strip_leading_label(&s);
    let s = resolve_emoji(&s);
    s.trim().to_string()
}

fn resolve_self_user(state: &Value) -> Option<(String, String)> {
    let user_id = state
        .get("bootData")
        .and_then(|b| b.get("user_id"))
        .and_then(|v| v.as_str())?;
    let members = state.get("members")?;
    let member = members.get(user_id)?;
    let real_name = member
        .get("real_name")
        .and_then(|n| n.as_str())
        .filter(|n| !n.is_empty())?;
    let handle = member
        .get("name")
        .and_then(|n| n.as_str())
        .filter(|n| !n.is_empty());
    let label = match handle {
        Some(h) => format!("{real_name} (@{h})"),
        None => real_name.to_string(),
    };
    Some((user_id.to_string(), label))
}

fn resolve_sender(msg: &Value, members: &Value, self_user: Option<&(String, String)>) -> String {
    if let Some(user_id) = msg.get("user").and_then(|v| v.as_str()) {
        if let Some(member) = members.get(user_id) {
            let real_name = member.get("real_name").and_then(|n| n.as_str()).filter(|n| !n.is_empty());
            let handle = member.get("name").and_then(|n| n.as_str()).filter(|n| !n.is_empty());
            match (real_name, handle) {
                (Some(rn), Some(h)) => return format!("{rn} (@{h})"),
                (Some(rn), None) => return rn.to_string(),
                (None, Some(h)) => return format!("@{h}"),
                _ => {}
            }
        }
        if let Some((self_id, self_name)) = self_user {
            if user_id == self_id {
                return self_name.clone();
            }
        }
    }
    msg.get("username")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string()
}

fn ts_to_ms(ts: &str) -> Option<i64> {
    let f: f64 = ts.parse().ok()?;
    let ms = (f * 1000.0) as i64;
    if ms > 0 { Some(ms) } else { None }
}

fn collect_messages(state: &Value, self_user: Option<&(String, String)>) -> Vec<ParsedMessage> {
    let messages = match state.get("messages").and_then(|v| v.as_object()) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let members = state.get("members").unwrap_or(&Value::Null);
    let channels = state.get("channels").unwrap_or(&Value::Null);
    let default_team = state
        .get("selfTeamIds")
        .and_then(|v| v.get("defaultWorkspaceId"))
        .and_then(|v| v.as_str())
        .or_else(|| state.get("defaultTeamId").and_then(|v| v.as_str()))
        .unwrap_or("");

    let mut parsed = Vec::new();

    for (channel_id, thread_bucket) in messages {
        let thread_bucket = match thread_bucket.as_object() {
            Some(b) => b,
            None => continue,
        };
        let channel_name = channels
            .get(channel_id)
            .and_then(|c| c.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or(channel_id);
        let is_dm = channels
            .get(channel_id)
            .and_then(|c| c.get("is_im"))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);

        for (_ts, msg) in thread_bucket {
            let ts = match msg.get("ts").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => continue,
            };
            let ts_ms = match ts_to_ms(ts) {
                Some(ms) => ms,
                None => continue,
            };
            if should_skip_message(msg, members) {
                continue;
            }

            let normalized = normalize(msg, members, channels, channel_id, channel_name);
            if normalized.trim().is_empty() {
                continue;
            }

            let thread_ts = msg.get("thread_ts").and_then(|v| v.as_str()).map(String::from);
            let subtype = msg.get("subtype").and_then(|v| v.as_str()).unwrap_or("");
            let is_broadcast =
                subtype == "thread_broadcast" || subtype == "reply_broadcast";
            let team_id = msg
                .get("source_team_id")
                .and_then(|v| v.as_str())
                .unwrap_or(default_team);

            parsed.push(ParsedMessage {
                ts: ts.to_string(),
                ts_ms,
                thread_ts,
                channel_id: channel_id.to_string(),
                channel_name: channel_name.to_string(),
                is_dm,
                sender_name: resolve_sender(msg, members, self_user),
                normalized_text: normalized,
                client_msg_id: msg
                    .get("client_msg_id")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                is_broadcast,
                team_id: team_id.to_string(),
            });
        }
    }

    parsed.sort_by_key(|m| m.ts_ms);
    parsed
}

fn dedup_broadcasts(messages: Vec<ParsedMessage>) -> Vec<ParsedMessage> {
    let mut by_client: HashMap<String, usize> = HashMap::new();
    let mut by_ts: HashMap<String, usize> = HashMap::new();
    let mut all: Vec<ParsedMessage> = Vec::new();

    for msg in messages {
        if let Some(ref cid) = msg.client_msg_id {
            if let Some(&prev_idx) = by_client.get(cid) {
                if !msg.is_broadcast && all[prev_idx].is_broadcast {
                    all[prev_idx] = msg;
                }
            } else {
                let idx = all.len();
                by_client.insert(cid.clone(), idx);
                all.push(msg);
            }
        } else {
            let ts_key = msg.ts.clone();
            if let Some(&prev_idx) = by_ts.get(&ts_key) {
                if !msg.is_broadcast && all[prev_idx].is_broadcast {
                    all[prev_idx] = msg;
                }
            } else {
                let idx = all.len();
                by_ts.insert(ts_key, idx);
                all.push(msg);
            }
        }
    }

    all.sort_by_key(|m| m.ts_ms);
    all
}

fn extract_channel_cursors(state: &Value) -> HashMap<String, String> {
    let mut cursors = HashMap::new();
    if let Some(obj) = state.get("channelCursors").and_then(|v| v.as_object()) {
        for (ch_id, ts_val) in obj {
            if let Some(ts) = ts_val.as_str() {
                cursors.insert(ch_id.clone(), ts.to_string());
            }
        }
    }
    cursors
}

fn group_into_conversations(
    messages: Vec<ParsedMessage>,
    cursors: &HashMap<String, String>,
) -> Vec<Conversation> {
    let mut conversations: HashMap<String, Conversation> = HashMap::new();
    let mut open_sessions: HashMap<String, OpenSession> = HashMap::new();

    for msg in &messages {
        let (key, root_ref_ts) = if let Some(ref tts) = msg.thread_ts {
            (
                format!("thr:{}-{}", msg.channel_id, tts),
                tts.clone(),
            )
        } else {
            let max_gap = if msg.is_dm {
                DM_MAX_GAP_MS
            } else {
                CHANNEL_MAX_GAP_MS
            };

            if let Some(session) = open_sessions.get_mut(&msg.channel_id) {
                let gap = msg.ts_ms - session.last_ts_ms;
                let span = msg.ts_ms - session.first_ts_ms;
                if gap >= 0 && gap <= max_gap && span <= MAX_SESSION_SPAN_MS {
                    session.last_ts_ms = msg.ts_ms;
                    (session.key.clone(), session.root_ref_ts.clone())
                } else {
                    let key = format!("sess:{}-{}", msg.channel_id, msg.ts);
                    let fresh = OpenSession {
                        key: key.clone(),
                        last_ts_ms: msg.ts_ms,
                        first_ts_ms: msg.ts_ms,
                        root_ref_ts: msg.ts.clone(),
                    };
                    let rts = msg.ts.clone();
                    open_sessions.insert(msg.channel_id.clone(), fresh);
                    (key, rts)
                }
            } else {
                let key = format!("sess:{}-{}", msg.channel_id, msg.ts);
                let fresh = OpenSession {
                    key: key.clone(),
                    last_ts_ms: msg.ts_ms,
                    first_ts_ms: msg.ts_ms,
                    root_ref_ts: msg.ts.clone(),
                };
                let rts = msg.ts.clone();
                open_sessions.insert(msg.channel_id.clone(), fresh);
                (key, rts)
            }
        };

        let labeled = format!("[{}] {}\n", msg.sender_name, msg.normalized_text);

        let conv = conversations.entry(key.clone()).or_insert_with(|| Conversation {
            key: key.clone(),
            channel_id: msg.channel_id.clone(),
            channel_name: msg.channel_name.clone(),
            is_dm: msg.is_dm,
            team_id: msg.team_id.clone(),
            root_ref_ts: root_ref_ts.clone(),
            first_ts_ms: msg.ts_ms,
            latest_ts_ms: msg.ts_ms,
            has_unread: false,
            unread_count: 0,
            messages: Vec::new(),
        });

        if let Some(cursor) = cursors.get(&msg.channel_id) {
            if msg.ts.as_str() > cursor.as_str() {
                conv.has_unread = true;
                conv.unread_count += 1;
            }
        }

        conv.latest_ts_ms = conv.latest_ts_ms.max(msg.ts_ms);
        conv.messages.push(ConvMessage {
            ts_ms: msg.ts_ms,
            labeled,
        });
    }

    let mut result: Vec<Conversation> = conversations.into_values().collect();
    for conv in &mut result {
        conv.messages.sort_by_key(|m| m.ts_ms);
    }
    result.sort_by_key(|c| c.first_ts_ms);
    result
}

fn ingest_conversations(db: &mut DB, conversations: Vec<Conversation>) -> Result<usize> {
    let mut count = 0;
    let mut channel_idx: HashMap<String, usize> = HashMap::new();
    for conv in &conversations {
        if conv.messages.is_empty() {
            continue;
        }

        let body: String = conv.messages.iter().map(|m| m.labeled.as_str()).collect();
        let lens: Vec<usize> = conv
            .messages
            .iter()
            .map(|m| m.labeled.chars().count())
            .collect();

        let uuid = Uuid::new_v5(
            &SLACK_NAMESPACE,
            format!("{}-{}", conv.channel_id, conv.root_ref_ts).as_bytes(),
        );

        let date_secs = conv.first_ts_ms / 1000;
        let date = chrono::DateTime::from_timestamp(date_secs, 0)
            .map(|dt| dt.to_rfc3339());

        let channel_label = if conv.is_dm {
            "DM".to_string()
        } else {
            conv.channel_name.clone()
        };

        let idx = channel_idx.entry(conv.channel_id.clone()).or_insert(0);
        let turn = *idx;
        *idx += 1;

        let metadata = serde_json::json!({
            "source": "slack",
            "project": channel_label,
            "channel_id": conv.channel_id,
            "channel_name": conv.channel_name,
            "is_dm": conv.is_dm,
            "team_id": conv.team_id,
            "conv_key": conv.key,
            "turn": turn,
            "session_id": conv.channel_id,
            "message_count": conv.messages.len(),
            "has_unread": conv.has_unread,
            "unread_count": conv.unread_count,
        });

        let ts = date.as_ref().map(|d| {
            iso8601_timestamp::Timestamp::parse(d).unwrap_or_else(|| iso8601_timestamp::Timestamp::now_utc())
        });

        db.add_doc(
            &uuid,
            ts,
            &metadata.to_string(),
            &body,
            Some(lens),
        )?;
        count += 1;
    }
    Ok(count)
}

pub fn ingest_slack(db: &mut DB) -> Result<usize> {
    let blob_dir = match find_slack_blob_dir() {
        Some(d) => d,
        None => return Ok(0),
    };

    let current_mtime = blob_dir_mtime(&blob_dir);
    let prev_watermark = read_watermark();
    if current_mtime > 0 && current_mtime <= prev_watermark {
        return Ok(0);
    }

    let tmp_dir = tempfile::tempdir().context("failed to create temp dir")?;
    let tmp_blob = tmp_dir.path().join("slack.blob");
    copy_dir_recursive(&blob_dir, &tmp_blob)?;

    let blobs = chromium_idb::dump_blobs(&tmp_blob, false)
        .map_err(|e| anyhow::anyhow!("failed to read Slack blobs: {}", e))?;

    let state = match extract_redux_state(&blobs) {
        Some(s) => s,
        None => return Ok(0),
    };

    let self_user = resolve_self_user(state);
    if let Some((ref id, ref name)) = self_user {
        cache_self_user(id, name);
    }
    let cursors = extract_channel_cursors(state);
    let messages = collect_messages(state, self_user.as_ref());
    let deduped = dedup_broadcasts(messages);
    let conversations = group_into_conversations(deduped, &cursors);
    let count = ingest_conversations(db, conversations)?;

    if current_mtime > 0 {
        write_watermark(current_mtime);
    }

    Ok(count)
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}
