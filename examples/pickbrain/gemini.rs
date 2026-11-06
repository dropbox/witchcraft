use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use witchcraft::DB;

pub const MAX_CHUNK_CODEPOINTS: usize = 4000;
pub const MAX_TURN_CODEPOINTS: usize = 6000;

static RE_META: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<ADDITIONAL_METADATA[\s>].*?(?:</ADDITIONAL_METADATA>|$)").unwrap()
});
static RE_SYS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<SYSTEM_MESSAGE[\s>].*?(?:</SYSTEM_MESSAGE>|$)").unwrap()
});
static RE_SETTINGS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<USER_SETTINGS_CHANGE[\s>].*?(?:</USER_SETTINGS_CHANGE>|$)").unwrap()
});
static RE_REQ: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)</?USER_REQUEST\b[^>]*>").unwrap()
});
static RE_FENCED_CODE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```+[\s\S]*?```+").unwrap()
});
static RE_UNCLOSED_CODE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```+[\s\S]*$").unwrap()
});
static RE_SPACES: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[ \t]+").unwrap()
});
static RE_NEWLINES: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\n[ \t]*\n+").unwrap()
});

pub fn sanitize_gemini_text(text: &str) -> String {
    let mut s = text.to_string();

    let phrase = "not actually sent by the user";
    if let Some(idx) = s.find(phrase) {
        if let Some(start) = s[..idx].rfind("The following is a") {
            let after = &s[idx + phrase.len()..];
            let end = after
                .find("<SYSTEM_MESSAGE")
                .map(|offset| idx + phrase.len() + offset)
                .unwrap_or(s.len());
            s.replace_range(start..end, "");
        }
    }

    s = RE_META.replace_all(&s, "").to_string();
    s = RE_SYS.replace_all(&s, "").to_string();
    s = RE_SETTINGS.replace_all(&s, "").to_string();
    s = RE_REQ.replace_all(&s, "").to_string();
    s = RE_FENCED_CODE.replace_all(&s, " ").to_string();
    s = RE_UNCLOSED_CODE.replace_all(&s, " ").to_string();

    s = RE_SPACES.replace_all(&s, " ").to_string();
    s = RE_NEWLINES.replace_all(&s, "\n\n").to_string();
    s = s.trim().to_string();

    if let Some((idx, _)) = s.char_indices().nth(MAX_CHUNK_CODEPOINTS) {
        s.truncate(idx);
    }

    s
}

// Stable UUID namespace for Gemini logs
const GEMINI_NAMESPACE: Uuid = Uuid::from_bytes([
    0xb4, 0x1a, 0xc8, 0xd1, 0x6e, 0x2b, 0x4a, 0x91, 0xb5, 0xd0, 0x8f, 0x1e, 0x3c, 0x7a, 0x9b,
    0x2e,
]);

use crate::watermark;

fn gemini_dirs() -> Vec<PathBuf> {
    watermark::source_dirs(".gemini/antigravity", "EXTRA_GEMINI_DIRS")
}

pub fn has_work(skip_session: Option<&str>) -> Result<bool> {
    let dirs = gemini_dirs();
    for root in &dirs {
        let brain_dir = root.join("brain");
        if !brain_dir.is_dir() {
            continue;
        }
        let wm_ts = watermark::mtime_for_dir("gemini", &dirs, root, &watermark::gemini_path());
        let entries = match fs::read_dir(&brain_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(skip) = skip_session {
                if path.file_name().is_some_and(|f| f == skip) {
                    continue;
                }
            }
            let transcript_path = path.join(".system_generated/logs/transcript.jsonl");
            let overview_path = path.join(".system_generated/logs/overview.txt");
            let target_path = if transcript_path.is_file() {
                Some(transcript_path)
            } else if overview_path.is_file() {
                Some(overview_path)
            } else {
                None
            };
            if let Some(target) = target_path {
                if watermark::file_newer_than(&target, wm_ts) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

pub fn ingest_gemini(
    db: &mut DB,
    skip_session: Option<&str>,
    quiet: bool,
) -> Result<usize> {
    let dirs = gemini_dirs();
    let wm_path = watermark::gemini_path();
    let mut count = 0;

    for root in &dirs {
        let brain_dir = root.join("brain");
        if !brain_dir.is_dir() {
            continue;
        }
        let wm_ts = watermark::mtime_for_dir("gemini", &dirs, root, &wm_path);
        let mut entries: Vec<_> = match fs::read_dir(&brain_dir) {
            Ok(e) => e.filter_map(|e| e.ok()).collect(),
            Err(_) => continue,
        };
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(skip) = skip_session {
                if path.file_name().is_some_and(|f| f == skip) {
                    continue;
                }
            }

            let transcript_path = path.join(".system_generated/logs/transcript.jsonl");
            let overview_path = path.join(".system_generated/logs/overview.txt");

            let target_path = if transcript_path.is_file() {
                Some(transcript_path)
            } else if overview_path.is_file() {
                Some(overview_path)
            } else {
                None
            };

            if let Some(target) = target_path {
                if !watermark::file_newer_than(&target, wm_ts) {
                    continue;
                }
                crate::print_ingest_path(&target, quiet);
                let mtime_ms = watermark::mtime_ms(&target);
                match parse_and_insert(db, &target, mtime_ms) {
                    Ok(n) => count += n,
                    Err(e) => {
                        log::warn!("failed to ingest gemini session {}: {e}", target.display());
                    }
                }
            }
        }
    }

    watermark::record_dirs("gemini", &dirs);
    watermark::touch(&wm_path);
    Ok(count)
}

#[derive(Clone, Debug)]
struct TurnRaw {
    role: String,
    text: String,
    timestamp: String,
    offset: u64,
    len: u64,
}

fn extract_project_name(content: &str) -> Option<String> {
    if let Some(idx) = content.find("Active Workspaces:") {
        let rest = &content[idx + "Active Workspaces:".len()..];
        for line in rest.lines() {
            let trimmed = line.trim();
            if let Some(p) = trimmed.strip_prefix("- ") {
                let p = p.trim();
                if !p.is_empty() {
                    return Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string());
                }
            }
        }
    }
    if let Some(idx) = content.find("Command Working Directory:") {
        let rest = &content[idx + "Command Working Directory:".len()..];
        if let Some(line) = rest.lines().next() {
            let p = line.trim();
            if !p.is_empty() {
                return Path::new(p)
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string());
            }
        }
    }
    None
}

pub fn parse_and_insert(db: &mut DB, path: &Path, mtime_ms: i64) -> Result<usize> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // Parent dir is session_id (.gemini/antigravity/brain/<session_id>/.system_generated/logs/transcript.jsonl)
    let session_id = path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "test_session".to_string());

    let mut interactions: Vec<Vec<TurnRaw>> = Vec::new();
    let mut current_turn_blocks: Vec<TurnRaw> = Vec::new();
    let mut detected_project: Option<String> = None;

    let mut line = String::new();
    let mut offset = 0u64;

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        let line_offset = offset;
        let line_len = bytes_read as u64;
        offset += line_len;

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // 1. Attempt Modern JSON-line Parsing
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            let msg_type = v["type"].as_str().unwrap_or("");
            let content = v["content"].as_str().unwrap_or("");
            let timestamp = v["created_at"].as_str().unwrap_or("").to_string();

            if msg_type == "USER_INPUT" {
                if detected_project.is_none() {
                    detected_project = extract_project_name(content);
                }
                if !current_turn_blocks.is_empty() {
                    interactions.push(current_turn_blocks.clone());
                    current_turn_blocks.clear();
                }
                let clean_text = sanitize_gemini_text(content);
                current_turn_blocks.push(TurnRaw {
                    role: "user".to_string(),
                    text: clean_text,
                    timestamp,
                    offset: line_offset,
                    len: line_len,
                });
            } else if msg_type == "PLANNER_RESPONSE" {
                if let Some(thinking) = v.get("thinking").and_then(|t| t.as_str()) {
                    if !thinking.is_empty() {
                        let clean_thinking = sanitize_gemini_text(thinking);
                        if !clean_thinking.is_empty() {
                            current_turn_blocks.push(TurnRaw {
                                role: "thinking".to_string(),
                                text: clean_thinking,
                                timestamp: timestamp.clone(),
                                offset: line_offset,
                                len: line_len,
                            });
                        }
                    }
                }
                if !content.is_empty() {
                    let clean_content = sanitize_gemini_text(content);
                    if !clean_content.is_empty() {
                        current_turn_blocks.push(TurnRaw {
                            role: "assistant".to_string(),
                            text: clean_content,
                            timestamp,
                            offset: line_offset,
                            len: line_len,
                        });
                    }
                }
            }
            continue;
        }

        // 2. Legacy Plain-Text Fallback
        if trimmed.starts_with("USER:") || trimmed.starts_with("MODEL:") {
            if trimmed.starts_with("USER:") && !current_turn_blocks.is_empty() {
                interactions.push(current_turn_blocks.clone());
                current_turn_blocks.clear();
            }
            if trimmed.starts_with("USER:") {
                let clean_text = sanitize_gemini_text(trimmed[5..].trim());
                current_turn_blocks.push(TurnRaw {
                    role: "user".to_string(),
                    text: clean_text,
                    timestamp: "".to_string(),
                    offset: line_offset,
                    len: line_len,
                });
            } else {
                let clean_text = sanitize_gemini_text(trimmed[6..].trim());
                current_turn_blocks.push(TurnRaw {
                    role: "assistant".to_string(),
                    text: clean_text,
                    timestamp: "".to_string(),
                    offset: line_offset,
                    len: line_len,
                });
            }
        } else if !current_turn_blocks.is_empty() {
            let last_idx = current_turn_blocks.len() - 1;
            current_turn_blocks[last_idx].text.push('\n');
            current_turn_blocks[last_idx].text.push_str(trimmed);
            current_turn_blocks[last_idx].len += line_len;
        }
    }

    if !current_turn_blocks.is_empty() {
        interactions.push(current_turn_blocks);
    }

    if interactions.is_empty() {
        return Ok(0);
    }

    let project_name = detected_project.unwrap_or_else(|| "antigravity".to_string());
    let mut count = 0;

    for (turn_idx, interaction) in interactions.iter().enumerate() {
        let last_thinking_idx = interaction.iter().rposition(|t| t.role == "thinking");

        let mut eligible_turns: Vec<TurnRaw> = Vec::new();
        for (idx, turn) in interaction.iter().enumerate() {
            if turn.role == "thinking" {
                if Some(idx) != last_thinking_idx {
                    continue;
                }
                let mut t = turn.clone();
                if let Some((char_idx, _)) = t.text.char_indices().nth(800) {
                    t.text.truncate(char_idx);
                }
                eligible_turns.push(t);
            } else {
                let mut t = turn.clone();
                if t.text.chars().count() > MAX_CHUNK_CODEPOINTS {
                    if let Some((char_idx, _)) = t.text.char_indices().nth(MAX_CHUNK_CODEPOINTS) {
                        t.text.truncate(char_idx);
                    }
                }
                eligible_turns.push(t);
            }
        }

        let header = format!("[gemini:{project_name}] Session {}\n", session_id);
        let mut all_parts = vec![header];
        let mut turns_meta = Vec::new();
        let mut total_chars: usize = all_parts[0].chars().count();

        for turn in eligible_turns {
            if total_chars >= MAX_TURN_CODEPOINTS {
                break;
            }

            let label = match turn.role.as_str() {
                "user" => "[User]",
                "thinking" => "[Thinking]",
                _ => "[Gemini]",
            };

            let mut part = format!("{label} {}\n", turn.text);
            let part_chars = part.chars().count();
            let remaining = MAX_TURN_CODEPOINTS - total_chars;

            if part_chars > remaining {
                if let Some((idx, _)) = part.char_indices().nth(remaining) {
                    part.truncate(idx);
                }
            }

            let final_part_chars = part.chars().count();
            if final_part_chars > 0 {
                total_chars += final_part_chars;
                all_parts.push(part);
                turns_meta.push(serde_json::json!({
                    "role": turn.role,
                    "timestamp": turn.timestamp,
                    "off": turn.offset,
                    "len": turn.len,
                }));
            }
        }

        let body = all_parts.join("");
        let lengths: Vec<usize> = all_parts.iter().map(|p| p.chars().count()).collect();

        let uuid = uuid::Uuid::new_v5(
            &GEMINI_NAMESPACE,
            format!("{session_id}:{turn_idx}").as_bytes(),
        );
        let metadata = serde_json::json!({
            "project": project_name,
            "session_id": session_id.to_string(),
            "turn": turn_idx,
            "path": path.to_string_lossy(),
            "mtime_ms": mtime_ms,
            "turns": turns_meta,
            "source": "gemini",
        })
        .to_string();

        let date = interaction
            .first()
            .and_then(|t| iso8601_timestamp::Timestamp::parse(&t.timestamp));

        db.add_doc(None, &uuid, date, &metadata, &body, Some(lengths))?;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_gemini_overview() {
        let raw_log = "USER: How do I fix the podman error?\nMODEL: You need to adjust the permissions on the root directory.";
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", raw_log).unwrap();

        let mut db = DB::new(tempfile::tempdir().unwrap().path().join("test.db")).unwrap();
        let inserted = parse_and_insert(&mut db, temp_file.path(), 0).unwrap();
        assert_eq!(inserted, 1);
    }

    #[test]
    fn test_parse_gemini_json() {
        let raw_log = r#"{"type": "USER_INPUT", "content": "How do I fix the podman error?", "created_at": "2026-04-26T11:00:00Z"}
{"type": "PLANNER_RESPONSE", "content": "You need to adjust the permissions.", "created_at": "2026-04-26T11:00:01Z"}"#;
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", raw_log).unwrap();

        let mut db =
            DB::new(tempfile::tempdir().unwrap().path().join("test_json.db")).unwrap();
        let inserted = parse_and_insert(&mut db, temp_file.path(), 0).unwrap();
        assert_eq!(inserted, 1);
    }

    #[test]
    fn test_parse_gemini_json_multi_turn() {
        let raw_log = r#"{"type": "USER_INPUT", "content": "Turn 1", "created_at": "2026-04-26T11:00:00Z"}
{"type": "PLANNER_RESPONSE", "content": "Response 1", "created_at": "2026-04-26T11:00:01Z"}
{"type": "USER_INPUT", "content": "Turn 2", "created_at": "2026-04-26T11:05:00Z"}
{"type": "PLANNER_RESPONSE", "content": "Response 2", "created_at": "2026-04-26T11:05:01Z"}"#;
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", raw_log).unwrap();

        let mut db = DB::new(
            tempfile::tempdir()
                .unwrap()
                .path()
                .join("test_json_multi.db"),
        )
        .unwrap();
        let inserted = parse_and_insert(&mut db, temp_file.path(), 0).unwrap();
        assert_eq!(inserted, 2);
    }

    #[test]
    fn test_parse_gemini_workspace_extraction() {
        let raw_log = r#"{"type": "USER_INPUT", "content": "Command Working Directory: /home/ch/homelab\nCheck the status", "created_at": "2026-04-26T11:00:00Z"}
{"type": "PLANNER_RESPONSE", "content": "All systems nominal.", "created_at": "2026-04-26T11:00:01Z"}"#;
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", raw_log).unwrap();

        let mut db = DB::new(
            tempfile::tempdir()
                .unwrap()
                .path()
                .join("test_ws.db"),
        )
        .unwrap();
        let inserted = parse_and_insert(&mut db, temp_file.path(), 0).unwrap();
        assert_eq!(inserted, 1);
    }

    #[test]
    fn test_parse_gemini_thinking_label() {
        let raw_log = r#"{"type": "USER_INPUT", "content": "How do I fix the podman error?", "created_at": "2026-04-26T11:00:00Z"}
{"type": "PLANNER_RESPONSE", "content": "", "thinking": "Let me investigate podman logs.", "created_at": "2026-04-26T11:00:01Z"}
{"type": "PLANNER_RESPONSE", "content": "You need to adjust permissions.", "created_at": "2026-04-26T11:00:02Z"}"#;
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", raw_log).unwrap();

        let mut db =
            DB::new(tempfile::tempdir().unwrap().path().join("test_thinking.db")).unwrap();
        let inserted = parse_and_insert(&mut db, temp_file.path(), 0).unwrap();
        assert_eq!(inserted, 1);

        let mut stmt = db.query("SELECT body FROM document").unwrap();
        let mut rows = stmt.query(()).unwrap();
        let row = rows.next().unwrap().unwrap();
        let body: String = row.get(0).unwrap();
        assert!(body.contains("[User] How do I fix the podman error?"));
        assert!(body.contains("[Thinking] Let me investigate podman logs."));
        assert!(body.contains("[Gemini] You need to adjust permissions."));
    }

    #[test]
    fn test_sanitize_gemini_text_strips_metadata_and_system_preserves_request() {
        let input = "The following is a <SYSTEM_MESSAGE> not actually sent by the user. It is provided by the system as important information to pay attention to.\n\n\
                     <SYSTEM_MESSAGE>\n[Message] timestamp=2026-08-26T14:34:07Z sender=system content=Server restart alert.\n</SYSTEM_MESSAGE>\n\
                     <ADDITIONAL_METADATA>\nActive Document: /home/ch/homelab/README.md\nCursor line: 1\n</ADDITIONAL_METADATA>\n\
                     <USER_SETTINGS_CHANGE>\nModel set to Gemini Flash\n</USER_SETTINGS_CHANGE>\n\
                     <USER_REQUEST>\nPlease fix the podman error on host 10.\nHere is some code:\n```bash\npodman ps -a\n```\nAll done.\n</USER_REQUEST>";

        let sanitized = sanitize_gemini_text(input);

        assert!(!sanitized.contains("Server restart alert"));
        assert!(!sanitized.contains("<SYSTEM_MESSAGE>"));
        assert!(!sanitized.contains("</SYSTEM_MESSAGE>"));
        assert!(!sanitized.contains("Active Document"));
        assert!(!sanitized.contains("<ADDITIONAL_METADATA>"));
        assert!(!sanitized.contains("</ADDITIONAL_METADATA>"));
        assert!(!sanitized.contains("Model set to Gemini Flash"));
        assert!(!sanitized.contains("<USER_SETTINGS_CHANGE>"));
        assert!(!sanitized.contains("</USER_SETTINGS_CHANGE>"));
        assert!(!sanitized.contains("<USER_REQUEST>"));
        assert!(!sanitized.contains("</USER_REQUEST>"));
        assert!(!sanitized.contains("podman ps -a"));
        assert!(sanitized.contains("Please fix the podman error on host 10."));
        assert!(sanitized.contains("Here is some code:"));
        assert!(sanitized.contains("All done."));

        // Long text truncation check
        let long_input = "x".repeat(5000);
        let long_sanitized = sanitize_gemini_text(&long_input);
        assert_eq!(long_sanitized.chars().count(), MAX_CHUNK_CODEPOINTS);
    }

    #[test]
    fn test_parse_and_insert_caps_oversized_turn_preserves_labels() {
        let user_content = format!("<USER_REQUEST>{}</USER_REQUEST>", "U".repeat(5000));
        let thinking_1 = "T1".repeat(1000);
        let thinking_2 = "T2".repeat(1000);
        let gemini_content = "G".repeat(5000);

        let raw_log = format!(
            r#"{{"type": "USER_INPUT", "content": "{user_content}", "created_at": "2026-04-26T11:00:00Z"}}
{{"type": "PLANNER_RESPONSE", "content": "", "thinking": "{thinking_1}", "created_at": "2026-04-26T11:00:01Z"}}
{{"type": "PLANNER_RESPONSE", "content": "", "thinking": "{thinking_2}", "created_at": "2026-04-26T11:00:02Z"}}
{{"type": "PLANNER_RESPONSE", "content": "{gemini_content}", "created_at": "2026-04-26T11:00:03Z"}}"#
        );

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", raw_log).unwrap();

        let mut db =
            DB::new(tempfile::tempdir().unwrap().path().join("test_oversized.db")).unwrap();
        let inserted = parse_and_insert(&mut db, temp_file.path(), 0).unwrap();
        assert_eq!(inserted, 1);

        let mut stmt = db.query("SELECT body FROM document").unwrap();
        let mut rows = stmt.query(()).unwrap();
        let row = rows.next().unwrap().unwrap();
        let body: String = row.get(0).unwrap();

        assert!(
            body.chars().count() <= MAX_TURN_CODEPOINTS,
            "body length {} exceeds MAX_TURN_CODEPOINTS {}",
            body.chars().count(),
            MAX_TURN_CODEPOINTS
        );
        assert!(body.contains("[User]"));
        assert!(body.contains("[Thinking]"));
        assert!(body.contains("[Gemini]"));
        // First thinking block should have been dropped
        assert!(!body.contains("T1T1T1"));
        // Last thinking block should be preserved (truncated to <= 800 chars)
        assert!(body.contains("T2T2T2"));
    }
}
