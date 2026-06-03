use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use witchcraft::DB;

const MIN_CHUNK_CODEPOINTS: usize = 5;
const MAX_CHUNK_CODEPOINTS: usize = 4000;

// Stable UUID namespace for Pi sessions.
const PI_NAMESPACE: Uuid = Uuid::from_bytes([
    0x70, 0x69, 0x63, 0x6b, 0x62, 0x72, 0x61, 0x69, 0x6e, 0x2d, 0x70, 0x69, 0x2d, 0x76, 0x31, 0x00,
]);

#[derive(Deserialize)]
struct Entry {
    #[serde(rename = "type")]
    entry_type: String,
    timestamp: Option<String>,
    id: Option<String>,
    cwd: Option<String>,
    message: Option<Message>,
    name: Option<String>,
    content: Option<Content>,
}

#[derive(Deserialize)]
struct Message {
    role: Option<String>,
    content: Option<Content>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
}

struct Chunk {
    role: String,
    text: String,
    timestamp: String,
    ts_ms: i64,
    byte_offset: u64,
    byte_len: u64,
}

struct SessionMeta {
    id: Option<String>,
    cwd: Option<String>,
    name: Option<String>,
}

fn codepoint_len(s: &str) -> usize {
    s.chars().count()
}

fn extract_text(content: &Content) -> Option<String> {
    match content {
        Content::Text(s) => Some(s.clone()),
        Content::Blocks(blocks) => {
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b.block_type.as_str() {
                    "text" => b.text.as_deref(),
                    "thinking" => b.thinking.as_deref(),
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
    }
}

fn sanitize(text: &str) -> String {
    let s = strip_code(text);
    let s = strip_tool_signatures(&s);
    compact(&s)
}

fn strip_code(text: &str) -> String {
    let re = Regex::new(r"```[\s\S]*?```").unwrap();
    let s = re.replace_all(text, " ").to_string();
    let re = Regex::new(r"```[\s\S]*$").unwrap();
    let s = re.replace_all(&s, " ").to_string();
    let re = Regex::new(r"`[^`]*`").unwrap();
    re.replace_all(&s, " ").to_string()
}

fn strip_tool_signatures(text: &str) -> String {
    // Pi stores provider-specific signatures beside assistant text/thinking in sibling
    // fields, not in the text itself. This catches pasted/exported variants if present.
    let re = Regex::new(r#"\{\"v\":\d+,\"id\":\"[^\"]+\"[\s\S]*?\}"#).unwrap();
    re.replace_all(text, " ").to_string()
}

fn compact(text: &str) -> String {
    let re = Regex::new(r"\s{2,}").unwrap();
    re.replace_all(text, " ").trim().to_string()
}

fn parse_session_file(path: &Path) -> (SessionMeta, Vec<Chunk>) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => {
            return (
                SessionMeta {
                    id: None,
                    cwd: None,
                    name: None,
                },
                vec![],
            );
        }
    };

    let mut meta = SessionMeta {
        id: None,
        cwd: None,
        name: None,
    };
    let mut chunks = Vec::new();
    let mut offset: u64 = 0;

    for line in raw.lines() {
        let line_offset = offset;
        offset += line.len() as u64 + 1; // +1 for newline
        if line.trim().is_empty() {
            continue;
        }

        let entry: Entry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        match entry.entry_type.as_str() {
            "session" => {
                if meta.id.is_none() {
                    meta.id = entry.id;
                }
                if meta.cwd.is_none() {
                    meta.cwd = entry.cwd;
                }
                continue;
            }
            "session_info" => {
                if let Some(name) = entry.name {
                    meta.name = Some(name);
                }
                continue;
            }
            "custom_message" => {
                let raw_text = match entry.content.as_ref().and_then(extract_text) {
                    Some(t) => t,
                    None => continue,
                };
                push_chunk(
                    &mut chunks,
                    "assistant".to_string(),
                    raw_text,
                    entry.timestamp,
                    line_offset,
                    line.len() as u64,
                );
            }
            "message" => {
                let msg = match entry.message {
                    Some(m) => m,
                    None => continue,
                };
                let role = match msg.role.as_deref() {
                    Some("user") => "user".to_string(),
                    Some("assistant") => "assistant".to_string(),
                    _ => continue,
                };
                let raw_text = match msg.content.as_ref().and_then(extract_text) {
                    Some(t) => t,
                    None => continue,
                };
                push_chunk(
                    &mut chunks,
                    role,
                    raw_text,
                    entry.timestamp,
                    line_offset,
                    line.len() as u64,
                );
            }
            _ => continue,
        }
    }

    chunks.sort_by_key(|c| c.ts_ms);
    (meta, chunks)
}

fn push_chunk(
    chunks: &mut Vec<Chunk>,
    role: String,
    raw_text: String,
    timestamp: Option<String>,
    byte_offset: u64,
    byte_len: u64,
) {
    let text = sanitize(&raw_text);
    if text.is_empty() {
        return;
    }
    let cp_len = codepoint_len(&text);
    if !(MIN_CHUNK_CODEPOINTS..=MAX_CHUNK_CODEPOINTS).contains(&cp_len) {
        return;
    }

    let timestamp = match timestamp {
        Some(ts) if !ts.is_empty() => ts,
        _ => return,
    };
    let ts_ms = chrono::DateTime::parse_from_rfc3339(&timestamp)
        .map(|dt| dt.timestamp_millis())
        .unwrap_or(0);
    if ts_ms <= 0 {
        return;
    }

    chunks.push(Chunk {
        role,
        text,
        timestamp,
        ts_ms,
        byte_offset,
        byte_len,
    });
}

fn project_from_cwd(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| cwd.to_string())
}

pub fn session_id_from_filename(path: &Path) -> String {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    // Filename: <timestamp>_<UUID>
    if let Some((_, id)) = stem.rsplit_once('_') {
        if id.chars().filter(|&c| c == '-').count() == 4 {
            return id.to_string();
        }
    }
    stem.to_string()
}

fn ingest_session(db: &mut DB, path: &Path, mtime_ms: i64) -> Result<usize> {
    let (meta, chunks) = parse_session_file(path);
    if chunks.is_empty() {
        return Ok(0);
    }

    let project_name = meta
        .cwd
        .as_deref()
        .map(project_from_cwd)
        .unwrap_or_default();
    let session_id = meta.id.unwrap_or_else(|| session_id_from_filename(path));
    let session_title = meta.name.unwrap_or_else(|| {
        chunks
            .iter()
            .find(|c| c.role == "user")
            .map(|c| c.text.chars().take(240).collect())
            .unwrap_or_default()
    });

    // Split into interactions starting at each user message.
    let mut interactions: Vec<&[Chunk]> = Vec::new();
    let mut start = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.role == "user" && i > start {
            interactions.push(&chunks[start..i]);
            start = i;
        }
    }
    interactions.push(&chunks[start..]);

    let mut count = 0;
    for (turn_idx, interaction) in interactions.iter().enumerate() {
        let header = format!("[pi:{project_name}] {session_title}\n");
        let mut all_parts = vec![header];
        let mut turns_meta: Vec<serde_json::Value> = Vec::new();

        for chunk in *interaction {
            let label = if chunk.role == "user" {
                "[User]"
            } else {
                "[Pi]"
            };
            all_parts.push(format!("{label} {}\n", chunk.text));
            turns_meta.push(serde_json::json!({
                "role": chunk.role,
                "timestamp": chunk.timestamp,
                "off": chunk.byte_offset,
                "len": chunk.byte_len,
            }));
        }

        let lengths: Vec<usize> = all_parts.iter().map(|p| p.chars().count()).collect();
        let body = all_parts.join("");
        if body.trim().is_empty() {
            continue;
        }

        let uuid = Uuid::new_v5(&PI_NAMESPACE, format!("{session_id}:{turn_idx}").as_bytes());
        let metadata = serde_json::json!({
            "source": "pi",
            "project": project_name,
            "session_id": session_id,
            "session_name": session_title,
            "turn": turn_idx,
            "path": path.to_string_lossy(),
            "cwd": meta.cwd,
            "mtime_ms": mtime_ms,
            "turns": turns_meta,
            "branch": null,
        })
        .to_string();

        let date = iso8601_timestamp::Timestamp::parse(&interaction[0].timestamp);
        db.add_doc(&uuid, date, &metadata, &body, Some(lengths))?;
        count += 1;
    }

    Ok(count)
}

fn file_mtime_ms(path: &Path) -> Option<i64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as i64)
}

use crate::watermark;

fn sessions_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".pi/agent/sessions")
}

fn collect_session_files(base: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|ext| ext == "jsonl") {
                out.push(p);
            }
        }
    }
    walk(base, &mut files);
    files.sort();
    files
}

fn matches_skip(path: &Path, skip_session: Option<&str>) -> bool {
    let Some(skip) = skip_session else {
        return false;
    };
    if skip.is_empty() {
        return false;
    }
    if path.to_string_lossy() == skip {
        return true;
    }
    session_id_from_filename(path) == skip
}

pub fn ingest_pi(db: &mut DB, skip_session: Option<&str>) -> Result<usize> {
    let dir = sessions_dir();
    if !dir.is_dir() {
        return Ok(0);
    }

    let wm_path = watermark::pi_path();
    let wm_ts = watermark::mtime_ms(&wm_path);
    let mut session_count = 0usize;
    let mut seen = HashSet::new();

    for jsonl_path in collect_session_files(&dir) {
        if !watermark::file_newer_than(&jsonl_path, wm_ts) {
            continue;
        }
        if matches_skip(&jsonl_path, skip_session) {
            continue;
        }
        if !seen.insert(jsonl_path.clone()) {
            continue;
        }
        let mtime_ms = file_mtime_ms(&jsonl_path).unwrap_or(0);
        crate::print_ingest_path(&jsonl_path);
        match ingest_session(db, &jsonl_path, mtime_ms) {
            Ok(n) => session_count += n,
            Err(e) => {
                log::warn!("failed to ingest pi {}: {e}", jsonl_path.display());
            }
        }
    }

    watermark::touch(&wm_path);
    Ok(session_count)
}

pub fn has_work(skip_session: Option<&str>) -> bool {
    let dir = sessions_dir();
    if !dir.is_dir() {
        return false;
    }

    let wm_ts = watermark::mtime_ms(&watermark::pi_path());
    collect_session_files(&dir)
        .iter()
        .any(|path| !matches_skip(path, skip_session) && watermark::file_newer_than(path, wm_ts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tempfile(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_pi_session() {
        let content = format!(
            "{}\n{}\n{}\n{}\n",
            r#"{"type":"session","version":3,"id":"abc","timestamp":"2026-01-01T00:00:00.000Z","cwd":"/Users/me/project"}"#,
            r#"{"type":"session_info","id":"n","parentId":null,"timestamp":"2026-01-01T00:00:00.000Z","name":"Named Session"}"#,
            r#"{"type":"message","id":"u","parentId":null,"timestamp":"2026-01-01T00:00:01.000Z","message":{"role":"user","content":[{"type":"text","text":"hello from user"}]}}"#,
            r#"{"type":"message","id":"a","parentId":"u","timestamp":"2026-01-01T00:00:02.000Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"thinking about it"},{"type":"text","text":"hello back"}]}}"#,
        );
        let f = write_tempfile(&content);
        let (meta, chunks) = parse_session_file(f.path());
        assert_eq!(meta.id.as_deref(), Some("abc"));
        assert_eq!(meta.cwd.as_deref(), Some("/Users/me/project"));
        assert_eq!(meta.name.as_deref(), Some("Named Session"));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].role, "user");
        assert!(chunks[1].text.contains("hello back"));
    }

    #[test]
    fn test_session_id_from_filename() {
        let p =
            PathBuf::from("2026-06-03T12-19-44-826Z_019e8d6c-e63a-7e52-8290-f77811e9fdac.jsonl");
        assert_eq!(
            session_id_from_filename(&p),
            "019e8d6c-e63a-7e52-8290-f77811e9fdac"
        );
    }
}
