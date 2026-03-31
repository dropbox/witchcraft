use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use text_splitter::TextSplitter;
use uuid::Uuid;

use crate::DB;

const MIN_CHUNK_CODEPOINTS: usize = 5;
const MAX_CHUNK_CODEPOINTS: usize = 4000;
const MAX_TITLE_CODEPOINTS: usize = 240;

// Stable UUID namespace for Claude Code sessions
const CLAUDE_CODE_NAMESPACE: Uuid = Uuid::from_bytes([
    0xa3, 0xf7, 0xc8, 0xd1, 0x6e, 0x2b, 0x4a, 0x91, 0xb5, 0xd0, 0x8f, 0x1e, 0x3c, 0x7a, 0x9b,
    0x2d,
]);

#[derive(Deserialize)]
struct SessionEntry {
    #[serde(rename = "type")]
    entry_type: String,
    timestamp: Option<String>,
    message: Option<Message>,
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
}

struct Chunk {
    role: String,
    text: String,
    timestamp: String,
    ts_ms: i64,
}

fn codepoint_len(s: &str) -> usize {
    s.chars().count()
}

fn truncate_codepoints(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn extract_text(content: &Content) -> Option<String> {
    match content {
        Content::Text(s) => Some(s.clone()),
        Content::Blocks(blocks) => {
            let texts: Vec<&str> = blocks
                .iter()
                .filter(|b| b.block_type == "text")
                .filter_map(|b| b.text.as_deref())
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
    let s = strip_system_content(text);
    let s = strip_code(&s);
    let s = strip_tables(&s);
    compact(&s)
}

fn strip_system_content(text: &str) -> String {
    let mut s = text.to_string();

    // Strip XML blocks whose tag name contains a hyphen or colon (system/namespace tags).
    // The regex crate doesn't support backreferences, so we find opening tags and match
    // to their closing tags manually.
    let re_open = Regex::new(r"<([a-z][a-z0-9]*(?:[-:][a-z0-9_]+)+)[\s>]").unwrap();
    loop {
        let m = match re_open.find(&s) {
            Some(m) => m,
            None => break,
        };
        let start = m.start();
        // Extract tag name (capture group 1)
        let caps = re_open.captures(&s[start..]).unwrap();
        let tag_name = caps.get(1).unwrap().as_str().to_string();
        let close_tag = format!("</{tag_name}>");

        // Find end of opening tag
        let after_open = match s[start..].find('>') {
            Some(i) => start + i + 1,
            None => break,
        };
        // Find closing tag
        let end = match s[after_open..].find(&close_tag) {
            Some(i) => after_open + i + close_tag.len(),
            None => {
                // Self-closing or unclosed — remove just the opening tag line
                let line_end = s[start..].find('\n').map(|i| start + i + 1).unwrap_or(s.len());
                s.replace_range(start..line_end, "");
                continue;
            }
        };
        s.replace_range(start..end, "");
    }

    // Strip [Request interrupted by user] variants
    let re = Regex::new(r"\[Request interrupted by user[^\]]*\]").unwrap();
    s = re.replace_all(&s, "").to_string();
    // Strip session continuation preamble
    if let Some(idx) = s.find("This session is being continued from a previous conversation") {
        s.truncate(idx);
    }
    s
}

fn strip_code(text: &str) -> String {
    // Fenced code blocks (closed)
    let re = Regex::new(r"```[\s\S]*?```").unwrap();
    let s = re.replace_all(text, " ").to_string();
    // Fenced code blocks (unclosed at EOF)
    let re = Regex::new(r"```[\s\S]*$").unwrap();
    let s = re.replace_all(&s, " ").to_string();
    // Inline code
    let re = Regex::new(r"`[^`]*`").unwrap();
    re.replace_all(&s, " ").to_string()
}

fn strip_tables(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with('|') && trimmed.ends_with('|'))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn compact(text: &str) -> String {
    let re = Regex::new(r"\s{2,}").unwrap();
    re.replace_all(text, " ").trim().to_string()
}

fn parse_session_file(path: &Path) -> Vec<Chunk> {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let mut chunks = Vec::new();

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let entry: SessionEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.entry_type != "user" && entry.entry_type != "assistant" {
            continue;
        }

        let msg = match &entry.message {
            Some(m) => m,
            None => continue,
        };

        let role = match &msg.role {
            Some(r) if r == "user" || r == "assistant" => r.clone(),
            _ => continue,
        };

        let content = match &msg.content {
            Some(c) => c,
            None => continue,
        };

        let raw_text = match extract_text(content) {
            Some(t) => t,
            None => continue,
        };

        let text = sanitize(&raw_text);
        if text.is_empty() {
            continue;
        }

        let cp_len = codepoint_len(&text);
        if cp_len < MIN_CHUNK_CODEPOINTS || cp_len > MAX_CHUNK_CODEPOINTS {
            continue;
        }

        let timestamp = match &entry.timestamp {
            Some(ts) if !ts.is_empty() => ts.clone(),
            _ => continue,
        };

        let ts_ms = chrono::DateTime::parse_from_rfc3339(&timestamp)
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(0);

        if ts_ms <= 0 {
            continue;
        }

        chunks.push(Chunk {
            role,
            text,
            timestamp,
            ts_ms,
        });
    }

    chunks.sort_by_key(|c| c.ts_ms);
    chunks
}

fn decode_project_name(dir_name: &str) -> String {
    dir_name.replace('-', "/").trim_start_matches('/').to_string()
}

fn ingest_session(db: &mut DB, path: &Path, project_name: &str, mtime_ms: i64) -> Result<usize> {
    let chunks = parse_session_file(path);
    if chunks.is_empty() {
        return Ok(0);
    }

    let session_id = path.file_stem().unwrap().to_string_lossy();
    let splitter = TextSplitter::new(300);

    // Split into interactions: each starts at a user message
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
        let title_chunk = interaction
            .iter()
            .find(|c| c.role == "user")
            .unwrap_or(&interaction[0]);
        let title = truncate_codepoints(&title_chunk.text, MAX_TITLE_CODEPOINTS);

        // Build labeled lines, then split each with text-splitter for sub-chunk lengths
        let mut all_parts = Vec::new();
        for chunk in *interaction {
            let label = if chunk.role == "user" {
                "[User]"
            } else {
                "[Claude]"
            };
            let line = format!("{label} {}", chunk.text);
            let parts: Vec<String> = splitter.chunks(&line).map(|c| format!("{c}\n")).collect();
            all_parts.extend(parts);
        }

        let lengths: Vec<usize> = all_parts.iter().map(|p| p.chars().count()).collect();
        let body = all_parts.join("");

        if body.trim().is_empty() {
            continue;
        }

        let uuid = Uuid::new_v5(
            &CLAUDE_CODE_NAMESPACE,
            format!("{session_id}:{turn_idx}").as_bytes(),
        );

        let metadata = serde_json::json!({
            "title": title,
            "source": "claude_code",
            "project": project_name,
            "session_id": session_id.to_string(),
            "turn": turn_idx,
            "path": path.to_string_lossy(),
            "mtime_ms": mtime_ms,
        })
        .to_string();

        let date = iso8601_timestamp::Timestamp::parse(&interaction[0].timestamp);
        db.add_doc(&uuid, date, &metadata, &body, Some(lengths))?;
        count += 1;
    }

    Ok(count)
}

fn ingest_memory_file(db: &mut DB, path: &Path, project_name: &str, mtime_ms: i64) -> Result<bool> {
    let raw = fs::read_to_string(path)?;
    if raw.trim().is_empty() {
        return Ok(false);
    }

    let filename = path.file_name().unwrap().to_string_lossy();
    let uuid = Uuid::new_v5(
        &CLAUDE_CODE_NAMESPACE,
        format!("memory:{project_name}:{filename}").as_bytes(),
    );

    // Strip YAML frontmatter
    let body_text = if raw.starts_with("---\n") {
        if let Some(end) = raw[4..].find("\n---\n") {
            raw[4 + end + 5..].to_string()
        } else {
            raw.clone()
        }
    } else {
        raw.clone()
    };

    let splitter = TextSplitter::new(500);
    let bodies: Vec<String> = splitter
        .chunks(&body_text)
        .map(|c| format!("{c}\n"))
        .collect();
    let lengths: Vec<usize> = bodies.iter().map(|b| b.chars().count()).collect();
    let body = bodies.join("");

    if body.trim().is_empty() {
        return Ok(false);
    }

    let title = truncate_codepoints(
        &filename.trim_end_matches(".md"),
        MAX_TITLE_CODEPOINTS,
    );

    let metadata = serde_json::json!({
        "title": title,
        "source": "claude_code_memory",
        "project": project_name,
        "path": path.to_string_lossy(),
        "mtime_ms": mtime_ms,
    })
    .to_string();

    db.add_doc(&uuid, None, &metadata, &body, Some(lengths))?;
    Ok(true)
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

fn load_watermarks(db: &DB) -> HashMap<String, i64> {
    let mut watermarks = HashMap::new();
    let mut stmt = match db.query(
        "SELECT json_extract(metadata, '$.path'), max(json_extract(metadata, '$.mtime_ms'))
         FROM document
         WHERE json_extract(metadata, '$.source') IN ('claude_code', 'claude_code_memory')
           AND json_extract(metadata, '$.mtime_ms') IS NOT NULL
         GROUP BY json_extract(metadata, '$.path')",
    ) {
        Ok(s) => s,
        Err(_) => return watermarks,
    };
    let rows = stmt.query_map((), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    });
    if let Ok(rows) = rows {
        for row in rows.flatten() {
            watermarks.insert(row.0, row.1);
        }
    }
    watermarks
}

pub fn ingest_claude_code(db: &mut DB) -> Result<(usize, usize)> {
    let home = std::env::var("HOME").unwrap_or_default();
    let projects_dir = PathBuf::from(&home).join(".claude/projects");

    if !projects_dir.is_dir() {
        println!("no Claude Code projects found at {}", projects_dir.display());
        return Ok((0, 0));
    }

    let watermarks = load_watermarks(db);

    let mut session_count = 0usize;
    let mut memory_count = 0usize;

    let mut entries: Vec<_> = fs::read_dir(&projects_dir)?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let dir_path = entry.path();
        if !dir_path.is_dir() {
            continue;
        }

        let dir_name = entry.file_name().to_string_lossy().to_string();
        let project_name = decode_project_name(&dir_name);

        // Ingest .jsonl session files
        let mut jsonl_files: Vec<PathBuf> = fs::read_dir(&dir_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        jsonl_files.sort();

        for jsonl_path in &jsonl_files {
            let path_str = jsonl_path.to_string_lossy().to_string();
            let mtime_ms = file_mtime_ms(jsonl_path).unwrap_or(0);
            if let Some(&prev_ms) = watermarks.get(&path_str) {
                if mtime_ms <= prev_ms {
                    continue;
                }
            }
            match ingest_session(db, jsonl_path, &project_name, mtime_ms) {
                Ok(n) => session_count += n,
                Err(e) => {
                    log::warn!("failed to ingest {}: {e}", jsonl_path.display());
                }
            }
        }

        // Ingest memory files
        let memory_dir = dir_path.join("memory");
        if memory_dir.is_dir() {
            let mut md_files: Vec<PathBuf> = fs::read_dir(&memory_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "md"))
                .collect();
            md_files.sort();

            for md_path in &md_files {
                let path_str = md_path.to_string_lossy().to_string();
                let mtime_ms = file_mtime_ms(md_path).unwrap_or(0);
                if let Some(&prev_ms) = watermarks.get(&path_str) {
                    if mtime_ms <= prev_ms {
                        continue;
                    }
                }
                match ingest_memory_file(db, md_path, &project_name, mtime_ms) {
                    Ok(true) => memory_count += 1,
                    Ok(false) => {}
                    Err(e) => {
                        log::warn!("failed to ingest memory {}: {e}", md_path.display());
                    }
                }
            }
        }
    }

    Ok((session_count, memory_count))
}
