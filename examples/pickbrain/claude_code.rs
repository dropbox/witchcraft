use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use text_splitter::MarkdownSplitter;
use uuid::Uuid;

use witchcraft::DB;

const MIN_CHUNK_CODEPOINTS: usize = 5;
const MAX_CHUNK_CODEPOINTS: usize = 4000;
const MD_SECTION_MAX: usize = 1500;

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
    #[serde(rename = "gitBranch")]
    git_branch: Option<String>,
    #[serde(rename = "customTitle")]
    custom_title: Option<String>,
    cwd: Option<String>,
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
    name: Option<String>,
    #[serde(default)]
    input: Option<ToolInput>,
}

#[derive(Deserialize)]
struct ToolInput {
    #[serde(default)]
    file_path: Option<String>,
}

struct Chunk {
    role: String,
    text: String,
    timestamp: String,
    ts_ms: i64,
    byte_offset: u64,
    byte_len: u64,
    git_branch: Option<String>,
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

struct SessionInfo {
    custom_title: Option<String>,
    cwd: Option<String>,
}

fn parse_session_file(path: &Path) -> (SessionInfo, Vec<Chunk>) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return (SessionInfo { custom_title: None, cwd: None }, vec![]),
    };

    let mut chunks = Vec::new();
    let mut info = SessionInfo { custom_title: None, cwd: None };
    let mut offset: u64 = 0;

    for line in raw.lines() {
        let line_offset = offset;
        offset += line.len() as u64 + 1; // +1 for newline
        if line.trim().is_empty() {
            continue;
        }

        let entry: SessionEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.entry_type == "custom-title" {
            if let Some(ref title) = entry.custom_title {
                info.custom_title = Some(title.clone());
            }
            continue;
        }

        // Use the cwd from JSONL entries (authoritative) rather than decoding
        // from the directory name, which is ambiguous when paths contain dashes.
        if info.cwd.is_none() {
            if let Some(ref cwd) = entry.cwd {
                info.cwd = Some(cwd.clone());
            }
        }

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
            byte_offset: line_offset,
            byte_len: line.len() as u64,
            git_branch: entry.git_branch,
        });
    }

    chunks.sort_by_key(|c| c.ts_ms);
    (info, chunks)
}

fn decode_project_name(dir_name: &str) -> String {
    dir_name.replace('-', "/").trim_start_matches('/').to_string()
}

fn ingest_session(db: &mut DB, path: &Path, project_name: &str, mtime_ms: i64) -> Result<usize> {
    let (info, chunks) = parse_session_file(path);
    if chunks.is_empty() {
        return Ok(0);
    }

    // Prefer the cwd from inside the JSONL (authoritative) over the decoded
    // directory name, which is lossy when paths contain dashes.
    let project_name = info
        .cwd
        .as_deref()
        .map(|cwd| cwd.trim_start_matches('/').to_string())
        .unwrap_or_else(|| project_name.to_string());

    let session_id = path.file_stem().unwrap().to_string_lossy();

    // Session title: custom name if set, otherwise first user message
    let session_title: String = info.custom_title.unwrap_or_else(|| {
        chunks
            .iter()
            .find(|c| c.role == "user")
            .map(|c| c.text.chars().take(240).collect())
            .unwrap_or_default()
    });

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
        // Header line with project and session context
        let header = format!("[{project_name}] {session_title}\n");
        // One entry per turn so sub_idx maps 1:1 to conversation turns
        let mut all_parts = vec![header];
        let mut turns_meta: Vec<serde_json::Value> = Vec::new();

        for chunk in *interaction {
            let label = if chunk.role == "user" {
                "[User]"
            } else {
                "[Claude]"
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

        let uuid = Uuid::new_v5(
            &CLAUDE_CODE_NAMESPACE,
            format!("{session_id}:{turn_idx}").as_bytes(),
        );

        let branch: Option<&str> = interaction
            .iter()
            .filter_map(|c| c.git_branch.as_deref())
            .last();

        let metadata = serde_json::json!({
            "source": "claude",
            "project": project_name,
            "session_id": session_id.to_string(),
            "session_name": session_title,
            "turn": turn_idx,
            "path": path.to_string_lossy(),
            "cwd": info.cwd,
            "mtime_ms": mtime_ms,
            "turns": turns_meta,
            "branch": branch,
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

    let header = format!("[{project_name}] {}\n", filename.trim_end_matches(".md"));
    let mut bodies = vec![header];
    bodies.extend(split_markdown(&body_text));
    let lengths: Vec<usize> = bodies.iter().map(|b| b.chars().count()).collect();
    let body = bodies.join("");

    if body.trim().is_empty() {
        return Ok(false);
    }

    let metadata = serde_json::json!({
        "source": "claude",
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

fn extract_written_paths(path: &Path) -> Vec<PathBuf> {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let mut paths = std::collections::HashSet::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: SessionEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let msg = match &entry.message {
            Some(m) => m,
            None => continue,
        };
        let content = match &msg.content {
            Some(c) => c,
            None => continue,
        };
        if let Content::Blocks(blocks) = content {
            for block in blocks {
                if block.block_type == "tool_use"
                    && block.name.as_deref() == Some("Write")
                {
                    if let Some(ref input) = block.input {
                        if let Some(ref fp) = input.file_path {
                            paths.insert(PathBuf::from(fp));
                        }
                    }
                }
            }
        }
    }

    let mut sorted: Vec<PathBuf> = paths.into_iter().collect();
    sorted.sort();
    sorted
}

fn ingest_authored_file(db: &mut DB, path: &Path, project_name: &str, mtime_ms: i64) -> Result<bool> {
    let raw = fs::read_to_string(path)?;
    if raw.trim().is_empty() {
        return Ok(false);
    }

    let path_str = path.to_string_lossy();
    let uuid = Uuid::new_v5(
        &CLAUDE_CODE_NAMESPACE,
        format!("authored:{path_str}").as_bytes(),
    );

    // Strip YAML frontmatter (same as memory files)
    let body_text = if raw.starts_with("---\n") {
        if let Some(end) = raw[4..].find("\n---\n") {
            raw[4 + end + 5..].to_string()
        } else {
            raw.clone()
        }
    } else {
        raw.clone()
    };

    let filename = path.file_name().unwrap().to_string_lossy();
    let header = format!("[{project_name}] {filename}\n");
    let mut bodies = vec![header];
    bodies.extend(split_markdown(&body_text));
    let lengths: Vec<usize> = bodies.iter().map(|b| b.chars().count()).collect();
    let body = bodies.join("");

    if body.trim().is_empty() {
        return Ok(false);
    }

    let metadata = serde_json::json!({
        "source": "claude",
        "project": project_name,
        "path": path_str,
        "mtime_ms": mtime_ms,
    })
    .to_string();

    db.add_doc(&uuid, None, &metadata, &body, Some(lengths))?;
    Ok(true)
}

fn split_markdown(text: &str) -> Vec<String> {
    let splitter = MarkdownSplitter::new(MD_SECTION_MAX);
    splitter.chunks(text).map(|c| c.to_string()).collect()
}

/// Extract `@path` references from a CLAUDE.md/AGENTS.md file.
/// Handles both `@filename` directives and `references/foo.md - description` lines.
fn extract_at_references(text: &str, base_dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // @path reference (e.g. "@AGENTS.md", "@references/foo.md")
        if let Some(rest) = trimmed.strip_prefix('@') {
            let path_str = rest.split_whitespace().next().unwrap_or("");
            if !path_str.is_empty() {
                paths.push(base_dir.join(path_str));
            }
        }
        // "references/foo.md - description" pattern
        if let Some(path_str) = trimmed.split(" - ").next() {
            let path_str = path_str.trim();
            if path_str.ends_with(".md") && !path_str.contains(' ') {
                let candidate = base_dir.join(path_str);
                if candidate.is_file() {
                    paths.push(candidate);
                }
            }
        }
    }
    paths
}

/// Collect CLAUDE.md and AGENTS.md (plus @ references) from a project directory.
fn collect_project_configs(project_dir: &Path) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    let mut queue: Vec<PathBuf> = Vec::new();

    for name in &["CLAUDE.md", "AGENTS.md"] {
        let p = project_dir.join(name);
        if p.is_file() {
            queue.push(p);
        }
    }

    let mut result = Vec::new();
    while let Some(path) = queue.pop() {
        let canonical = match path.canonicalize() {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !seen.insert(canonical.clone()) {
            continue;
        }
        if let Ok(text) = fs::read_to_string(&canonical) {
            let base = canonical.parent().unwrap_or(project_dir);
            for referenced in extract_at_references(&text, base) {
                if referenced.is_file() {
                    queue.push(referenced);
                }
            }
        }
        result.push(canonical);
    }
    result.sort();
    result
}

use crate::watermark;

pub fn ingest_claude_code(db: &mut DB, skip_session: Option<&str>) -> Result<(usize, usize, usize, usize)> {
    let home = std::env::var("HOME").unwrap_or_default();
    let projects_dir = PathBuf::from(&home).join(".claude/projects");

    if !projects_dir.is_dir() {
        println!("no Claude Code projects found at {}", projects_dir.display());
        return Ok((0, 0, 0, 0));
    }

    let wm_path = watermark::claude_path();
    let wm_ts = watermark::mtime_ms(&wm_path);

    let mut session_count = 0usize;
    let mut memory_count = 0usize;
    let mut authored_count = 0usize;
    let mut config_count = 0usize;
    let mut ingested_paths: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

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

        // Ingest .jsonl session files, collecting authored file paths
        let mut jsonl_files: Vec<PathBuf> = fs::read_dir(&dir_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        jsonl_files.sort();

        let mut authored_paths: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

        for jsonl_path in &jsonl_files {
            if !watermark::file_newer_than(jsonl_path, wm_ts) {
                continue;
            }
            if let Some(skip_id) = skip_session {
                if jsonl_path.file_stem().is_some_and(|s| s == skip_id) {
                    continue;
                }
            }
            let mtime_ms = file_mtime_ms(jsonl_path).unwrap_or(0);
            println!("{}", jsonl_path.display());
            for p in extract_written_paths(jsonl_path) {
                if p.extension().is_some_and(|ext| ext == "md") && p.is_file() {
                    authored_paths.insert(p);
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
                authored_paths.remove(md_path);
                if !watermark::file_newer_than(md_path, wm_ts) {
                    continue;
                }
                println!("{}", md_path.display());
                ingested_paths.insert(md_path.clone());
                let mtime_ms = file_mtime_ms(md_path).unwrap_or(0);
                match ingest_memory_file(db, md_path, &project_name, mtime_ms) {
                    Ok(true) => memory_count += 1,
                    Ok(false) => {}
                    Err(e) => {
                        log::warn!("failed to ingest memory {}: {e}", md_path.display());
                    }
                }
            }
        }

        // Ingest authored .md files found in sessions
        let mut authored_sorted: Vec<PathBuf> = authored_paths.into_iter().collect();
        authored_sorted.sort();
        for md_path in &authored_sorted {
            if !watermark::file_newer_than(md_path, wm_ts) {
                continue;
            }
            let mtime_ms = file_mtime_ms(md_path).unwrap_or(0);
            println!("{}", md_path.display());
            ingested_paths.insert(md_path.clone());
            match ingest_authored_file(db, md_path, &project_name, mtime_ms) {
                Ok(true) => authored_count += 1,
                Ok(false) => {}
                Err(e) => {
                    log::warn!("failed to ingest authored {}: {e}", md_path.display());
                }
            }
        }

        // Ingest CLAUDE.md, AGENTS.md, and their @ references from the project dir
        let real_project_dir = PathBuf::from("/").join(&project_name);
        for config_path in collect_project_configs(&real_project_dir) {
            if ingested_paths.contains(&config_path) {
                continue;
            }
            if !watermark::file_newer_than(&config_path, wm_ts) {
                continue;
            }
            let mtime_ms = file_mtime_ms(&config_path).unwrap_or(0);
            println!("{}", config_path.display());
            match ingest_authored_file(db, &config_path, &project_name, mtime_ms) {
                Ok(true) => config_count += 1,
                Ok(false) => {}
                Err(e) => {
                    log::warn!("failed to ingest config {}: {e}", config_path.display());
                }
            }
        }
    }

    watermark::touch(&wm_path);
    Ok((session_count, memory_count, authored_count, config_count))
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

    fn jsonl_line(entry_type: &str, role: &str, text: &str, ts: &str, branch: Option<&str>, cwd: Option<&str>) -> String {
        let msg = serde_json::json!({
            "role": role,
            "content": text,
        });
        let mut entry = serde_json::json!({
            "type": entry_type,
            "timestamp": ts,
            "message": msg,
        });
        if let Some(b) = branch {
            entry["gitBranch"] = serde_json::Value::String(b.to_string());
        }
        if let Some(c) = cwd {
            entry["cwd"] = serde_json::Value::String(c.to_string());
        }
        serde_json::to_string(&entry).unwrap()
    }

    // -- sanitize tests --

    #[test]
    fn test_sanitize_strips_code_and_compacts() {
        let input = "Hello ```rust\nfn main() {}\n``` world  and  `inline`  code";
        let result = sanitize(input);
        assert_eq!(result, "Hello world and code");
    }

    #[test]
    fn test_sanitize_strips_system_xml() {
        let input = "Before <system-reminder>secret stuff</system-reminder> after";
        let result = sanitize(input);
        assert_eq!(result, "Before after");
    }

    #[test]
    fn test_sanitize_strips_tables() {
        let input = "Header\n| col1 | col2 |\n| --- | --- |\n| a | b |\nFooter";
        let result = sanitize(input);
        // strip_tables removes table rows, compact collapses multi-whitespace
        // but a single \n between Header and Footer doesn't trigger compaction
        assert!(!result.contains("col1"));
        assert!(result.contains("Header"));
        assert!(result.contains("Footer"));
    }

    #[test]
    fn test_sanitize_strips_interrupted_and_continuation() {
        assert_eq!(
            sanitize("Hello [Request interrupted by user] world"),
            "Hello world"
        );
        assert_eq!(
            sanitize("Keep this. This session is being continued from a previous conversation and more"),
            "Keep this."
        );
    }

    // -- parse_session_file tests --

    #[test]
    fn test_parse_extracts_custom_title() {
        let content = format!(
            "{}\n{}\n",
            r#"{"type":"custom-title","customTitle":"My Title"}"#,
            jsonl_line("user", "user", "hello world test message", "2025-01-15T10:00:00Z", None, None),
        );
        let f = write_tempfile(&content);
        let (info, chunks) = parse_session_file(f.path());
        assert_eq!(info.custom_title.as_deref(), Some("My Title"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].role, "user");
    }

    #[test]
    fn test_parse_extracts_cwd() {
        let content = format!(
            "{}\n",
            jsonl_line("user", "user", "hello world test message", "2025-01-15T10:00:00Z", None, Some("/Users/me/src/dash-search-debugger")),
        );
        let f = write_tempfile(&content);
        let (info, _) = parse_session_file(f.path());
        assert_eq!(info.cwd.as_deref(), Some("/Users/me/src/dash-search-debugger"));
    }

    #[test]
    fn test_parse_last_branch_wins() {
        let content = format!(
            "{}\n{}\n{}\n",
            jsonl_line("user", "user", "first user message here", "2025-01-15T10:00:00Z", Some("branch-a"), None),
            jsonl_line("assistant", "assistant", "first assistant response here", "2025-01-15T10:01:00Z", Some("branch-b"), None),
            jsonl_line("user", "user", "second user message here", "2025-01-15T10:02:00Z", Some("branch-c"), None),
        );
        let f = write_tempfile(&content);
        let (_, chunks) = parse_session_file(f.path());
        assert_eq!(chunks.len(), 3);
        // Last chunk has branch-c
        assert_eq!(chunks[2].git_branch.as_deref(), Some("branch-c"));
        // The last branch in the full sequence is branch-c
        let last_branch = chunks.iter().filter_map(|c| c.git_branch.as_deref()).last();
        assert_eq!(last_branch, Some("branch-c"));
    }

    #[test]
    fn test_parse_skips_short_and_long_text() {
        let short = "hi"; // < MIN_CHUNK_CODEPOINTS
        let long = "x".repeat(MAX_CHUNK_CODEPOINTS + 1);
        let ok = "this is a valid chunk of text";
        let content = format!(
            "{}\n{}\n{}\n",
            jsonl_line("user", "user", short, "2025-01-15T10:00:00Z", None, None),
            jsonl_line("user", "user", &long, "2025-01-15T10:01:00Z", None, None),
            jsonl_line("user", "user", ok, "2025-01-15T10:02:00Z", None, None),
        );
        let f = write_tempfile(&content);
        let (_, chunks) = parse_session_file(f.path());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("valid chunk"));
    }

    #[test]
    fn test_parse_empty_file() {
        let f = write_tempfile("");
        let (info, chunks) = parse_session_file(f.path());
        assert!(info.custom_title.is_none());
        assert!(info.cwd.is_none());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parse_nonexistent_file() {
        let (info, chunks) = parse_session_file(Path::new("/nonexistent/file.jsonl"));
        assert!(info.custom_title.is_none());
        assert!(chunks.is_empty());
    }

    // -- decode_project_name tests --

    #[test]
    fn test_decode_project_name() {
        assert_eq!(decode_project_name("-Users-eider-src-server"), "Users/eider/src/server");
        // Ambiguous: dashes in the actual path
        assert_eq!(
            decode_project_name("-Users-eider-src-dash-search-debugger"),
            "Users/eider/src/dash/search/debugger"
        );
    }

    #[test]
    fn test_split_markdown_no_bare_headings() {
        let md = "\
# Open Source Release Checklist

## Blockers (must fix before release)

- [ ] Add LICENSE file
- [ ] Add fields to Cargo.toml

## High priority (should fix)

- [ ] Remove internal docs
- [ ] Fix hardcoded paths

## Verified clean (no action needed)

- No secrets in tracked code";

        let chunks = split_markdown(md);

        // No chunk should be just a heading with no body
        for chunk in &chunks {
            let lines: Vec<&str> = chunk.trim().lines().collect();
            assert!(
                lines.len() > 1 || !lines[0].starts_with('#'),
                "bare heading chunk: {chunk:?}"
            );
        }

        // Every heading should appear somewhere
        assert!(chunks.iter().any(|c| c.contains("# Open Source Release Checklist")));
        assert!(chunks.iter().any(|c| c.contains("## Blockers")));
        assert!(chunks.iter().any(|c| c.contains("## High priority")));
        assert!(chunks.iter().any(|c| c.contains("## Verified clean")));

        // Headings stay with their body content
        let blockers = chunks.iter().find(|c| c.contains("## Blockers")).unwrap();
        assert!(blockers.contains("Add LICENSE file"), "Blockers heading should include its list items");

        let high = chunks.iter().find(|c| c.contains("## High priority")).unwrap();
        assert!(high.contains("Remove internal docs"), "High priority heading should include its list items");
    }
}
