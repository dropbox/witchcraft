use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use uuid::Uuid;

use witchcraft::DB;

const MIN_CHUNK_CODEPOINTS: usize = 5;
const MAX_CHUNK_CODEPOINTS: usize = 4000;

const CODEX_NAMESPACE: Uuid = Uuid::from_bytes([
    0xb4, 0xe8, 0xd9, 0xe2, 0x7f, 0x3c, 0x5b, 0xa2, 0xc6, 0xe1, 0x90, 0x2f, 0x4d, 0x8b, 0xac,
    0x3e,
]);

#[derive(Deserialize)]
struct Entry {
    timestamp: Option<String>,
    #[serde(rename = "type")]
    entry_type: String,
    #[serde(default)]
    payload: serde_json::Value,
}

struct Chunk {
    role: String,
    text: String,
    timestamp: String,
    ts_ms: i64,
    byte_offset: u64,
    byte_len: u64,
}

fn codepoint_len(s: &str) -> usize {
    s.chars().count()
}

fn sanitize(text: &str) -> String {
    let s = strip_xml(text);
    let s = strip_code(&s);
    compact(&s)
}

fn strip_xml(text: &str) -> String {
    let re = Regex::new(r"<[^>]+>").unwrap();
    re.replace_all(text, "").to_string()
}

fn strip_code(text: &str) -> String {
    let re = Regex::new(r"```[\s\S]*?```").unwrap();
    let s = re.replace_all(text, " ").to_string();
    let re = Regex::new(r"```[\s\S]*$").unwrap();
    let s = re.replace_all(&s, " ").to_string();
    let re = Regex::new(r"`[^`]*`").unwrap();
    re.replace_all(&s, " ").to_string()
}

fn compact(text: &str) -> String {
    let re = Regex::new(r"\s{2,}").unwrap();
    re.replace_all(text, " ").trim().to_string()
}

fn extract_user_text(payload: &serde_json::Value) -> Option<String> {
    if payload.get("type")?.as_str()? != "message" {
        return None;
    }
    if payload.get("role")?.as_str()? != "user" {
        return None;
    }
    let content = payload.get("content")?.as_array()?;
    let texts: Vec<&str> = content
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

fn extract_reasoning_text(payload: &serde_json::Value) -> Option<String> {
    if payload.get("type")?.as_str()? != "agent_reasoning" {
        return None;
    }
    payload.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
}

/// Extract a branch name from git checkout/switch output in exec command results.
/// Matches patterns like:
///   "Switched to a new branch 'foo'"
///   "Switched to branch 'foo'"
///   "Already on 'foo'"
///
/// TODO: git's single-quote format isn't guaranteed across versions/locales.
/// Covers all standard English git output; revisit if we hit edge cases.
static BRANCH_SWITCH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:Switched to (?:a new )?branch|Already on) '([^']+)'").unwrap()
});

fn extract_branch_from_output(payload: &serde_json::Value) -> Option<String> {
    let ty = payload.get("type")?.as_str()?;
    if ty != "function_call_output" {
        return None;
    }
    let output = payload.get("output")?.as_str()?;
    BRANCH_SWITCH_RE
        .captures(output)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

struct SessionMeta {
    cwd: Option<String>,
    branch: Option<String>,
}

fn parse_session_file(path: &Path) -> (SessionMeta, Vec<Chunk>) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return (SessionMeta { cwd: None, branch: None }, vec![]),
    };

    let mut meta = SessionMeta { cwd: None, branch: None };
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

        if entry.entry_type == "session_meta" {
            if meta.cwd.is_none() {
                meta.cwd = entry.payload.get("cwd").and_then(|c| c.as_str()).map(|s| s.to_string());
            }
            // Only set from session_meta if no branch seen yet — exec output
            // overwrites unconditionally below, so the last checkout always wins.
            if meta.branch.is_none() {
                meta.branch = entry.payload
                    .get("git")
                    .and_then(|g| g.get("branch"))
                    .and_then(|b| b.as_str())
                    .map(|s| s.to_string());
            }
            continue;
        }

        // Detect branch changes from git checkout/switch in exec output.
        // Unconditional overwrite: the last branch switch in the session wins.
        if let Some(switched) = extract_branch_from_output(&entry.payload) {
            meta.branch = Some(switched);
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

        let (role, raw_text) = if entry.entry_type == "response_item" {
            if let Some(text) = extract_user_text(&entry.payload) {
                ("user".to_string(), text)
            } else {
                continue;
            }
        } else if entry.entry_type == "event_msg" {
            if let Some(text) = extract_reasoning_text(&entry.payload) {
                ("assistant".to_string(), text)
            } else {
                continue;
            }
        } else {
            continue;
        };

        let text = sanitize(&raw_text);
        if text.is_empty() {
            continue;
        }
        let cp_len = codepoint_len(&text);
        if cp_len < MIN_CHUNK_CODEPOINTS || cp_len > MAX_CHUNK_CODEPOINTS {
            continue;
        }

        chunks.push(Chunk {
            role,
            text,
            timestamp,
            ts_ms,
            byte_offset: line_offset,
            byte_len: line.len() as u64,
        });
    }

    chunks.sort_by_key(|c| c.ts_ms);
    (meta, chunks)
}

fn project_from_cwd(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| cwd.to_string())
}

fn session_id_from_filename(path: &Path) -> String {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    // Filename: rollout-YYYY-MM-DDTHH-mm-ss-<UUID>
    // Extract the UUID part (last 36 chars of the stem)
    if stem.len() >= 36 {
        let uuid_part = &stem[stem.len() - 36..];
        if uuid_part.chars().filter(|&c| c == '-').count() == 4 {
            return uuid_part.to_string();
        }
    }
    stem.to_string()
}

fn ingest_session(
    db: &mut DB,
    path: &Path,
    mtime_ms: i64,
    session_name: Option<&str>,
) -> Result<usize> {
    let (meta, chunks) = parse_session_file(path);
    if chunks.is_empty() {
        return Ok(0);
    }

    let project_name = meta.cwd.as_deref().map(project_from_cwd).unwrap_or_default();
    let session_id = session_id_from_filename(path);

    let session_title: String = session_name.map(|s| s.to_string()).unwrap_or_else(|| {
        chunks
            .iter()
            .find(|c| c.role == "user")
            .map(|c| c.text.chars().take(240).collect())
            .unwrap_or_default()
    });

    // Split into interactions starting at each user message
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
        let header = format!("[codex:{project_name}] {session_title}\n");
        // One entry per turn so sub_idx maps 1:1 to conversation turns
        let mut all_parts = vec![header];
        let mut turns_meta: Vec<serde_json::Value> = Vec::new();

        for chunk in *interaction {
            let label = if chunk.role == "user" {
                "[User]"
            } else {
                "[Codex]"
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
            &CODEX_NAMESPACE,
            format!("{session_id}:{turn_idx}").as_bytes(),
        );

        let metadata = serde_json::json!({
            "source": "codex",
            "project": project_name,
            "session_id": session_id,
            "session_name": session_title,
            "turn": turn_idx,
            "path": path.to_string_lossy(),
            "cwd": meta.cwd,
            "mtime_ms": mtime_ms,
            "turns": turns_meta,
            "branch": meta.branch,
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

/// Walk ~/.codex/sessions/ recursively for .jsonl files
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

fn load_session_index() -> std::collections::HashMap<String, String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let index_path = PathBuf::from(&home).join(".codex/session_index.jsonl");
    let mut names = std::collections::HashMap::new();
    let raw = match fs::read_to_string(&index_path) {
        Ok(s) => s,
        Err(_) => return names,
    };
    for line in raw.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(id), Some(name)) = (
                v.get("id").and_then(|i| i.as_str()),
                v.get("thread_name").and_then(|n| n.as_str()),
            ) {
                names.insert(id.to_string(), name.to_string());
            }
        }
    }
    names
}

pub fn ingest_codex(db: &mut DB) -> Result<usize> {
    let home = std::env::var("HOME").unwrap_or_default();
    let sessions_dir = PathBuf::from(&home).join(".codex/sessions");

    if !sessions_dir.is_dir() {
        return Ok(0);
    }

    let session_names = load_session_index();
    let wm_path = watermark::codex_path();
    let wm_ts = watermark::mtime_ms(&wm_path);
    let mut session_count = 0usize;

    for jsonl_path in collect_session_files(&sessions_dir) {
        if !watermark::file_newer_than(&jsonl_path, wm_ts) {
            continue;
        }
        let mtime_ms = file_mtime_ms(&jsonl_path).unwrap_or(0);
        let sid = session_id_from_filename(&jsonl_path);
        let name = session_names.get(&sid).map(|s| s.as_str());
        println!("{}", jsonl_path.display());
        match ingest_session(db, &jsonl_path, mtime_ms, name) {
            Ok(n) => session_count += n,
            Err(e) => {
                log::warn!("failed to ingest codex {}: {e}", jsonl_path.display());
            }
        }
    }

    watermark::touch(&wm_path);
    Ok(session_count)
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

    // -- extract_branch_from_output tests --

    #[test]
    fn test_extract_branch_from_output() {
        let cases = vec![
            // (output text, expected branch)
            ("Switched to a new branch 'feature/foo'", Some("feature/foo")),
            ("Switched to branch 'main'", Some("main")),
            ("Already on 'develop'", Some("develop")),
            ("Switched to a new branch 'codex/reserve-deprecated-signals'", Some("codex/reserve-deprecated-signals")),
            // No match
            ("some random output", None),
            ("branch 'foo' deleted", None),
        ];
        for (output, expected) in cases {
            let payload = serde_json::json!({
                "type": "function_call_output",
                "output": output,
            });
            assert_eq!(
                extract_branch_from_output(&payload).as_deref(),
                expected,
                "failed for output: {output}"
            );
        }
    }

    #[test]
    fn test_extract_branch_wrong_type() {
        let payload = serde_json::json!({
            "type": "agent_reasoning",
            "output": "Switched to a new branch 'foo'",
        });
        assert_eq!(extract_branch_from_output(&payload), None);
    }

    // -- parse_session_file tests --

    fn session_meta_line(cwd: &str, branch: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "type": "session_meta",
            "payload": {
                "cwd": cwd,
                "git": { "branch": branch }
            }
        })).unwrap()
    }

    fn user_msg_line(text: &str, ts: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "type": "response_item",
            "timestamp": ts,
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }]
            }
        })).unwrap()
    }

    fn checkout_line(output: &str) -> String {
        serde_json::to_string(&serde_json::json!({
            "type": "event_msg",
            "timestamp": "2025-01-15T10:05:00Z",
            "payload": {
                "type": "function_call_output",
                "output": output,
            }
        })).unwrap()
    }

    #[test]
    fn test_codex_parse_initial_branch() {
        let content = format!(
            "{}\n{}\n",
            session_meta_line("/Users/me/src/server", "main"),
            user_msg_line("hello world from the user", "2025-01-15T10:00:00Z"),
        );
        let f = write_tempfile(&content);
        let (meta, chunks) = parse_session_file(f.path());
        assert_eq!(meta.branch.as_deref(), Some("main"));
        assert_eq!(meta.cwd.as_deref(), Some("/Users/me/src/server"));
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_codex_parse_branch_switch_overrides() {
        let content = format!(
            "{}\n{}\n{}\n",
            session_meta_line("/Users/me/src/server", "main"),
            checkout_line("Switched to a new branch 'feature/xyz'"),
            user_msg_line("do some work on the branch", "2025-01-15T10:10:00Z"),
        );
        let f = write_tempfile(&content);
        let (meta, _) = parse_session_file(f.path());
        assert_eq!(meta.branch.as_deref(), Some("feature/xyz"));
    }

    #[test]
    fn test_codex_parse_multiple_switches_last_wins() {
        let content = format!(
            "{}\n{}\n{}\n{}\n",
            session_meta_line("/Users/me/src/server", "main"),
            checkout_line("Switched to a new branch 'first-branch'"),
            checkout_line("Switched to branch 'second-branch'"),
            user_msg_line("work on second branch now", "2025-01-15T10:10:00Z"),
        );
        let f = write_tempfile(&content);
        let (meta, _) = parse_session_file(f.path());
        assert_eq!(meta.branch.as_deref(), Some("second-branch"));
    }

    #[test]
    fn test_codex_parse_session_meta_after_checkout_doesnt_overwrite() {
        // session_meta branch is first-write-only, so a later session_meta
        // should NOT overwrite a branch detected from exec output
        let content = format!(
            "{}\n{}\n{}\n{}\n",
            session_meta_line("/Users/me/src/server", "main"),
            checkout_line("Switched to a new branch 'feature/new'"),
            // second session_meta (e.g. from a reconnect)
            session_meta_line("/Users/me/src/server", "main"),
            user_msg_line("still on feature branch", "2025-01-15T10:10:00Z"),
        );
        let f = write_tempfile(&content);
        let (meta, _) = parse_session_file(f.path());
        assert_eq!(meta.branch.as_deref(), Some("feature/new"));
    }

    #[test]
    fn test_codex_parse_empty_file() {
        let f = write_tempfile("");
        let (meta, chunks) = parse_session_file(f.path());
        assert!(meta.branch.is_none());
        assert!(meta.cwd.is_none());
        assert!(chunks.is_empty());
    }

    // -- session_id_from_filename --

    #[test]
    fn test_session_id_from_filename() {
        let cases = vec![
            ("rollout-2025-01-15T10-00-00-abcdef01-2345-6789-abcd-ef0123456789.jsonl",
             "abcdef01-2345-6789-abcd-ef0123456789"),
            ("short.jsonl", "short"),
        ];
        for (filename, expected) in cases {
            let path = PathBuf::from(filename);
            assert_eq!(session_id_from_filename(&path), expected, "failed for {filename}");
        }
    }

    // -- project_from_cwd --

    #[test]
    fn test_project_from_cwd() {
        assert_eq!(project_from_cwd("/Users/me/src/server"), "server");
        assert_eq!(project_from_cwd("/Users/me/src/dash-search-debugger"), "dash-search-debugger");
        // Root path has no file_name, falls back to the input string
        assert_eq!(project_from_cwd("/"), "/");
    }
}
