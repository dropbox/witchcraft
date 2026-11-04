use anyhow::Result;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use witchcraft::DB;

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
                current_turn_blocks.push(TurnRaw {
                    role: "user".to_string(),
                    text: content.to_string(),
                    timestamp,
                    offset: line_offset,
                    len: line_len,
                });
            } else if msg_type == "PLANNER_RESPONSE" {
                let mut text = content.to_string();
                if text.is_empty() {
                    if let Some(thinking) = v.get("thinking").and_then(|t| t.as_str()) {
                        text = thinking.to_string();
                    }
                }
                if !text.is_empty() {
                    current_turn_blocks.push(TurnRaw {
                        role: "assistant".to_string(),
                        text,
                        timestamp,
                        offset: line_offset,
                        len: line_len,
                    });
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
                current_turn_blocks.push(TurnRaw {
                    role: "user".to_string(),
                    text: trimmed[5..].trim().to_string(),
                    timestamp: "".to_string(),
                    offset: line_offset,
                    len: line_len,
                });
            } else {
                current_turn_blocks.push(TurnRaw {
                    role: "assistant".to_string(),
                    text: trimmed[6..].trim().to_string(),
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
        let mut all_parts = vec![format!("[gemini:{project_name}] Session {}\n", session_id)];
        let mut turns_meta = Vec::new();

        for turn in interaction {
            let label = if turn.role == "user" {
                "[User]"
            } else {
                "[Gemini]"
            };
            all_parts.push(format!("{label} {}\n", turn.text));
            turns_meta.push(serde_json::json!({
                "role": turn.role,
                "timestamp": turn.timestamp,
                "off": turn.offset,
                "len": turn.len,
            }));
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
}
