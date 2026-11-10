use serde_json::{json, Value};

const DEFAULT_READ_TURN_WINDOW: usize = 20;
const MAX_READ_SESSION_CHARS: usize = 32_000;
const MAX_SEARCH_INTERACTION_CHARS: usize = 2_000;
const VALID_SOURCES: &[&str] = &["gemini", "claude", "codex", "pi", "slack"];

#[derive(Debug, Clone)]
pub enum ToolError {
    InvalidParams(String),
    ExecutionFailed(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::InvalidParams(msg) | ToolError::ExecutionFailed(msg) => f.write_str(msg),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchParams {
    pub query: String,
    pub session_id: Option<String>,
    pub project: Option<String>,
    pub source: Option<String>,
    pub limit: usize,
}

pub fn list_tools() -> Value {
    json!({
        "tools": [
            {
                "name": "search_ai_history",
                "description": "Searches indexed historical AI pair-programming and Slack sessions (Gemini, Claude, Codex, Pi, Slack) in pickbrain.db. Run `pickbrain` or `pickbrain --watch` to keep the index fresh.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Non-empty semantic search query." },
                        "session_id": { "type": "string", "description": "Optional session ID, Slack channel name/ID, or Slack thread key (thr:...)." },
                        "project": { "type": "string", "description": "Optional workspace directory basename filter (exact match, e.g. 'witchcraft'; see list_projects)." },
                        "source": { "type": "string", "enum": ["gemini", "claude", "codex", "pi", "slack"] },
                        "limit": { "type": "integer", "description": "Maximum number of results to return (default: 5, 0 = unlimited)" }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "read_session",
                "description": "Reads conversation turns from a specific session ID, Slack channel, or Slack thread (thr:...). Defaults to the first 20 turns when `turns` is omitted to conserve context tokens.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string", "description": "Session ID, channel name/ID, or Slack thread key (thr:...)." },
                        "turns": { "type": "string", "description": "Optional turn range like '7-9' or single turn '7' to conserve context tokens" },
                        "since": { "type": "string", "description": "Optional duration filter like '24h', '7d', '2w'" },
                        "include_thinking": { "type": "boolean", "description": "Include intermediate thinking blocks (default: false)" }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "list_projects",
                "description": "Lists all indexed project directory basenames (and session counts) in pickbrain.db.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "list_recent_sessions",
                "description": "Lists the 20 most recently indexed AI sessions in pickbrain.db with a 150-character preview.",
                "inputSchema": { "type": "object", "properties": {} }
            }
        ]
    })
}

fn extract_optional_string(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<String>, ToolError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(ToolError::InvalidParams(format!(
            "Invalid argument '{key}': expected a string"
        ))),
    }
}

pub fn parse_search_params(params: Option<&Value>) -> Result<SearchParams, ToolError> {
    let Some(obj) = params.and_then(|p| p.as_object()) else {
        return Err(ToolError::InvalidParams(
            "Missing required arguments object for search_ai_history".to_string(),
        ));
    };

    let query = match obj.get("query") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
        Some(Value::String(_)) | None | Some(Value::Null) => {
            return Err(ToolError::InvalidParams(
                "Missing or empty required argument: 'query'".to_string(),
            ));
        }
        Some(_) => {
            return Err(ToolError::InvalidParams(
                "Invalid argument 'query': expected a non-empty string".to_string(),
            ));
        }
    };

    let session_id = extract_optional_string(obj, "session_id")?;
    let project = extract_optional_string(obj, "project")?;
    let source = extract_optional_string(obj, "source")?;
    if let Some(ref src) = source {
        if !VALID_SOURCES.contains(&src.as_str()) {
            return Err(ToolError::InvalidParams(format!(
                "Invalid 'source' value '{src}': expected one of {}",
                VALID_SOURCES.join(", ")
            )));
        }
    }

    let limit = match obj.get("limit") {
        None | Some(Value::Null) => 5usize,
        Some(Value::Number(n)) => {
            if let Some(u) = n.as_u64() {
                u as usize
            } else {
                return Err(ToolError::InvalidParams(
                    "Invalid argument 'limit': expected a non-negative integer".to_string(),
                ));
            }
        }
        Some(_) => {
            return Err(ToolError::InvalidParams(
                "Invalid argument 'limit': expected a non-negative integer".to_string(),
            ));
        }
    };

    Ok(SearchParams {
        query,
        session_id,
        project,
        source,
        limit,
    })
}

pub fn validate_db_tool_args(name: &str, params: Option<&Value>) -> Result<(), ToolError> {
    match name {
        "read_session" => {
            let Some(obj) = params.and_then(|p| p.as_object()) else {
                return Err(ToolError::InvalidParams(
                    "Missing required arguments object for read_session".to_string(),
                ));
            };
            match obj.get("session_id") {
                Some(Value::String(s)) if !s.trim().is_empty() => {}
                Some(Value::String(_)) | None | Some(Value::Null) => {
                    return Err(ToolError::InvalidParams(
                        "Missing required argument: session_id".to_string(),
                    ));
                }
                Some(_) => {
                    return Err(ToolError::InvalidParams(
                        "Invalid argument 'session_id': expected a string".to_string(),
                    ));
                }
            }
            let _ = extract_optional_string(obj, "turns")?;
            if let Some(since_str) = extract_optional_string(obj, "since")? {
                if crate::parse_since(&since_str).is_none() {
                    return Err(ToolError::InvalidParams(format!(
                        "Invalid 'since' value '{since_str}' (use e.g. '24h', '7d', '2w')"
                    )));
                }
            }
            if let Some(v) = obj.get("include_thinking") {
                if !v.is_null() && !v.is_boolean() {
                    return Err(ToolError::InvalidParams(
                        "Invalid argument 'include_thinking': expected a boolean".to_string(),
                    ));
                }
            }
            Ok(())
        }
        "list_projects" | "list_recent_sessions" => Ok(()),
        _ => Err(ToolError::InvalidParams(format!("Unknown tool: {name}"))),
    }
}

pub fn execute_search(
    params: SearchParams,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    cache: &mut witchcraft::EmbeddingsCache,
) -> Result<Value, ToolError> {
    let types: Vec<String> = params
        .source
        .as_ref()
        .map(|s| vec![s.clone()])
        .unwrap_or_default();

    match crate::run_search_with(
        db,
        embedder,
        cache,
        &params.query,
        params.session_id.as_deref(),
        params.project.as_deref(),
        None,
        &[],
        None,
        &types,
        params.limit,
        false,
        false,
        false,
    ) {
        Ok((results, _ms)) => {
            let effective_limit = if params.limit == 0 {
                usize::MAX
            } else {
                params.limit
            };
            let mut formatted = String::new();
            for r in results.iter().take(effective_limit) {
                let raw_interaction = r
                    .bodies
                    .get(r.match_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let read_target = if !r.conv_key.is_empty() && r.conv_key.starts_with("thr:") {
                    r.conv_key.as_str()
                } else {
                    r.session_id.as_str()
                };
                let interaction = if raw_interaction.chars().count() > MAX_SEARCH_INTERACTION_CHARS
                {
                    let byte_idx = raw_interaction
                        .char_indices()
                        .nth(MAX_SEARCH_INTERACTION_CHARS)
                        .map(|(i, _)| i)
                        .unwrap_or(raw_interaction.len());
                    format!(
                        "{}\n... [Turn snippet truncated at {} chars; call `read_session` with `session_id=\"{}\"` and `turns=\"{}\"` for full turn text.]",
                        &raw_interaction[..byte_idx],
                        MAX_SEARCH_INTERACTION_CHARS,
                        read_target,
                        r.turn
                    )
                } else {
                    raw_interaction.to_string()
                };

                let mut meta_lines = format!(
                    "Source: {} | Session: {}",
                    r.source, r.session_id
                );
                if !r.session_name.is_empty() {
                    meta_lines.push_str(&format!(" | Title: {}", r.session_name));
                }
                if !r.timestamp.is_empty() {
                    meta_lines.push_str(&format!(" | Date: {}", r.timestamp));
                }
                meta_lines.push('\n');
                meta_lines.push_str(&format!("Project: {} | Turn: {}", r.project, r.turn));
                if !r.branch.is_empty() {
                    meta_lines.push_str(&format!(" | Branch: {}", r.branch));
                }
                if !r.conv_key.is_empty() {
                    meta_lines.push_str(&format!(" | ConvKey: {}", r.conv_key));
                }
                if !r.path.is_empty() {
                    meta_lines.push_str(&format!(" | Path: {}", r.path));
                }

                formatted.push_str(&format!(
                    "{meta_lines}\nInteraction:\n{interaction}\n\n"
                ));
            }
            if formatted.is_empty() {
                formatted = "No relevant history found.".to_string();
            }
            Ok(json!({ "content": [{ "type": "text", "text": formatted }] }))
        }
        Err(e) => Err(ToolError::ExecutionFailed(format!("Search error: {e}"))),
    }
}

pub fn handle_db_only_tool_call(
    name: &str,
    params: Option<Value>,
    db: &witchcraft::DB,
) -> Result<Value, ToolError> {
    validate_db_tool_args(name, params.as_ref())?;
    match name {
        "read_session" => {
            let obj = params.as_ref().and_then(|p| p.as_object()).unwrap();
            let session_id = obj
                .get("session_id")
                .and_then(|s| s.as_str())
                .unwrap()
                .trim();
            let turns = extract_optional_string(obj, "turns")?;
            let since = extract_optional_string(obj, "since")?;
            let since_ms = since.as_deref().and_then(crate::parse_since);
            let include_thinking = obj
                .get("include_thinking")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);

            match crate::fetch_session_text_bounded(
                db,
                session_id,
                turns.as_deref(),
                since_ms,
                !include_thinking,
                Some(DEFAULT_READ_TURN_WINDOW),
                Some(MAX_READ_SESSION_CHARS),
            ) {
                Ok(text) => Ok(json!({ "content": [{ "type": "text", "text": text }] })),
                Err(e) => Err(ToolError::ExecutionFailed(format!(
                    "Failed to read session {session_id}: {e}"
                ))),
            }
        }
        "list_projects" => {
            let mut stmt = db
                .query(
                    "SELECT COALESCE(NULLIF(json_extract(metadata, '$.project'), ''), '(unknown)') as proj,
                            COUNT(DISTINCT json_extract(metadata, '$.session_id')) as sessions
                     FROM document
                     WHERE json_extract(metadata, '$.session_id') IS NOT NULL
                     GROUP BY proj
                     ORDER BY sessions DESC, proj ASC",
                )
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let rows = stmt
                .query_map((), |row| {
                    Ok(format!(
                        "{} ({} sessions)",
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?
                    ))
                })
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let projects = rows
                .collect::<rusqlite::Result<Vec<String>>>()
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let out = if projects.is_empty() {
                "No projects found.".to_string()
            } else {
                projects.join("\n")
            };
            Ok(json!({ "content": [{ "type": "text", "text": out }] }))
        }
        "list_recent_sessions" => {
            let mut stmt = db
                .query(
                    "SELECT COALESCE(json_extract(d.metadata, '$.session_id'), '(unknown)') as sid,
                            COALESCE(NULLIF(json_extract(d.metadata, '$.project'), ''), '(unknown)') as proj,
                            d.date,
                            SUBSTR(d.body, 1, 150)
                     FROM document d
                     INNER JOIN (
                         SELECT json_extract(metadata, '$.session_id') as sid,
                                MIN(CAST(COALESCE(json_extract(metadata, '$.turn'), 0) AS INTEGER)) as min_turn
                         FROM document
                         WHERE json_extract(metadata, '$.session_id') IS NOT NULL
                         GROUP BY sid
                     ) first_turns
                       ON json_extract(d.metadata, '$.session_id') = first_turns.sid
                      AND CAST(COALESCE(json_extract(d.metadata, '$.turn'), 0) AS INTEGER) = first_turns.min_turn
                     GROUP BY sid
                     ORDER BY d.date DESC
                     LIMIT 20",
                )
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let rows = stmt
                .query_map((), |row| {
                    Ok(format!(
                        "ID: {} | Project: {} | Date: {}\nPreview: {}...",
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?
                    ))
                })
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let sessions = rows
                .collect::<rusqlite::Result<Vec<String>>>()
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

            let out = if sessions.is_empty() {
                "No sessions found.".to_string()
            } else {
                sessions.join("\n\n---\n\n")
            };
            Ok(json!({ "content": [{ "type": "text", "text": out }] }))
        }
        _ => Err(ToolError::InvalidParams(format!("Unknown tool: {name}"))),
    }
}

pub fn handle_tool_call_typed(
    name: &str,
    params: Option<Value>,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    cache: &mut witchcraft::EmbeddingsCache,
) -> Result<Value, ToolError> {
    match name {
        "search_ai_history" => {
            let search_params = parse_search_params(params.as_ref())?;
            execute_search(search_params, db, embedder, cache)
        }
        "read_session" | "list_projects" | "list_recent_sessions" => {
            handle_db_only_tool_call(name, params, db)
        }
        _ => Err(ToolError::InvalidParams(format!("Unknown tool: {name}"))),
    }
}

pub fn handle_tool_call(
    name: &str,
    params: Option<Value>,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    cache: &mut witchcraft::EmbeddingsCache,
) -> Result<Value, String> {
    handle_tool_call_typed(name, params, db, embedder, cache).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tests::shared_embedder;
    use uuid::Uuid;

    #[test]
    fn test_list_projects_and_recent_sessions_include_sessions_without_turn_zero_and_full_150_char_preview() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        let body_200 = "X".repeat(200);
        let meta = json!({
            "project": "partial-repo",
            "session_id": "sess-no-turn-0",
            "turn": 2,
            "source": "claude"
        })
        .to_string();
        let date = iso8601_timestamp::Timestamp::parse("2026-11-10T12:00:00Z");
        db.add_doc(
            None,
            &Uuid::from_u128(1),
            date,
            &meta,
            &body_200,
            Some(vec![body_200.len()]),
        )
        .unwrap();

        let proj_res = handle_tool_call("list_projects", None, &db, &embedder, &mut cache).unwrap();
        let proj_text = proj_res["content"][0]["text"].as_str().unwrap();
        assert!(
            proj_text.contains("partial-repo (1 sessions)"),
            "expected partial-repo in list_projects even without turn 0, got: {proj_text}"
        );

        let recent_res =
            handle_tool_call("list_recent_sessions", None, &db, &embedder, &mut cache).unwrap();
        let recent_text = recent_res["content"][0]["text"].as_str().unwrap();
        assert!(
            recent_text.contains("sess-no-turn-0"),
            "expected sess-no-turn-0 in list_recent_sessions even without turn 0, got: {recent_text}"
        );
        let expected_preview = format!("Preview: {}...", "X".repeat(150));
        assert!(
            recent_text.contains(&expected_preview),
            "expected 150-char preview (not 149 from SUBSTR(body, 0, 150)), got: {recent_text}"
        );
    }

    #[test]
    fn test_read_session_default_turn_window_and_truncation_guidance() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        for turn in 0..30u64 {
            let body = format!("[claude:repo] Title\n[User] Question {turn}\n[Claude] Answer {turn}\n");
            let len = body.chars().count();
            let meta = json!({
                "project": "repo",
                "session_id": "long-session-30",
                "turn": turn,
                "source": "claude"
            })
            .to_string();
            let date = iso8601_timestamp::Timestamp::parse("2026-11-10T12:00:00Z");
            db.add_doc(
                None,
                &Uuid::from_u128(100 + turn as u128),
                date,
                &meta,
                &body,
                Some(vec![len]),
            )
            .unwrap();
        }

        let res = handle_tool_call(
            "read_session",
            Some(json!({"session_id": "long-session-30"})),
            &db,
            &embedder,
            &mut cache,
        )
        .unwrap();
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("turns"),
            "expected truncation guidance mentioning `turns` parameter when reading a 30-turn session without `turns`, got: {text}"
        );
        assert!(
            !text.contains("Question 29") || !text.contains("Question 0"),
            "expected default turn window to bound a 30-turn session rather than dumping all 30 turns"
        );
    }

    #[test]
    fn test_search_ai_history_project_pushdown_limit_zero_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let device = witchcraft::make_device();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        // Insert 12 noise documents in "noise-proj" and 2 target documents in "needle-proj"
        for i in 0..12u64 {
            let body = format!(
                "[User] OAuth authentication token middleware configuration step {i}\n[Claude] Updated OAuth middleware step {i}.\n"
            );
            let len = body.chars().count();
            let meta = json!({
                "project": "noise-proj",
                "session_id": format!("noise-sess-{i}"),
                "session_name": format!("Noise Session {i}"),
                "turn": 0,
                "branch": "main",
                "path": format!("/tmp/noise-{i}.jsonl"),
                "source": "claude"
            })
            .to_string();
            let date = iso8601_timestamp::Timestamp::parse("2026-11-10T12:00:00Z");
            db.add_doc(
                None,
                &Uuid::from_u128(1000 + i as u128),
                date,
                &meta,
                &body,
                Some(vec![len]),
            )
            .unwrap();
        }

        for i in 0..2u64 {
            let body = format!(
                "[User] Discussed authentication middleware in Slack thread {i}\n[Slack] Replied on authentication middleware {i}.\n"
            );
            let len = body.chars().count();
            let meta = json!({
                "project": "needle-proj",
                "session_id": "C12345",
                "session_name": "security-arch",
                "turn": i,
                "branch": "feat/auth",
                "path": "/tmp/slack.db",
                "conv_key": format!("thr:C12345-170000000{i}.000100"),
                "source": "slack"
            })
            .to_string();
            let date = iso8601_timestamp::Timestamp::parse("2026-11-09T08:30:00Z");
            db.add_doc(
                None,
                &Uuid::from_u128(2000 + i as u128),
                date,
                &meta,
                &body,
                Some(vec![len]),
            )
            .unwrap();
        }

        crate::embed_and_index(&db, &embedder, &device).unwrap();

        // 1. Filtering by project="needle-proj" must find needle-proj even though noise-proj has 12 closer hits
        let proj_res = handle_tool_call(
            "search_ai_history",
            Some(json!({
                "query": "OAuth authentication token middleware configuration",
                "project": "needle-proj",
                "limit": 5
            })),
            &db,
            &embedder,
            &mut cache,
        )
        .unwrap();
        let proj_text = proj_res["content"][0]["text"].as_str().unwrap();
        assert!(
            proj_text.contains("needle-proj"),
            "expected project filter to push down to SQL and return needle-proj results, got: {proj_text}"
        );
        // Verify rich metadata (timestamp, session_name, branch, conv_key) is present
        assert!(
            proj_text.contains("thr:C12345-170000000"),
            "expected search result to include conv_key for Slack threads, got: {proj_text}"
        );
        assert!(
            proj_text.contains("feat/auth"),
            "expected search result to include branch metadata, got: {proj_text}"
        );

        // 2. limit: 0 must act as unlimited (or not return false-negative "No relevant history found.")
        let limit_zero_res = handle_tool_call(
            "search_ai_history",
            Some(json!({
                "query": "authentication middleware",
                "limit": 0
            })),
            &db,
            &embedder,
            &mut cache,
        )
        .unwrap();
        let limit_zero_text = limit_zero_zero_text(&limit_zero_res);
        assert!(
            !limit_zero_text.contains("No relevant history found."),
            "limit: 0 must not return false-negative 'No relevant history found.'"
        );
        assert!(
            limit_zero_text.matches("Session:").count() > 10,
            "expected limit: 0 (unlimited) to return >10 matches across 14 documents, got {}",
            limit_zero_text.matches("Session:").count()
        );
    }

    fn limit_zero_zero_text(val: &Value) -> &str {
        val["content"][0]["text"].as_str().unwrap_or("")
    }

    #[test]
    fn test_list_recent_sessions_orders_by_latest_turn_activity() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        // Session A started earlier (Nov 1) but had a turn more recently (Nov 10)
        db.add_doc(
            None,
            &Uuid::from_u128(10),
            iso8601_timestamp::Timestamp::parse("2026-11-01T10:00:00Z"),
            &json!({"project": "proj-a", "session_id": "sess-resumed-recently", "turn": 0, "source": "claude"}).to_string(),
            "Initial turn on Nov 1",
            Some(vec![21]),
        )
        .unwrap();
        db.add_doc(
            None,
            &Uuid::from_u128(11),
            iso8601_timestamp::Timestamp::parse("2026-11-10T18:00:00Z"),
            &json!({"project": "proj-a", "session_id": "sess-resumed-recently", "turn": 1, "source": "claude"}).to_string(),
            "Follow-up turn on Nov 10",
            Some(vec![24]),
        )
        .unwrap();

        // Session B started on Nov 5 and never had further activity
        db.add_doc(
            None,
            &Uuid::from_u128(12),
            iso8601_timestamp::Timestamp::parse("2026-11-05T10:00:00Z"),
            &json!({"project": "proj-b", "session_id": "sess-stale", "turn": 0, "source": "claude"}).to_string(),
            "Single turn on Nov 5",
            Some(vec![20]),
        )
        .unwrap();

        let recent_res =
            handle_tool_call("list_recent_sessions", None, &db, &embedder, &mut cache).unwrap();
        let recent_text = recent_res["content"][0]["text"].as_str().unwrap();
        let pos_resumed = recent_text
            .find("sess-resumed-recently")
            .expect("sess-resumed-recently should appear");
        let pos_stale = recent_text
            .find("sess-stale")
            .expect("sess-stale should appear");
        assert!(
            pos_resumed < pos_stale,
            "expected session with latest turn activity (Nov 10) to sort before stale session (Nov 5), got:\n{recent_text}"
        );
    }

    #[test]
    fn test_read_session_out_of_range_turns_reports_available_range() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        for turn in 0..3u64 {
            let body = format!("[User] Turn {turn}\n[Claude] Reply {turn}\n");
            let len = body.chars().count();
            db.add_doc(
                None,
                &Uuid::from_u128(50 + turn as u128),
                iso8601_timestamp::Timestamp::parse("2026-11-10T12:00:00Z"),
                &json!({"project": "repo", "session_id": "short-sess", "turn": turn, "source": "claude"}).to_string(),
                &body,
                Some(vec![len]),
            )
            .unwrap();
        }

        let err = handle_tool_call(
            "read_session",
            Some(json!({"session_id": "short-sess", "turns": "50-60"})),
            &db,
            &embedder,
            &mut cache,
        )
        .expect_err("out-of-range turns should return an execution error with available range");
        assert!(
            err.contains("0-2"),
            "expected out-of-range error to report available turns 0-2 instead of claiming session not found, got: {err}"
        );
    }

    #[test]
    fn test_search_ai_history_supports_since_filter_and_snippet_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let device = witchcraft::make_device();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let recent_iso = chrono::DateTime::from_timestamp(now_secs - 3600, 0)
            .unwrap()
            .to_rfc3339();
        let old_iso = chrono::DateTime::from_timestamp(now_secs - 30 * 86400, 0)
            .unwrap()
            .to_rfc3339();

        let long_body = format!(
            "[User] Configure authentication middleware {}\n",
            "Z".repeat(2400)
        );
        db.add_doc(
            None,
            &Uuid::from_u128(3001),
            iso8601_timestamp::Timestamp::parse(&recent_iso),
            &json!({"project": "proj", "session_id": "recent-sess", "turn": 0, "source": "claude"}).to_string(),
            &long_body,
            Some(vec![long_body.chars().count()]),
        )
        .unwrap();

        let old_body = "[User] Configure authentication middleware old session\n";
        db.add_doc(
            None,
            &Uuid::from_u128(3002),
            iso8601_timestamp::Timestamp::parse(&old_iso),
            &json!({"project": "proj", "session_id": "old-sess", "turn": 0, "source": "claude"}).to_string(),
            old_body,
            Some(vec![old_body.chars().count()]),
        )
        .unwrap();

        crate::embed_and_index(&db, &embedder, &device).unwrap();

        let res = handle_tool_call(
            "search_ai_history",
            Some(json!({
                "query": "Configure authentication middleware",
                "since": "24h"
            })),
            &db,
            &embedder,
            &mut cache,
        )
        .unwrap();
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("recent-sess"),
            "expected recent-sess in since='24h' search results, got: {text}"
        );
        assert!(
            !text.contains("old-sess"),
            "expected old-sess (30d old) to be excluded by since='24h', got: {text}"
        );
        assert!(
            text.contains("Turn snippet truncated at 2000 chars"),
            "expected 2400-char interaction body to be truncated with notice, got: {text}"
        );
    }
}
