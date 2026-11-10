use serde_json::{json, Value};

pub fn list_tools() -> Value {
    json!({
        "tools": [
            {
                "name": "search_ai_history",
                "description": "Searches across all historical AI pair-programming sessions (Gemini, Claude, Codex, Pi, Slack) for the requested context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "session_id": { "type": "string" },
                        "project": { "type": "string" },
                        "source": { "type": "string", "enum": ["gemini", "claude", "codex", "pi", "slack"] },
                        "limit": { "type": "integer", "description": "Maximum number of results to return (default: 5)" }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "read_session",
                "description": "Reads full conversation context from a specific session ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": { "type": "string" },
                        "turns": { "type": "string", "description": "Optional turn range like '7-9' or single turn '7' to conserve context tokens" },
                        "since": { "type": "string", "description": "Optional duration filter like '24h', '7d', '2w'" },
                        "include_thinking": { "type": "boolean", "description": "Include intermediate thinking blocks (default: false)" }
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "list_projects",
                "description": "Lists all projects (repositories) that have been indexed.",
                "inputSchema": { "type": "object", "properties": {} }
            },
            {
                "name": "list_recent_sessions",
                "description": "Lists the last 20 AI sessions chronologically.",
                "inputSchema": { "type": "object", "properties": {} }
            }
        ]
    })
}

pub fn handle_tool_call(
    name: &str,
    params: Option<Value>,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    _cache: &mut witchcraft::EmbeddingsCache,
) -> Result<Value, String> {
    match name {
        "search_ai_history" => {
            let q = params
                .as_ref()
                .and_then(|p| p.get("query"))
                .and_then(|q| q.as_str())
                .unwrap_or("");
            let sid = params
                .as_ref()
                .and_then(|p| p.get("session_id"))
                .and_then(|s| s.as_str());
            let proj = params
                .as_ref()
                .and_then(|p| p.get("project"))
                .and_then(|s| s.as_str());
            let src = params
                .as_ref()
                .and_then(|p| p.get("source"))
                .and_then(|s| s.as_str());
            let limit = params
                .as_ref()
                .and_then(|p| p.get("limit"))
                .and_then(|l| l.as_u64())
                .unwrap_or(5) as usize;

            let types: Vec<String> = src.map(|s| vec![s.to_string()]).unwrap_or_default();

            match crate::run_search_with(
                db,
                embedder,
                q,
                sid,
                None,
                &[],
                None,
                &types,
                false,
                false,
                false,
            ) {
                Ok((results, _ms)) => {
                    let mut formatted = String::new();
                    let filtered: Box<dyn Iterator<Item = &crate::SearchResult>> = match proj {
                        Some(p) => Box::new(results.iter().filter(move |r| r.project == p)),
                        None => Box::new(results.iter()),
                    };
                    for r in filtered.take(limit) {
                        // Smart Context: Grouped by interaction naturally by Ingestor bodies
                        let interaction =
                            r.bodies.get(r.match_idx).map(|s| s.as_str()).unwrap_or("");
                        formatted.push_str(&format!(
                            "Source: {} | Session: {}\nProject: {}\nTurn: {}\nInteraction:\n{}\n\n",
                            r.source, r.session_id, r.project, r.turn, interaction
                        ));
                    }
                    if formatted.is_empty() {
                        formatted = "No relevant history found.".to_string();
                    }
                    Ok(json!({ "content": [{ "type": "text", "text": formatted }] }))
                }
                Err(e) => Err(format!("Search error: {}", e)),
            }
        }
        "read_session" => {
            let sid = params
                .as_ref()
                .and_then(|p| p.get("session_id"))
                .and_then(|s| s.as_str());
            let turns = params
                .as_ref()
                .and_then(|p| p.get("turns"))
                .and_then(|s| s.as_str());
            let since = params
                .as_ref()
                .and_then(|p| p.get("since"))
                .and_then(|s| s.as_str());
            let since_ms = since.and_then(crate::parse_since);
            let include_thinking = params
                .as_ref()
                .and_then(|p| p.get("include_thinking"))
                .and_then(|b| b.as_bool())
                .unwrap_or(false);

            if let Some(session_id) = sid {
                match crate::fetch_session_text(
                    db,
                    session_id,
                    turns,
                    since_ms,
                    !include_thinking,
                ) {
                    Ok(text) => Ok(json!({ "content": [{ "type": "text", "text": text }] })),
                    Err(e) => Err(format!("Failed to read session {}: {}", session_id, e)),
                }
            } else {
                Err("Missing required argument: session_id".to_string())
            }
        }
        "list_projects" => {
            let mut stmt = db.query(
                "SELECT DISTINCT json_extract(metadata, '$.project') as proj, COUNT(*) as sessions 
                 FROM document 
                 WHERE json_extract(metadata, '$.turn') = 0 
                 GROUP BY proj 
                 ORDER BY sessions DESC"
            ).map_err(|e| e.to_string())?;

            let projects: Vec<String> = stmt
                .query_map((), |row| {
                    Ok(format!(
                        "{} ({} sessions)",
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?
                    ))
                })
                .map_err(|e| e.to_string())?
                .filter_map(|r| r.ok())
                .collect();

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
                    "SELECT json_extract(metadata, '$.session_id'), 
                        json_extract(metadata, '$.project'), 
                        date, 
                        SUBSTR(body, 0, 150)
                 FROM document 
                 WHERE json_extract(metadata, '$.turn') = 0 
                 ORDER BY date DESC 
                 LIMIT 20",
                )
                .map_err(|e| e.to_string())?;

            let sessions: Vec<String> = stmt
                .query_map((), |row| {
                    Ok(format!(
                        "ID: {} | Project: {} | Date: {}\nPreview: {}...",
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?
                    ))
                })
                .map_err(|e| e.to_string())?
                .filter_map(|r| r.ok())
                .collect();

            let out = if sessions.is_empty() {
                "No sessions found.".to_string()
            } else {
                sessions.join("\n\n---\n\n")
            };
            Ok(json!({ "content": [{ "type": "text", "text": out }] }))
        }
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::OnceLock;
    use uuid::Uuid;

    fn shared_embedder() -> &'static witchcraft::Embedder {
        static EMBEDDER: OnceLock<witchcraft::Embedder> = OnceLock::new();
        EMBEDDER.get_or_init(|| {
            let device = witchcraft::make_device();
            witchcraft::Embedder::new(&device, Path::new("assets")).unwrap()
        })
    }

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

        let proj_res = handle_tool_call("list_projects", None, &db, embedder, &mut cache).unwrap();
        let proj_text = proj_res["content"][0]["text"].as_str().unwrap();
        assert!(
            proj_text.contains("partial-repo (1 sessions)"),
            "expected partial-repo in list_projects even without turn 0, got: {proj_text}"
        );

        let recent_res =
            handle_tool_call("list_recent_sessions", None, &db, embedder, &mut cache).unwrap();
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
            embedder,
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

        crate::embed_and_index(&db, embedder, &device).unwrap();

        // 1. Filtering by project="needle-proj" must find needle-proj even though noise-proj has 12 closer hits
        let proj_res = handle_tool_call(
            "search_ai_history",
            Some(json!({
                "query": "OAuth authentication token middleware configuration",
                "project": "needle-proj",
                "limit": 5
            })),
            &db,
            embedder,
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
            embedder,
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
}
