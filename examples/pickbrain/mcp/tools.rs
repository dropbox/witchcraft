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

// In unit tests we omit state execution by ignoring test outputs
// We only run mcp struct tests on a mock integration suite within core tests.
