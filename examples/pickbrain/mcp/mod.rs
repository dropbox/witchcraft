use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum JsonRpcId {
    Number(u64),
    String(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

pub mod tools;

pub struct ServerState {
    pub db: witchcraft::DB,
    pub embedder: witchcraft::Embedder,
    pub cache: witchcraft::EmbeddingsCache,
}

impl ServerState {
    pub fn init(db_name: &std::path::Path, assets: &std::path::Path) -> Result<Self> {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, assets)?;
        let cache = witchcraft::EmbeddingsCache::new(1);
        let db = witchcraft::DB::new_reader(db_name.to_path_buf())?;
        Ok(Self {
            db,
            embedder,
            cache,
        })
    }
}

pub fn start_server(db_name: &std::path::PathBuf, assets: &std::path::PathBuf) -> Result<()> {
    use std::io::IsTerminal;
    let stdin = io::stdin();
    if stdin.is_terminal() && io::stderr().is_terminal() {
        eprintln!(
            "pickbrain {} MCP server listening on stdio (JSON-RPC 2.0)...",
            env!("CARGO_PKG_VERSION")
        );
    }
    let mut reader = stdin.lock();
    let mut line = String::new();
    let mut state: Option<ServerState> = None;

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break; // EOF
        }

        if let Some(resp_json) = dispatch_request(&line, |id, params| {
            if state.is_none() {
                match ServerState::init(db_name, assets) {
                    Ok(s) => state = Some(s),
                    Err(e) => {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id,
                            result: None,
                            error: Some(serde_json::json!({
                                "code": -32603,
                                "message": format!("Backend initialization failed: {}", e)
                            })),
                        };
                    }
                }
            }
            let s = state.as_mut().unwrap();
            execute_tool_call(id, params, &s.db, &s.embedder, &mut s.cache)
        }) {
            println!("{}", resp_json);
            io::stdout().flush()?;
        }
    }
    Ok(())
}

fn execute_tool_call(
    id: Option<JsonRpcId>,
    params: Option<serde_json::Value>,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    cache: &mut witchcraft::EmbeddingsCache,
) -> JsonRpcResponse {
    if let Some(params) = params {
        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned();

        match tools::handle_tool_call(name, args, db, embedder, cache) {
            Ok(result) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(result),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(serde_json::json!({
                    "code": -32603,
                    "message": format!("Tool execution failed: {}", e)
                })),
            },
        }
    } else {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(serde_json::json!({
                "code": -32602,
                "message": "Missing params for tools/call"
            })),
        }
    }
}

pub fn dispatch_request<F>(raw_json: &str, mut on_tool_call: F) -> Option<String>
where
    F: FnMut(Option<JsonRpcId>, Option<serde_json::Value>) -> JsonRpcResponse,
{
    let req: JsonRpcRequest = match serde_json::from_str(raw_json) {
        Ok(r) => r,
        Err(_) => return None,
    };

    if req.id.is_none() {
        return None;
    }

    let resp = match req.method.as_str() {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "pickbrain",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        },
        "ping" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(serde_json::json!({})),
            error: None,
        },
        "resources/list" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(serde_json::json!({ "resources": [] })),
            error: None,
        },
        "prompts/list" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(serde_json::json!({ "prompts": [] })),
            error: None,
        },
        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: Some(tools::list_tools()),
            error: None,
        },
        "tools/call" => on_tool_call(req.id, req.params),
        _ => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: req.id,
            result: None,
            error: Some(serde_json::json!({
                "code": -32601,
                "message": format!("Method not found: {}", req.method)
            })),
        },
    };

    Some(serde_json::to_string(&resp).unwrap())
}

#[allow(dead_code)]
pub fn handle_request(
    raw_json: &str,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    cache: &mut witchcraft::EmbeddingsCache,
) -> Option<String> {
    dispatch_request(raw_json, |id, params| {
        execute_tool_call(id, params, db, embedder, cache)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::path::Path;
    use std::sync::OnceLock;

    fn shared_embedder() -> &'static witchcraft::Embedder {
        static EMBEDDER: OnceLock<witchcraft::Embedder> = OnceLock::new();
        EMBEDDER.get_or_init(|| {
            let device = witchcraft::make_device();
            witchcraft::Embedder::new(&device, Path::new("assets")).unwrap()
        })
    }

    fn dummy_on_tool_call(
        id: Option<JsonRpcId>,
        _params: Option<Value>,
    ) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({})),
            error: None,
        }
    }

    #[test]
    fn test_dispatch_unparseable_json_returns_parse_error_32700() {
        let raw = r#"{"jsonrpc":"2.0","id":9,"method":"#;
        let resp_str = dispatch_request(raw, dummy_on_tool_call)
            .expect("unparseable JSON must return a -32700 error response, not None");
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["error"]["code"], -32700);
        assert!(resp["id"].is_null());
    }

    #[test]
    fn test_dispatch_missing_method_returns_invalid_request_32600() {
        let raw = r#"{"jsonrpc":"2.0","id":10}"#;
        let resp_str = dispatch_request(raw, dummy_on_tool_call)
            .expect("request missing method must return -32600, not None");
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["error"]["code"], -32600);
        assert_eq!(resp["id"], 10);
    }

    #[test]
    fn test_dispatch_batch_array_returns_invalid_request_32600() {
        let raw = r#"[{"jsonrpc":"2.0","id":11,"method":"ping"}]"#;
        let resp_str = dispatch_request(raw, dummy_on_tool_call)
            .expect("batch request array must not be silently dropped");
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["error"]["code"], -32600);
    }

    #[test]
    fn test_dispatch_explicit_null_id_returns_response() {
        let raw = r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#;
        let resp_str = dispatch_request(raw, dummy_on_tool_call)
            .expect("request with explicit id:null is not a notification and must return a response");
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.get("result").is_some() || resp.get("error").is_some());
    }

    #[test]
    fn test_dispatch_negative_and_float_id_returns_response() {
        let raw_neg = r#"{"jsonrpc":"2.0","id":-5,"method":"ping"}"#;
        let resp_str = dispatch_request(raw_neg, dummy_on_tool_call)
            .expect("request with negative integer id must not be silently dropped");
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["id"], -5);
        assert_eq!(resp["result"], json!({}));

        let raw_float = r#"{"jsonrpc":"2.0","id":1.5,"method":"ping"}"#;
        let resp_float_str = dispatch_request(raw_float, dummy_on_tool_call)
            .expect("request with fractional id must not be silently dropped");
        let resp_float: Value = serde_json::from_str(&resp_float_str).unwrap();
        assert_eq!(resp_float["id"], 1.5);
    }

    #[test]
    fn test_dispatch_genuine_notification_returns_none() {
        let raw = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert!(dispatch_request(raw, dummy_on_tool_call).is_none());
    }

    #[test]
    fn test_handle_request_unknown_tool_returns_32602() {
        let dir = tempfile::tempdir().unwrap();
        let db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        let req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        });
        let resp_str = handle_request(&req.to_string(), &db, embedder, &mut cache).unwrap();
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["error"]["code"], -32602);
    }

    #[test]
    fn test_handle_request_missing_or_empty_query_returns_32602() {
        let dir = tempfile::tempdir().unwrap();
        let db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        for args in [json!({}), json!({"query": ""}), json!({"query": "   "})] {
            let req = json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search_ai_history",
                    "arguments": args
                }
            });
            let resp_str = handle_request(&req.to_string(), &db, embedder, &mut cache).unwrap();
            let resp: Value = serde_json::from_str(&resp_str).unwrap();
            assert_eq!(
                resp["error"]["code"], -32602,
                "expected -32602 for invalid search_ai_history arguments {args}, got {resp}"
            );
        }
    }

    #[test]
    fn test_handle_request_invalid_limit_type_returns_32602() {
        let dir = tempfile::tempdir().unwrap();
        let db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        for bad_limit in [json!("5"), json!(-1), json!(true)] {
            let req = json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "search_ai_history",
                    "arguments": {
                        "query": "auth",
                        "limit": bad_limit
                    }
                }
            });
            let resp_str = handle_request(&req.to_string(), &db, embedder, &mut cache).unwrap();
            let resp: Value = serde_json::from_str(&resp_str).unwrap();
            assert_eq!(
                resp["error"]["code"], -32602,
                "expected -32602 for bad limit {bad_limit}, got {resp}"
            );
        }
    }

    #[test]
    fn test_handle_request_missing_session_id_returns_32602() {
        let dir = tempfile::tempdir().unwrap();
        let db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        let req = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "read_session",
                "arguments": {}
            }
        });
        let resp_str = handle_request(&req.to_string(), &db, embedder, &mut cache).unwrap();
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["error"]["code"], -32602);
    }

    #[test]
    fn test_handle_request_runtime_failure_returns_is_error_true() {
        let dir = tempfile::tempdir().unwrap();
        let db = witchcraft::DB::new(dir.path().join("test.db")).unwrap();
        let embedder = shared_embedder();
        let mut cache = witchcraft::EmbeddingsCache::new(1);

        let req = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "read_session",
                "arguments": {
                    "session_id": "nope"
                }
            }
        });
        let resp_str = handle_request(&req.to_string(), &db, embedder, &mut cache).unwrap();
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert!(
            resp.get("error").is_none(),
            "runtime tool failure must not return JSON-RPC error: {resp}"
        );
        assert_eq!(resp["result"]["isError"], true);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap_or("");
        assert!(text.contains("nope"), "expected error text to mention session id, got: {text}");
    }
}
