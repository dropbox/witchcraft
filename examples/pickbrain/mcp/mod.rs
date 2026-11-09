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
