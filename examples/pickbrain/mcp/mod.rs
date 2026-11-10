use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Number, Value};
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::SystemTime;

pub mod tools;

const MAX_RPC_LINE_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_PROTOCOL_VERSION: &str = "2024-11-05";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum JsonRpcId {
    Number(Number),
    String(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<JsonRpcId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<JsonRpcId>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<JsonRpcId>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(json!({
                "code": code,
                "message": message.into()
            })),
        }
    }

    pub fn tool_error(id: Option<JsonRpcId>, message: impl Into<String>) -> Self {
        Self::success(
            id,
            json!({
                "content": [{ "type": "text", "text": message.into() }],
                "isError": true
            }),
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
struct FileIdentity {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev_ino: (u64, u64),
}

impl FileIdentity {
    fn from_path(path: &Path) -> io::Result<Self> {
        let meta = fs::metadata(path)?;
        #[cfg(unix)]
        let dev_ino = {
            use std::os::unix::fs::MetadataExt;
            (meta.dev(), meta.ino())
        };
        Ok(Self {
            len: meta.len(),
            modified: meta.modified().ok(),
            #[cfg(unix)]
            dev_ino,
        })
    }
}

/// Read-only DB handle that automatically reopens when the underlying file is recreated or modified.
pub struct StaleAwareDbReader {
    db_path: PathBuf,
    current: Option<(witchcraft::DB, FileIdentity)>,
}

impl StaleAwareDbReader {
    pub fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            current: None,
        }
    }

    pub fn get(&mut self) -> std::result::Result<&witchcraft::DB, String> {
        let identity = match FileIdentity::from_path(&self.db_path) {
            Ok(id) => id,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                self.current = None;
                return Err(format!(
                    "Database not found at {} — run 'pickbrain' once to build the index (and 'pickbrain --watch' to keep it fresh).",
                    self.db_path.display()
                ));
            }
            Err(e) => {
                self.current = None;
                return Err(format!(
                    "Unable to stat database at {}: {}",
                    self.db_path.display(),
                    e
                ));
            }
        };

        let needs_reopen = match &self.current {
            Some((_, prev_id)) => prev_id != &identity,
            None => true,
        };

        if needs_reopen {
            let db = witchcraft::DB::new_reader(self.db_path.clone()).map_err(|e| {
                format!(
                    "Backend initialization failed opening {}: {} (run 'pickbrain' once to build the index)",
                    self.db_path.display(),
                    e
                )
            })?;
            self.current = Some((db, identity));
        }

        Ok(&self.current.as_ref().unwrap().0)
    }
}

struct SearchWorkerRequest {
    params: tools::SearchParams,
    reply: mpsc::Sender<std::result::Result<Value, tools::ToolError>>,
}

struct BackgroundSearchWorker {
    tx: mpsc::Sender<SearchWorkerRequest>,
}

impl BackgroundSearchWorker {
    fn spawn(db_path: PathBuf, assets_path: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel::<SearchWorkerRequest>();
        let _ = std::thread::Builder::new()
            .name("pickbrain-mcp-embedder".to_string())
            .spawn(move || {
                lower_background_thread_priority();

                let init_embedder = |assets: &Path| {
                    let device = witchcraft::make_device();
                    witchcraft::Embedder::new(&device, assets)
                        .map(|emb| (emb, witchcraft::EmbeddingsCache::new(16)))
                        .map_err(|e| format!("Failed to initialize embedding model: {e}"))
                };

                // Eagerly initialize right away at lower priority so the model is warm before the first search tool call
                let mut embedder_state = Some(init_embedder(&assets_path));
                let mut db_reader = StaleAwareDbReader::new(db_path);

                while let Ok(req) = rx.recv() {
                    let res = match db_reader.get() {
                        Ok(db) => {
                            if embedder_state
                                .as_ref()
                                .is_none_or(|state| state.is_err())
                            {
                                embedder_state = Some(init_embedder(&assets_path));
                            }
                            match embedder_state.as_mut().unwrap() {
                                Ok((embedder, cache)) => {
                                    tools::execute_search(req.params, db, embedder, cache)
                                }
                                Err(e) => Err(tools::ToolError::ExecutionFailed(e.clone())),
                            }
                        }
                        Err(e) => Err(tools::ToolError::ExecutionFailed(e)),
                    };
                    let _ = req.reply.send(res);
                }
            });
        Self { tx }
    }

    fn search(
        &self,
        params: tools::SearchParams,
    ) -> std::result::Result<Value, tools::ToolError> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(SearchWorkerRequest {
                params,
                reply: reply_tx,
            })
            .map_err(|_| {
                tools::ToolError::ExecutionFailed(
                    "Background search worker thread terminated unexpectedly".to_string(),
                )
            })?;
        reply_rx.recv().map_err(|_| {
            tools::ToolError::ExecutionFailed(
                "Background search worker dropped response channel".to_string(),
            )
        })?
    }
}

fn lower_background_thread_priority() {
    #[cfg(target_os = "macos")]
    unsafe {
        const QOS_CLASS_UTILITY: libc::c_uint = 0x11;
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(
                qos_class: libc::c_uint,
                relative_priority: libc::c_int,
            ) -> libc::c_int;
        }
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }

    #[cfg(target_os = "linux")]
    unsafe {
        // On Linux, PRIO_PROCESS with who=0 adjusts the calling thread's nice value (+10).
        let _ = libc::setpriority(libc::PRIO_PROCESS, 0, 10);
    }

    #[cfg(windows)]
    unsafe {
        use windows::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
        };
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

pub fn start_server(db_name: &Path, assets: &Path) -> Result<()> {
    use std::io::IsTerminal;
    let stdin = io::stdin();
    if stdin.is_terminal() && io::stderr().is_terminal() {
        eprintln!(
            "pickbrain {} MCP server listening on stdio (JSON-RPC 2.0)...",
            env!("CARGO_PKG_VERSION")
        );
    }

    // Spawn the lower-priority background embedder worker immediately so model loading
    // runs in the background while handshake, ping, and DB-only tools remain instant.
    let worker = BackgroundSearchWorker::spawn(db_name.to_path_buf(), assets.to_path_buf());
    let mut db_reader = StaleAwareDbReader::new(db_name.to_path_buf());

    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = (&mut reader)
            .take((MAX_RPC_LINE_BYTES + 1) as u64)
            .read_line(&mut line)?;
        if bytes_read == 0 {
            break; // EOF
        }
        if bytes_read > MAX_RPC_LINE_BYTES && !line.ends_with('\n') {
            // Drain the remainder of the oversized line so subsequent requests stay framed
            let mut discard = [0u8; 8192];
            loop {
                let n = reader.read(&mut discard)?;
                if n == 0 || discard[..n].contains(&b'\n') {
                    break;
                }
            }
            let resp = JsonRpcResponse::error(
                None,
                -32600,
                format!("Invalid Request: line exceeds maximum length of {MAX_RPC_LINE_BYTES} bytes"),
            );
            println!("{}", serde_json::to_string(&resp)?);
            io::stdout().flush()?;
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }

        if let Some(resp_json) = dispatch_request(&line, |id, params| {
            execute_tool_call_with_worker(id, params, &mut db_reader, &worker)
        }) {
            println!("{resp_json}");
            io::stdout().flush()?;
        }
    }
    Ok(())
}

fn execute_tool_call_with_worker(
    id: Option<JsonRpcId>,
    params: Option<Value>,
    db_reader: &mut StaleAwareDbReader,
    worker: &BackgroundSearchWorker,
) -> JsonRpcResponse {
    let Some(params) = params.filter(|p| p.is_object()) else {
        return JsonRpcResponse::error(id, -32602, "Invalid params: expected JSON object for tools/call");
    };
    let Some(name) = params.get("name").and_then(|n| n.as_str()).filter(|s| !s.is_empty()) else {
        return JsonRpcResponse::error(id, -32602, "Invalid params: missing tool 'name'");
    };
    let args = params.get("arguments").cloned();

    let outcome = match name {
        "search_ai_history" => match tools::parse_search_params(args.as_ref()) {
            Ok(search_params) => {
                // Fast check for missing DB before waiting on the background embedder
                if let Err(e) = db_reader.get() {
                    Err(tools::ToolError::ExecutionFailed(e))
                } else {
                    worker.search(search_params)
                }
            }
            Err(e) => Err(e),
        },
        "read_session" | "list_projects" | "list_recent_sessions" => {
            // Validate arguments before opening the DB so invalid-params errors return -32602 first
            if let Err(e) = tools::validate_db_tool_args(name, args.as_ref()) {
                Err(e)
            } else {
                match db_reader.get() {
                    Ok(db) => tools::handle_db_only_tool_call(name, args, db),
                    Err(e) => Err(tools::ToolError::ExecutionFailed(e)),
                }
            }
        }
        _ => Err(tools::ToolError::InvalidParams(format!("Unknown tool: {name}"))),
    };

    tool_outcome_to_response(id, outcome)
}

fn execute_tool_call(
    id: Option<JsonRpcId>,
    params: Option<Value>,
    db: &witchcraft::DB,
    embedder: &witchcraft::Embedder,
    cache: &mut witchcraft::EmbeddingsCache,
) -> JsonRpcResponse {
    let Some(params) = params.filter(|p| p.is_object()) else {
        return JsonRpcResponse::error(id, -32602, "Invalid params: expected JSON object for tools/call");
    };
    let Some(name) = params.get("name").and_then(|n| n.as_str()).filter(|s| !s.is_empty()) else {
        return JsonRpcResponse::error(id, -32602, "Invalid params: missing tool 'name'");
    };
    let args = params.get("arguments").cloned();

    let outcome = tools::handle_tool_call_typed(name, args, db, embedder, cache);
    tool_outcome_to_response(id, outcome)
}

fn tool_outcome_to_response(
    id: Option<JsonRpcId>,
    outcome: std::result::Result<Value, tools::ToolError>,
) -> JsonRpcResponse {
    match outcome {
        Ok(result) => JsonRpcResponse::success(id, result),
        Err(tools::ToolError::InvalidParams(msg)) => JsonRpcResponse::error(id, -32602, msg),
        Err(tools::ToolError::ExecutionFailed(msg)) => JsonRpcResponse::tool_error(id, msg),
    }
}

fn negotiate_protocol_version(params: Option<&Value>) -> &'static str {
    if let Some(requested) = params
        .and_then(|p| p.get("protocolVersion"))
        .and_then(|v| v.as_str())
    {
        for &supported in SUPPORTED_PROTOCOL_VERSIONS {
            if requested == supported {
                return supported;
            }
        }
    }
    DEFAULT_PROTOCOL_VERSION
}

pub fn dispatch_request<F>(raw_json: &str, mut on_tool_call: F) -> Option<String>
where
    F: FnMut(Option<JsonRpcId>, Option<Value>) -> JsonRpcResponse,
{
    let val: Value = match serde_json::from_str(raw_json) {
        Ok(v) => v,
        Err(e) => {
            let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
            return Some(serde_json::to_string(&resp).unwrap());
        }
    };

    let Some(obj) = val.as_object() else {
        let resp = JsonRpcResponse::error(
            None,
            -32600,
            "Invalid Request: expected a JSON-RPC 2.0 request object",
        );
        return Some(serde_json::to_string(&resp).unwrap());
    };

    let has_id_key = obj.contains_key("id");
    let id = match obj.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::Number(n)) => Some(JsonRpcId::Number(n.clone())),
        Some(Value::String(s)) => Some(JsonRpcId::String(s.clone())),
        Some(_) => {
            let resp = JsonRpcResponse::error(
                None,
                -32600,
                "Invalid Request: 'id' must be a string, number, or null",
            );
            return Some(serde_json::to_string(&resp).unwrap());
        }
    };

    if obj.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
        if has_id_key {
            let resp = JsonRpcResponse::error(
                id,
                -32600,
                "Invalid Request: 'jsonrpc' must be \"2.0\"",
            );
            return Some(serde_json::to_string(&resp).unwrap());
        }
        return None;
    }

    let Some(method) = obj.get("method").and_then(|m| m.as_str()) else {
        if has_id_key {
            let resp = JsonRpcResponse::error(
                id,
                -32600,
                "Invalid Request: missing or non-string 'method'",
            );
            return Some(serde_json::to_string(&resp).unwrap());
        }
        return None;
    };

    // Genuine JSON-RPC 2.0 notification (no 'id' member present)
    if !has_id_key {
        return None;
    }

    let params = obj.get("params").cloned();

    let resp = match method {
        "initialize" => {
            let negotiated = negotiate_protocol_version(params.as_ref());
            JsonRpcResponse::success(
                id,
                json!({
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    },
                    "serverInfo": {
                        "name": "pickbrain",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
        }
        "ping" => JsonRpcResponse::success(id, json!({})),
        "resources/list" => JsonRpcResponse::success(id, json!({ "resources": [] })),
        "prompts/list" => JsonRpcResponse::success(id, json!({ "prompts": [] })),
        "tools/list" => JsonRpcResponse::success(id, tools::list_tools()),
        "tools/call" => on_tool_call(id, params),
        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {method}")),
    };

    Some(serde_json::to_string(&resp).unwrap())
}

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
    use std::cell::RefCell;
    use std::path::Path;
    use std::rc::Rc;

    pub(crate) fn shared_embedder() -> Rc<witchcraft::Embedder> {
        thread_local! {
            static EMBEDDER: RefCell<Option<Rc<witchcraft::Embedder>>> = const { RefCell::new(None) };
        }
        EMBEDDER.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                let device = witchcraft::make_device();
                let emb = witchcraft::Embedder::new(&device, Path::new("assets")).unwrap();
                *opt = Some(Rc::new(emb));
            }
            Rc::clone(opt.as_ref().unwrap())
        })
    }

    fn dummy_on_tool_call(
        id: Option<JsonRpcId>,
        _params: Option<Value>,
    ) -> JsonRpcResponse {
        JsonRpcResponse::success(id, json!({}))
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
    fn test_initialize_negotiates_protocol_version() {
        let req_2025 = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": "2025-06-18" }
        });
        let resp_str = dispatch_request(&req_2025.to_string(), dummy_on_tool_call).unwrap();
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(resp["result"]["capabilities"]["tools"]["listChanged"], false);
    }

    #[test]
    fn test_stale_aware_db_reader_reopens_on_recreate_and_hints_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("pickbrain.db");
        let mut reader = StaleAwareDbReader::new(db_path.clone());

        // 1. Missing DB gives actionable hint
        let err = match reader.get() {
            Ok(_) => panic!("expected error when DB does not exist"),
            Err(e) => e,
        };
        assert!(
            err.contains("run 'pickbrain' once to build the index"),
            "unexpected error message: {err}"
        );

        // 2. Create initial DB with 1 document
        {
            let mut db = witchcraft::DB::new(db_path.clone()).unwrap();
            let meta = json!({"project": "proj-v1", "session_id": "s1", "turn": 0}).to_string();
            db.add_doc(
                None,
                &uuid::Uuid::from_u128(1),
                None,
                &meta,
                "first version",
                Some(vec![13]),
            )
            .unwrap();
        }
        let db_ref = reader.get().unwrap();
        let res_v1 = tools::handle_db_only_tool_call("list_projects", None, db_ref).unwrap();
        assert!(res_v1["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("proj-v1"));

        // 3. Remove and recreate DB at a new inode / size with different project
        std::fs::remove_file(&db_path).unwrap();
        witchcraft::DB::remove_all_sidecars(&db_path);
        {
            let mut db2 = witchcraft::DB::new(db_path.clone()).unwrap();
            let meta1 = json!({"project": "proj-v2", "session_id": "s2", "turn": 0}).to_string();
            let meta2 = json!({"project": "proj-v2", "session_id": "s3", "turn": 0}).to_string();
            let body_long = "second version with different file size".repeat(20);
            db2.add_doc(
                None,
                &uuid::Uuid::from_u128(2),
                None,
                &meta1,
                &body_long,
                Some(vec![body_long.len()]),
            )
            .unwrap();
            db2.add_doc(
                None,
                &uuid::Uuid::from_u128(3),
                None,
                &meta2,
                &body_long,
                Some(vec![body_long.len()]),
            )
            .unwrap();
        }

        let db_ref2 = reader.get().unwrap();
        let res_v2 = tools::handle_db_only_tool_call("list_projects", None, db_ref2).unwrap();
        let text_v2 = res_v2["content"][0]["text"].as_str().unwrap();
        assert!(
            text_v2.contains("proj-v2 (2 sessions)"),
            "expected StaleAwareDbReader to reopen recreated DB and see proj-v2, got: {text_v2}"
        );
        assert!(!text_v2.contains("proj-v1"));
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
        let resp_str = handle_request(&req.to_string(), &db, &embedder, &mut cache).unwrap();
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
            let resp_str = handle_request(&req.to_string(), &db, &embedder, &mut cache).unwrap();
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
            let resp_str = handle_request(&req.to_string(), &db, &embedder, &mut cache).unwrap();
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
        let resp_str = handle_request(&req.to_string(), &db, &embedder, &mut cache).unwrap();
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
        let resp_str = handle_request(&req.to_string(), &db, &embedder, &mut cache).unwrap();
        let resp: Value = serde_json::from_str(&resp_str).unwrap();
        assert!(
            resp.get("error").is_none(),
            "runtime tool failure must not return JSON-RPC error: {resp}"
        );
        assert_eq!(resp["result"]["isError"], true);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap_or("");
        assert!(
            text.contains("nope"),
            "expected error text to mention session id, got: {text}"
        );
    }
}
