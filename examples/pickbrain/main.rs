use anyhow::{Context, Result};
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::io::Write;
use std::path::PathBuf;

mod claude_code;
mod codex;
mod pi;
mod slack;
mod watermark;

use witchcraft::{DB, Embedder};

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;
const PICKBRAIN_DIR_ENV: &str = "PICKBRAIN_DIR";

fn pickbrain_dir_overridden() -> bool {
    env::var(PICKBRAIN_DIR_ENV)
        .map(|dir| !dir.trim().is_empty())
        .unwrap_or(false)
}

pub(crate) fn print_ingest_path(path: &std::path::Path, quiet: bool) {
    if !quiet {
        println!("{}", path.display());
    }
}

pub(crate) fn pickbrain_dir() -> PathBuf {
    if let Ok(dir) = env::var(PICKBRAIN_DIR_ENV) {
        if !dir.trim().is_empty() {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).ok();
            return dir;
        }
    }
    let home = env::var("HOME").unwrap_or_default();
    let dir = PathBuf::from(home).join(".pickbrain");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn db_path() -> PathBuf {
    let dir = pickbrain_dir();
    dir.join("pickbrain.db")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

struct IngestLock {
    db: DB,
}

impl IngestLock {
    fn try_acquire(db_name: &PathBuf) -> Result<Option<Self>> {
        let db = DB::new(db_name.clone())?;
        if db.was_recreated() {
            reset_ingest_watermarks();
        }

        Self::ensure_table(&db)?;
        if Self::insert_lock(&db)? {
            return Ok(Some(Self { db }));
        }

        if let Some(pid) = Self::locked_by(&db)? {
            if !process_is_running(pid) {
                Self::clear_lock(&db, pid)?;
                if Self::insert_lock(&db)? {
                    return Ok(Some(Self { db }));
                }
            }
        }

        Ok(None)
    }

    fn ensure_table(db: &DB) -> Result<()> {
        db.execute(
            "CREATE TABLE IF NOT EXISTS pickbrain_ingest_lock(
                id INTEGER PRIMARY KEY CHECK(id = 1),
                pid INTEGER NOT NULL,
                created_ms INTEGER NOT NULL)",
        )?;
        Ok(())
    }

    fn insert_lock(db: &DB) -> Result<bool> {
        let pid = std::process::id() as i64;
        let created_ms = now_ms();
        let mut stmt = db.query(
            "INSERT OR IGNORE INTO pickbrain_ingest_lock(id, pid, created_ms)
             VALUES(1, ?1, ?2)",
        )?;
        Ok(stmt.execute((pid, created_ms))? == 1)
    }

    fn locked_by(db: &DB) -> Result<Option<u32>> {
        let mut stmt = db.query("SELECT pid FROM pickbrain_ingest_lock WHERE id = 1")?;
        let mut rows = stmt.query(())?;
        if let Some(row) = rows.next()? {
            let pid: i64 = row.get(0)?;
            if pid > 0 && pid <= u32::MAX as i64 {
                return Ok(Some(pid as u32));
            }
        }
        Ok(None)
    }

    fn clear_lock(db: &DB, pid: u32) -> Result<()> {
        let mut stmt = db.query("DELETE FROM pickbrain_ingest_lock WHERE id = 1 AND pid = ?1")?;
        stmt.execute((pid as i64,))?;
        Ok(())
    }
}

impl Drop for IngestLock {
    fn drop(&mut self) {
        let pid = std::process::id() as i64;
        if let Ok(mut stmt) =
            self.db
                .query("DELETE FROM pickbrain_ingest_lock WHERE id = 1 AND pid = ?1")
        {
            let _ = stmt.execute((pid,));
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn process_is_running(pid: u32) -> bool {
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_is_running(_pid: u32) -> bool {
    true
}

fn wants_source(types: &[String], source: &str) -> bool {
    types.is_empty() || types.iter().any(|t| t == source)
}

fn needs_ingest(
    db_name: &PathBuf,
    skip_session: Option<&str>,
    stale_ms: i64,
    types: &[String],
) -> Result<bool> {
    if !db_name.exists() {
        return Ok(true);
    }

    if wants_source(types, "claude") {
        let claude_skip = skip_session
            .filter(|_| watermark::is_fresh(&watermark::claude_path(), stale_ms));
        if claude_code::has_work(claude_skip)? {
            return Ok(true);
        }
    }

    if wants_source(types, "codex") && codex::has_work() {
        return Ok(true);
    }

    if wants_source(types, "pi") {
        let pi_skip = skip_session.filter(|_| watermark::is_fresh(&watermark::pi_path(), stale_ms));
        if pi::has_work(pi_skip) {
            return Ok(true);
        }
    }

    if wants_source(types, "slack") && slack::has_work() {
        return Ok(true);
    }

    Ok(false)
}

fn warn_lookup_only(reason: impl std::fmt::Display) {
    eprintln!("warning: ingest needed but {reason}; using lookup-only mode");
}

fn reset_ingest_watermarks() {
    watermark::remove(&watermark::claude_path());
    watermark::remove(&watermark::codex_path());
    watermark::remove(&watermark::pi_path());
    slack::remove_watermark();
}

/// Walk up the process tree to find the calling Claude Code session ID.
fn detect_active_session() -> Option<String> {
    for key in ["PICKBRAIN_ACTIVE_SESSION_ID", "PI_SESSION_ID"] {
        if let Ok(value) = env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    for key in ["PICKBRAIN_ACTIVE_SESSION_FILE", "PI_SESSION_FILE"] {
        if let Ok(value) = env::var(key) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(pi::session_id_from_filename(std::path::Path::new(value)));
            }
        }
    }

    let home = env::var("HOME").ok()?;
    let sessions_dir = PathBuf::from(&home).join(".claude/sessions");
    let mut pid = std::process::id() as i32;
    while pid > 1 {
        let session_file = sessions_dir.join(format!("{pid}.json"));
        if let Ok(data) = std::fs::read_to_string(&session_file) {
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            return v["sessionId"].as_str().map(|s| s.to_string());
        }
        pid = get_ppid(pid)?;
    }
    None
}

#[cfg(target_os = "macos")]
fn get_ppid(pid: i32) -> Option<i32> {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as i32;
    let ret = unsafe {
        libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0,
            &mut info as *mut _ as *mut libc::c_void, size)
    };
    if ret == size {
        let ppid = info.pbi_ppid as i32;
        if ppid > 0 { Some(ppid) } else { None }
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn get_ppid(pid: i32) -> Option<i32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // Field 4 (0-indexed: 3) is ppid. Fields 1 is (comm) which may contain spaces,
    // so skip past the closing paren first.
    let after_comm = stat.rfind(')')? + 2;
    let fields: Vec<&str> = stat[after_comm..].split_whitespace().collect();
    // fields[0] = state, fields[1] = ppid
    let ppid: i32 = fields.get(1)?.parse().ok()?;
    if ppid > 0 { Some(ppid) } else { None }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_ppid(_pid: i32) -> Option<i32> {
    None
}

fn ingest(
    db_name: &PathBuf,
    skip_session: Option<&str>,
    stale_ms: i64,
    types: &[String],
    quiet: bool,
) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();

    let want = |src: &str| types.is_empty() || types.iter().any(|t| t == src);

    if db.was_recreated() {
        reset_ingest_watermarks();
    }

    let (sessions, memories, authored, configs) = if want("claude") {
        let claude_skip = skip_session
            .filter(|_| watermark::is_fresh(&watermark::claude_path(), stale_ms));
        claude_code::ingest_claude_code(&mut db, claude_skip, quiet)?
    } else {
        (0, 0, 0, 0)
    };

    let codex_sessions = if want("codex") {
        codex::ingest_codex(&mut db, quiet)?
    } else {
        0
    };

    let pi_sessions = if want("pi") {
        let pi_skip = skip_session.filter(|_| watermark::is_fresh(&watermark::pi_path(), stale_ms));
        pi::ingest_pi(&mut db, pi_skip, quiet)?
    } else {
        0
    };

    let slack_conversations = if want("slack") {
        slack::ingest_slack(&mut db)?
    } else {
        0
    };

    let total = sessions + memories + authored + configs + codex_sessions + pi_sessions + slack_conversations;
    if total == 0 {
        if !quiet {
            eprintln!("No new sessions to ingest.");
        }
        return Ok(false);
    }
    if !quiet {
        eprintln!(
            "ingested {sessions} claude sessions, {codex_sessions} codex sessions, {pi_sessions} pi sessions, {slack_conversations} slack conversations, {memories} memory files, {authored} authored files, {configs} config files"
        );
    }
    Ok(true)
}

fn embed_and_index(db: &DB, embedder: &Embedder, _device: &candle_core::Device) -> Result<()> {
    let _embedded = witchcraft::embed_chunks(db, embedder, None)?;
    let options = witchcraft::IndexOptions::default().force_flush();
    witchcraft::index_chunks_with_options(db, None, false, options)?;
    Ok(())
}

// --- Search result data ---

#[derive(Clone)]
struct TurnMeta {
    role: String,
    timestamp: String,
    byte_offset: u64,
    byte_len: u64,
}

#[derive(Clone)]
struct SearchResult {
    timestamp: String,
    project: String,
    session_id: String,
    session_name: String,
    turn: u64,
    path: String,
    cwd: String,
    source: String,
    branch: String,
    conv_key: String,
    slack_open: Option<SlackOpenTarget>,
    bodies: Vec<String>,
    match_idx: usize,
    turns: Vec<TurnMeta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SlackOpenTarget {
    team_id: String,
    channel_id: String,
    root_ts: String,
    conv_key: String,
}

// A turn from the original JSONL session file
struct SessionTurn {
    role: String,
    text: String,
    timestamp: String,
}

fn read_jsonl_line(path: &str, offset: u64, len: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn extract_pi_content_text(content: &serde_json::Value) -> Option<String> {
    if content.is_string() {
        return content.as_str().map(|s| s.to_string());
    }
    let blocks = content.as_array()?;
    let texts: Vec<&str> = blocks
        .iter()
        .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => b.get("text").and_then(|t| t.as_str()),
            Some("thinking") => b.get("thinking").and_then(|t| t.as_str()),
            _ => None,
        })
        .collect();
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

fn read_turn_at(path: &str, source: &str, tm: &TurnMeta) -> Option<SessionTurn> {
    let line = read_jsonl_line(path, tm.byte_offset, tm.byte_len)?;
    let v: serde_json::Value = serde_json::from_str(&line).ok()?;

    let text = if source == "codex" {
        let payload = v.get("payload")?;
        let ptype = payload.get("type")?.as_str()?;
        if ptype == "message" {
            let text_type = match payload.get("role")?.as_str()? {
                "user" => "input_text",
                "assistant" => "output_text",
                _ => return None,
            };
            let content = payload.get("content")?.as_array()?;
            let texts: Vec<&str> = content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some(text_type))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect();
            texts.join("\n")
        } else if ptype == "agent_reasoning" {
            payload.get("text")?.as_str()?.to_string()
        } else {
            return None;
        }
    } else if source == "pi" {
        let content = if v.get("type").and_then(|t| t.as_str()) == Some("custom_message") {
            v.get("content")?
        } else {
            v.get("message")?.get("content")?
        };
        extract_pi_content_text(content)?
    } else {
        // Claude Code
        let msg = v.get("message")?;
        match msg.get("content")? {
            c if c.is_string() => c.as_str()?.to_string(),
            c if c.is_array() => {
                let blocks = c.as_array()?;
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => return None,
        }
    };

    Some(SessionTurn {
        role: tm.role.clone(),
        text,
        timestamp: tm.timestamp.clone(),
    })
}


/// Parse a duration string like "24h", "7d", "2w" into milliseconds.
fn parse_since(s: &str) -> Option<i64> {
    let s = s.trim();
    let (num_str, unit) = s.split_at(s.len().saturating_sub(1));
    let n: i64 = num_str.parse().ok()?;
    let ms_per_unit = match unit {
        "h" => 3_600_000,
        "d" => 86_400_000,
        "w" => 604_800_000,
        _ => return None,
    };
    Some(n * ms_per_unit)
}

fn build_sql_filter(
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Option<witchcraft::types::SqlStatementInternal> {
    use witchcraft::types::*;
    let mut conditions: Vec<SqlStatementInternal> = Vec::new();
    if let Some(id) = session {
        let name = id.strip_prefix('#').unwrap_or(id);
        let thread_ts = name.strip_prefix("thr:").unwrap_or(name);
        let thread_like = format!("%-{thread_ts}");
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::Or),
            statements: Some(vec![
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.session_id".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(id.to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.channel_id".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(id.to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.channel_name".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(name.to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.conv_key".to_string(),
                        operator: SqlOperator::Like,
                        value: Some(SqlValue::String(thread_like)),
                    }),
                    logic: None,
                    statements: None,
                },
            ]),
        });
    }
    if let Some(br) = branch {
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.branch".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String(br.to_string())),
            }),
            logic: None,
            statements: None,
        });
    }
    for id in exclude {
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.session_id".to_string(),
                operator: SqlOperator::NotEquals,
                value: Some(SqlValue::String(id.clone())),
            }),
            logic: None,
            statements: None,
        });
    }
    if let Some(ms) = since_ms {
        let cutoff_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - ms / 1000;
        let cutoff_dt = chrono::DateTime::from_timestamp(cutoff_secs, 0).unwrap();
        let cutoff_iso = cutoff_dt.to_rfc3339();
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "date".to_string(),
                operator: SqlOperator::GreaterThanOrEquals,
                value: Some(SqlValue::String(cutoff_iso)),
            }),
            logic: None,
            statements: None,
        });
    }
    if !types.is_empty() {
        if types.len() == 1 {
            conditions.push(SqlStatementInternal {
                statement_type: SqlStatementType::Condition,
                condition: Some(SqlConditionInternal {
                    key: "$.source".to_string(),
                    operator: SqlOperator::Equals,
                    value: Some(SqlValue::String(types[0].clone())),
                }),
                logic: None,
                statements: None,
            });
        } else {
            let stmts: Vec<SqlStatementInternal> = types
                .iter()
                .map(|t| SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.source".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String(t.clone())),
                    }),
                    logic: None,
                    statements: None,
                })
                .collect();
            conditions.push(SqlStatementInternal {
                statement_type: SqlStatementType::Group,
                condition: None,
                logic: Some(SqlLogic::Or),
                statements: Some(stmts),
            });
        }
    }
    if dm_only {
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.is_dm".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::Number(1.0)),
            }),
            logic: None,
            statements: None,
        });
    } else if no_dm {
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.is_dm".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::Number(0.0)),
            }),
            logic: None,
            statements: None,
        });
    }
    if unread_only {
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.has_unread".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::Number(1.0)),
            }),
            logic: None,
            statements: None,
        });
    }
    if conditions.is_empty() {
        None
    } else if conditions.len() == 1 {
        Some(conditions.remove(0))
    } else {
        Some(SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::And),
            statements: Some(conditions),
        })
    }
}

fn parse_search_results(
    results: Vec<(f32, String, Vec<String>, u32, String)>,
) -> Vec<SearchResult> {
    results
        .into_iter()
        .map(|(_score, metadata, bodies, sub_idx, date)| {
            let meta: serde_json::Value = serde_json::from_str(&metadata).unwrap_or_default();
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            let turns_arr: Vec<TurnMeta> = meta["turns"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|v| TurnMeta {
                            role: v["role"].as_str().unwrap_or("").to_string(),
                            timestamp: v["timestamp"].as_str().unwrap_or("").to_string(),
                            byte_offset: v["off"].as_u64().unwrap_or(0),
                            byte_len: v["len"].as_u64().unwrap_or(0),
                        })
                        .collect()
                })
                .unwrap_or_default();
            SearchResult {
                timestamp: format_date(&date),
                project: meta["project"].as_str().unwrap_or("").to_string(),
                session_id: meta["session_id"].as_str().unwrap_or("").to_string(),
                session_name: meta["session_name"].as_str().unwrap_or("").to_string(),
                turn: meta["turn"].as_u64().unwrap_or(0),
                path: meta["path"].as_str().unwrap_or("").to_string(),
                cwd: meta["cwd"].as_str().unwrap_or("").to_string(),
                source: meta["source"].as_str().unwrap_or("claude").to_string(),
                branch: meta["branch"].as_str().unwrap_or("").to_string(),
                conv_key: meta["conv_key"].as_str().unwrap_or("").to_string(),
                slack_open: slack_open_target_from_metadata(&meta),
                bodies,
                match_idx: idx,
                turns: turns_arr,
            }
        })
        .collect()
}

fn run_search(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    num_results: usize,
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<(Vec<SearchResult>, u128)> {
    let device = witchcraft::make_device();
    let embedder = witchcraft::Embedder::new(&device, assets)?;
    let db = match DB::new_reader(db_name.clone()) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "warning: lookup-only mode could not open {}: {e}",
                db_name.display()
            );
            return Ok((Vec::new(), 0));
        }
    };
    let sql_filter = build_sql_filter(session, branch, exclude, since_ms, types, dm_only, no_dm, unread_only);
    let effective_limit = if num_results == 0 { usize::MAX } else { num_results };
    let now = std::time::Instant::now();
    let results = if q.is_empty() {
        use witchcraft::sql_generator::build_filter_sql_and_params;
        let (filter_sql, params) = build_filter_sql_and_params(sql_filter.as_ref())?;
        let where_clause = if filter_sql.is_empty() { String::new() } else { format!("WHERE {filter_sql}") };
        let limit_clause = if num_results == 0 { String::new() } else { " LIMIT ?".to_string() };
        let sql = format!(
            "SELECT metadata, body, lens, date FROM document {where_clause} ORDER BY date DESC{limit_clause}",
        );
        let mut stmt = db.query(&sql)?;
        let mut param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| &**p as &dyn rusqlite::ToSql).collect();
        let limit = num_results as i64;
        if num_results > 0 {
            param_refs.push(&limit);
        }
        let rows: Vec<(f32, String, Vec<String>, u32, String)> = stmt
            .query_map(param_refs.as_slice(), |row| {
                let metadata: String = row.get(0)?;
                let body: String = row.get(1)?;
                let lens_str: String = row.get(2)?;
                let date: String = row.get(3)?;
                let lens: Vec<usize> = lens_str.split(',').map(|x| x.parse::<usize>().unwrap_or(0)).collect();
                let bodies: Vec<String> = witchcraft::split_by_codepoints(&body, &lens)
                    .into_iter().map(|s| s.to_string()).collect();
                Ok((0.0f32, metadata, bodies, 0u32, date))
            })?
            .filter_map(|r| r.ok())
            .collect();
        rows
    } else {
        witchcraft::search(
            &db,
            Some(&embedder),
            &mut witchcraft::EmbeddingsCache::new(1),
            q,
            0.5,
            effective_limit,
            true,
            sql_filter.as_ref(),
        )?
    };
    let search_ms = now.elapsed().as_millis();
    Ok((parse_search_results(results), search_ms))
}

fn run_search_with(
    db: &DB,
    embedder: &Embedder,
    q: &str,
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<(Vec<SearchResult>, u128)> {
    let mut cache = witchcraft::EmbeddingsCache::new(1);
    let sql_filter = build_sql_filter(session, branch, exclude, since_ms, types, dm_only, no_dm, unread_only);
    let now = std::time::Instant::now();
    let results = witchcraft::search(
        db,
        Some(embedder),
        &mut cache,
        q,
        0.5,
        10,
        true,
        sql_filter.as_ref(),
    )?;
    let search_ms = now.elapsed().as_millis();
    Ok((parse_search_results(results), search_ms))
}

// --- TUI ---

enum View {
    List,
    Detail(usize),
}

fn search_tui(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: Option<&str>,
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    num_results: usize,
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<Option<BranchSession>> {
    let device = witchcraft::make_device();
    let embedder = witchcraft::Embedder::new(&device, assets)?;
    let db = match DB::new_reader(db_name.clone()) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "warning: lookup-only mode could not open {}: {e}",
                db_name.display()
            );
            return Ok(None);
        }
    };
    let (mut results, mut search_ms) = match q {
        Some(q) if !q.is_empty() => {
            run_search_with(&db, &embedder, q, session, branch, exclude, since_ms, types, dm_only, no_dm, unread_only)?
        }
        Some(_) => {
            // Empty query with filters: use filter-only search
            run_search(db_name, assets, "", session, branch, exclude, since_ms, types, num_results, dm_only, no_dm, unread_only)?
        }
        None => {
            (find_recent_sessions(db_name, branch)?, 0)
        }
    };
    if results.is_empty() {
        eprintln!("no results");
        return Ok(None);
    }
    let mut active_query = q.unwrap_or("sessions").to_string();

    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{ListState, Paragraph, Wrap};
    use ratatui::Terminal;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut view = View::List;
    let mut selected: usize = 0;
    let mut list_state = ListState::default();
    list_state.select(Some(0));
    let mut scroll_offset: usize = 0;
    let mut resume_session: Option<BranchSession> = None;
    let mut slack_to_open: Option<SlackOpenTarget> = None;
    let mut confirm_resume: Option<BranchSession> = None;
    let mut searching = false;
    let mut search_filter = String::new();
    let mut saved_search: Option<(String, Vec<SearchResult>, u128)> = None;
    struct DetailState {
        result_idx: usize,
        turns: Vec<SessionTurn>,
        highlight: usize,
    }
    let mut detail_cache: Option<DetailState> = None;

    loop {
        if selected >= results.len() {
            selected = results.len().saturating_sub(1);
        }
        list_state.select(if results.is_empty() { None } else { Some(selected) });

        terminal.draw(|f| {
            let area = f.area();
            let show_footer = confirm_resume.is_some() && matches!(view, View::Detail(_));
            let show_search = searching || !search_filter.is_empty();
            let current_result = match view {
                View::List => results.get(selected),
                View::Detail(idx) => results.get(idx),
            };
            let can_open_slack = current_result.and_then(|r| r.slack_open.as_ref()).is_some();
            let can_resume = current_result.map(can_resume_result).unwrap_or(false);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(if show_footer {
                    vec![Constraint::Length(2), Constraint::Length(if show_search { 1 } else { 0 }), Constraint::Min(0), Constraint::Length(1)]
                } else {
                    vec![Constraint::Length(2), Constraint::Length(if show_search { 1 } else { 0 }), Constraint::Min(0), Constraint::Length(0)]
                })
                .split(area);

            // Header
            let help_text = if searching {
                "type to search  ⏎ done  esc undo"
            } else {
                match view {
                    View::List => "↑↓/jk navigate  ⏎ open  / search  q quit",
                    View::Detail(_) if can_open_slack => {
                        "↑↓/jk scroll  r open slack  / search  esc back  q quit"
                    }
                    View::Detail(_) if can_resume => {
                        "↑↓/jk scroll  r resume  / search  esc back  q quit"
                    }
                    View::Detail(_) => "↑↓/jk scroll  / search  esc back  q quit",
                }
            };
            let stats = if search_ms > 0 {
                format!("  {} results  {} ms  ", results.len(), search_ms)
            } else {
                format!("  {} sessions  ", results.len())
            };
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("[[ {} ]]", active_query),
                    Style::default()
                        .fg(Color::Rgb(0, 255, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    stats,
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(help_text, Style::default().fg(Color::DarkGray)),
            ]));
            f.render_widget(header, chunks[0]);

            // Search input bar
            if show_search {
                let bar_style = if searching {
                    Style::default().bg(Color::DarkGray)
                } else {
                    Style::default()
                };
                let search_bar = Paragraph::new(Line::from(vec![
                    Span::styled("/ ", Style::default().fg(if searching { Color::White } else { Color::DarkGray })),
                    Span::styled(
                        &search_filter,
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                    if searching {
                        Span::styled("_", Style::default().fg(Color::DarkGray))
                    } else {
                        Span::raw("")
                    },
                ])).style(bar_style);
                f.render_widget(search_bar, chunks[1]);
            }

            let content_area = chunks[2];
            let footer_area = chunks[3];

            // Footer: resume confirmation
            if show_footer {
                let cr = confirm_resume.as_ref().unwrap();
                let cwd = if !cr.cwd.is_empty() { &cr.cwd } else { "?" };
                let sid = &cr.session_id;
                let src = &cr.source;
                let footer = Paragraph::new(Line::from(vec![
                    Span::styled(
                        format!(" Exit pickbrain and resume {src} session {sid} in {cwd}? "),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "(Y/n)",
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]));
                f.render_widget(footer, footer_area);
            }

            match view {
                View::List => {
                    let width = content_area.width as usize;
                    let items: Vec<ratatui::widgets::ListItem> = results
                        .iter()
                        .map(|r| {
                            let preview_idx = if r.match_idx == 0 && r.bodies.len() > 1 {
                                1
                            } else {
                                r.match_idx
                            };
                            let raw_preview = first_line(&r.bodies[preview_idx]);
                            let preview = strip_body_prefix(&raw_preview);
                            let matched_tm = if r.match_idx > 0 {
                                r.turns.get(r.match_idx - 1)
                            } else {
                                r.turns.first()
                            };
                            let ts = matched_tm
                                .filter(|tm| !tm.timestamp.is_empty())
                                .map(|tm| format_date(&tm.timestamp))
                                .unwrap_or_else(|| r.timestamp.clone());
                            let mut meta_spans = session_meta_spans(
                                &ts, &r.project, &r.session_id, &r.session_name, &r.source, &r.branch, &r.conv_key,
                            );
                            if r.path.ends_with(".md") {
                                meta_spans.push(Span::styled(
                                    format!("  {}", r.path),
                                    Style::default().fg(Color::Yellow),
                                ));
                            }
                            if !r.session_id.is_empty() {
                                meta_spans.push(Span::styled(
                                    format!("  turn {}", r.turn),
                                    Style::default().fg(Color::DarkGray),
                                ));
                            }
                            let match_role = matched_tm.map(|tm| tm.role.as_str()).unwrap_or("");
                            let role_prefix = if r.source == "slack" {
                                String::new()
                            } else if match_role == "user" {
                                "[User] ".to_string()
                            } else if match_role == "assistant" {
                                format!("{} ", role_label(&r.source))
                            } else {
                                String::new()
                            };
                            ratatui::widgets::ListItem::new(vec![
                                Line::from(meta_spans),
                                Line::from(vec![
                                    Span::styled(
                                        format!("  {role_prefix}"),
                                        Style::default().fg(if match_role == "user" {
                                            Color::Rgb(0, 255, 0)
                                        } else {
                                            Color::Cyan
                                        }),
                                    ),
                                    Span::raw(truncate(&preview, width.saturating_sub(4 + role_prefix.len()))),
                                ]),
                                Line::from(""),
                            ])
                        })
                        .collect();

                    let highlight = if searching {
                        Style::default()
                    } else {
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD)
                    };
                    let list = ratatui::widgets::List::new(items).highlight_style(highlight);
                    f.render_stateful_widget(list, content_area, &mut list_state);
                }
                View::Detail(idx) => {
                    let r = &results[idx];
                    let mut lines: Vec<Line> = Vec::new();

                    // Session header
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("{} ", r.timestamp),
                            Style::default().fg(Color::Green),
                        ),
                        Span::styled(&r.project, Style::default().fg(Color::Cyan)),
                    ]));
                    if !r.session_id.is_empty() {
                        let mut session_spans = vec![
                            Span::styled(&r.session_id, Style::default().fg(Color::Magenta)),
                        ];
                        if !r.session_name.is_empty() {
                            session_spans.push(Span::styled(
                                format!("  \"{}\"", r.session_name),
                                Style::default().fg(Color::White),
                            ));
                        }
                        session_spans.push(Span::styled(
                            format!("  turn {}", r.turn),
                            Style::default().fg(Color::DarkGray),
                        ));
                        if !r.branch.is_empty() {
                            session_spans.push(Span::styled(
                                format!("  {}", r.branch),
                                Style::default().fg(Color::Yellow),
                            ));
                        }
                        lines.push(Line::from(session_spans));
                    }
                    lines.push(Line::from(""));

                    // If we have a JSONL path and a session, show the real conversation
                    let dw = detail_cache
                        .as_ref()
                        .filter(|dw| dw.result_idx == idx);

                    if let Some(dw) = dw {
                        for (i, turn) in dw.turns.iter().enumerate() {
                            let is_highlight = i == dw.highlight;
                            let role_style = if turn.role == "user" {
                                Style::default()
                                    .fg(Color::Rgb(0, 255, 0))
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD)
                            };
                            let role_prefix = if r.source == "slack" {
                                String::new()
                            } else if turn.role == "user" {
                                "[User] ".to_string()
                            } else {
                                format!("{} ", role_label(&r.source))
                            };
                            lines.push(Line::from(vec![
                                Span::styled(role_prefix, role_style),
                                Span::styled(
                                    format_date(&turn.timestamp),
                                    Style::default().fg(Color::DarkGray),
                                ),
                            ]));
                            let text_style = if is_highlight {
                                Style::default().fg(Color::White)
                            } else {
                                Style::default().fg(Color::DarkGray)
                            };
                            for line in turn.text.lines() {
                                lines.push(Line::styled(format!("  {line}"), text_style));
                            }
                            lines.push(Line::from(""));
                        }
                    } else {
                        // Fallback: show indexed bodies (for .md files etc.)
                        for (i, chunk) in r.bodies.iter().enumerate() {
                            let style = if i == r.match_idx {
                                Style::default().add_modifier(Modifier::BOLD)
                            } else {
                                Style::default().fg(Color::DarkGray)
                            };
                            for line in chunk.lines().filter(|l| !l.is_empty()) {
                                lines.push(Line::styled(format!("  {line}"), style));
                            }
                            lines.push(Line::from(""));
                        }
                    }

                    let detail = Paragraph::new(lines)
                        .wrap(Wrap { trim: false })
                        .scroll((scroll_offset as u16, 0));
                    f.render_widget(detail, content_area);
                }
            }
        })?;

        if let Event::Key(key) = event::read()? {
            // Search mode: live-search as the user types
            if searching {
                match (key.code, key.modifiers) {
                    (KeyCode::Esc, _) => {
                        searching = false;
                        search_filter.clear();
                        if let Some((q, r, ms)) = saved_search.take() {
                            active_query = q;
                            results = r;
                            search_ms = ms;
                            selected = 0;
                            detail_cache = None;
                            view = View::List;
                        }
                        continue;
                    }
                    (KeyCode::Enter, _) => {
                        searching = false;
                        saved_search = None;
                        continue;
                    }
                    (KeyCode::Down, _) | (KeyCode::Up, _) => {
                        searching = false;
                        saved_search = None;
                        view = View::List;
                        continue;
                    }
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                    (KeyCode::Backspace, _) => {
                        search_filter.pop();
                        if search_filter.chars().count() >= 3 && search_filter != active_query {
                            if let Ok((new_results, ms)) = run_search_with(
                                &db, &embedder, &search_filter, session, branch, exclude, since_ms, types, dm_only, no_dm, unread_only,
                            ) {
                                active_query = search_filter.clone();
                                results = new_results;
                                search_ms = ms;
                                selected = 0;
                                detail_cache = None;
                                view = View::List;
                            }
                        } else if search_filter.is_empty() {
                            if let Some((ref q, ref r, ms)) = saved_search {
                                active_query = q.clone();
                                results = r.clone();
                                search_ms = ms;
                                selected = 0;
                                detail_cache = None;
                                view = View::List;
                            }
                        }
                        continue;
                    }
                    (KeyCode::Char(c), _) => {
                        search_filter.push(c);
                        if search_filter.chars().count() >= 3 && search_filter != active_query {
                            if let Ok((new_results, ms)) = run_search_with(
                                &db, &embedder, &search_filter, session, branch, exclude, since_ms, types, dm_only, no_dm, unread_only,
                            ) {
                                active_query = search_filter.clone();
                                results = new_results;
                                search_ms = ms;
                                selected = 0;
                                detail_cache = None;
                                view = View::List;
                            }
                        }
                        continue;
                    }
                    _ => { continue; }
                }
            }

            match (&view, key.code, key.modifiers) {
                (_, KeyCode::Char('q') | KeyCode::Esc, _) if confirm_resume.is_some() => {
                    confirm_resume = None;
                }
                (View::Detail(_), KeyCode::Char('q') | KeyCode::Esc, _) => {
                    view = View::List;
                }
                (View::List, KeyCode::Char('q'), _) => break,
                (View::List, KeyCode::Esc, _) => break,
                (_, KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                (_, KeyCode::Char('f'), KeyModifiers::CONTROL) |
                (_, KeyCode::Char('/'), _) => {
                    searching = true;
                    search_filter.clear();
                    saved_search = Some((active_query.clone(), results.clone(), search_ms));
                }
                (_, KeyCode::Char('z'), KeyModifiers::CONTROL) => {
                    disable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        LeaveAlternateScreen,
                        crossterm::cursor::Show
                    )?;
                    unsafe { libc::raise(libc::SIGTSTP); }
                    enable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        EnterAlternateScreen,
                        crossterm::cursor::Hide
                    )?;
                    terminal.clear()?;
                }

                // List view
                (View::List, KeyCode::Down | KeyCode::Char('j'), _) => {
                    if selected + 1 < results.len() {
                        selected += 1;
                        list_state.select(Some(selected));
                    }
                }
                (View::List, KeyCode::Up | KeyCode::Char('k'), _) => {
                    if selected > 0 {
                        selected -= 1;
                        list_state.select(Some(selected));
                    }
                }
                (View::List, KeyCode::Enter, _) => {
                    if selected < results.len() {
                        let r = &results[selected];
                        if !r.session_id.is_empty() && !r.path.is_empty() && !r.turns.is_empty() {
                            let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
                            let mut turns = Vec::new();
                            for (turn_idx, tm) in r.turns.iter().enumerate() {
                                if let Some(turn) = read_or_indexed_turn(r, turn_idx, tm) {
                                    turns.push(turn);
                                }
                            }
                            let mut pre_lines: usize = if r.session_id.is_empty() { 2 } else { 3 };
                            for t in &turns[..mi.min(turns.len())] {
                                pre_lines += 2 + t.text.lines().count();
                            }
                            scroll_offset = pre_lines;
                            detail_cache = Some(DetailState {
                                result_idx: selected,
                                turns,
                                highlight: mi,
                            });
                        } else {
                            scroll_offset = 0;
                            detail_cache = None;
                        }
                        view = View::Detail(selected);
                    }
                }

                // Detail view
                (View::Detail(_), KeyCode::Down | KeyCode::Char('j'), _) => {
                    scroll_offset = scroll_offset.saturating_add(1);
                }
                (View::Detail(_), KeyCode::Up | KeyCode::Char('k'), _) => {
                    scroll_offset = scroll_offset.saturating_sub(1);
                }
                (View::Detail(idx), KeyCode::Char('r'), _) => {
                    let r = &results[*idx];
                    if let Some(target) = r.slack_open.clone() {
                        slack_to_open = Some(target);
                        break;
                    } else if can_resume_result(r) {
                        let cwd = if !r.cwd.is_empty() {
                            r.cwd.clone()
                        } else {
                            read_cwd_from_jsonl(&r.path, &r.source)
                                .unwrap_or_default()
                        };
                        confirm_resume = Some(BranchSession {
                            session_id: r.session_id.clone(),
                            source: r.source.clone(),
                            branch: r.branch.clone(),
                            cwd,
                        });
                    }
                }
                (View::Detail(_), KeyCode::Char('y') | KeyCode::Enter, _) => {
                    if confirm_resume.is_some() {
                        resume_session = confirm_resume.take();
                        break;
                    }
                }
                (View::Detail(_), KeyCode::Char('n'), _) => {
                    confirm_resume = None;
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), LeaveAlternateScreen)?;
    if let Some(target) = slack_to_open {
        let url = open_slack_target(&target)?;
        eprintln!("opened {url}");
    }
    Ok(resume_session)
}

fn read_cwd_from_jsonl(path: &str, source: &str) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    for line in raw.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if source == "codex" {
            // Codex: cwd is in payload of session_meta entries
            if v.get("type").and_then(|t| t.as_str()) == Some("session_meta") {
                if let Some(cwd) = v.get("payload").and_then(|p| p.get("cwd")).and_then(|c| c.as_str()) {
                    return Some(cwd.to_string());
                }
            }
        } else if source == "pi" {
            // Pi: cwd is in the session header
            if v.get("type").and_then(|t| t.as_str()) == Some("session") {
                if let Some(cwd) = v.get("cwd").and_then(|c| c.as_str()) {
                    return Some(cwd.to_string());
                }
            }
        } else {
            // Claude: cwd is a top-level field
            if let Some(cwd) = v.get("cwd").and_then(|c| c.as_str()) {
                return Some(cwd.to_string());
            }
        }
    }
    None
}

/// Returns true only if `git status --porcelain` succeeds with empty output
/// (i.e. no staged, unstaged, or untracked changes). Returns false if not
/// in a git repo or if the tree is dirty.
fn git_working_tree_clean() -> bool {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| o.status.success() && o.stdout.is_empty())
        .unwrap_or(false)
}

fn maybe_checkout_branch(branch: &str) {
    use std::io::IsTerminal;
    if branch.is_empty() || !std::io::stderr().is_terminal() {
        return;
    }
    let current = current_git_branch().unwrap_or_default();
    if current == branch {
        return;
    }
    if !git_working_tree_clean() {
        eprintln!(
            "warning: working tree is dirty, staying on '{current}' \
             (stash or commit before switching to '{branch}')"
        );
        return;
    }
    eprint!("Switch from '{current}' to '{branch}'? [y/N] ");
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).ok();
    if answer.trim().eq_ignore_ascii_case("y") {
        let status = std::process::Command::new("git")
            .args(["checkout", branch])
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(_) => eprintln!("warning: git checkout '{branch}' failed, continuing on '{current}'"),
            Err(e) => eprintln!("warning: git checkout failed: {e}"),
        }
    }
}

fn launch_resume(s: &BranchSession, checkout_branch: bool) -> Result<()> {
    use std::os::unix::process::CommandExt;
    if !s.cwd.is_empty() {
        let _ = std::env::set_current_dir(&s.cwd);
    }
    if checkout_branch {
        maybe_checkout_branch(&s.branch);
    }
    let session_id = &s.session_id;
    if s.source == "codex" {
        eprintln!("Resuming codex session {session_id}...");
        let err = std::process::Command::new("codex")
            .args(["resume", session_id])
            .exec();
        Err(err.into())
    } else if s.source == "pi" {
        eprintln!("Resuming pi session {session_id}...");
        let err = std::process::Command::new("pi")
            .args(["--session", session_id])
            .exec();
        Err(err.into())
    } else {
        eprintln!("Resuming claude session {session_id}...");
        let err = std::process::Command::new("claude")
            .args(["--resume", session_id])
            .exec();
        Err(err.into())
    }
}

fn parse_slack_conv_key(conv_key: &str) -> Option<(String, String, bool)> {
    let (rest, is_thread) = if let Some(rest) = conv_key.strip_prefix("thr:") {
        (rest, true)
    } else if let Some(rest) = conv_key.strip_prefix("sess:") {
        (rest, false)
    } else {
        return None;
    };
    let (channel_id, ts) = rest.rsplit_once('-')?;
    if channel_id.is_empty() || ts.is_empty() {
        None
    } else {
        Some((channel_id.to_string(), ts.to_string(), is_thread))
    }
}

fn slack_open_target_from_metadata(meta: &serde_json::Value) -> Option<SlackOpenTarget> {
    if meta["source"].as_str()? != "slack" {
        return None;
    }

    let conv_key = meta["conv_key"].as_str().unwrap_or("").to_string();
    let parsed = parse_slack_conv_key(&conv_key);
    let channel_id = meta["channel_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| parsed.as_ref().map(|(channel_id, _, _)| channel_id.clone()))?;
    let root_ts = meta["root_ts"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| parsed.as_ref().map(|(_, ts, _)| ts.clone()))
        .unwrap_or_default();

    Some(SlackOpenTarget {
        team_id: meta["team_id"].as_str().unwrap_or("").to_string(),
        channel_id,
        root_ts,
        conv_key,
    })
}

fn slack_deep_link(team_id: &str, channel_id: &str, ts: &str) -> Option<String> {
    if team_id.is_empty() || channel_id.is_empty() || ts.is_empty() {
        None
    } else {
        Some(format!(
            "slack://channel?team={team_id}&id={channel_id}&message={ts}"
        ))
    }
}

fn slack_channel_redirect(team_id: &str, channel_id: &str) -> Option<String> {
    if channel_id.is_empty() {
        None
    } else if team_id.is_empty() {
        Some(format!("https://slack.com/app_redirect?channel={channel_id}"))
    } else {
        Some(format!(
            "https://slack.com/app_redirect?channel={channel_id}&team={team_id}"
        ))
    }
}

fn slack_thread_redirect(team_id: &str, channel_id: &str, thread_ts: &str) -> Option<String> {
    if channel_id.is_empty() || thread_ts.is_empty() {
        return None;
    }
    let base = if team_id.is_empty() {
        "https://app.slack.com/client/".to_string()
    } else {
        format!("https://app.slack.com/client/{team_id}/")
    };
    Some(format!("{base}{channel_id}/thread/{channel_id}-{thread_ts}"))
}

fn slack_open_url(target: &SlackOpenTarget) -> Option<String> {
    let is_thread = parse_slack_conv_key(&target.conv_key)
        .map(|(_, _, is_thread)| is_thread)
        .unwrap_or(false);
    if is_thread {
        slack_thread_redirect(&target.team_id, &target.channel_id, &target.root_ts)
    } else {
        slack_deep_link(&target.team_id, &target.channel_id, &target.root_ts)
            .or_else(|| slack_channel_redirect(&target.team_id, &target.channel_id))
    }
}

fn open_url(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut cmd = std::process::Command::new("open");
        cmd.arg(url);
        cmd
    };

    #[cfg(target_os = "windows")]
    let mut command = {
        let mut cmd = std::process::Command::new("explorer");
        cmd.arg(url);
        cmd
    };

    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut cmd = std::process::Command::new("xdg-open");
        cmd.arg(url);
        cmd
    };

    #[cfg(not(any(unix, target_os = "windows")))]
    compile_error!("unsupported platform for open_url");

    let status = command
        .status()
        .with_context(|| format!("failed to launch opener for {url}"))?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("opener exited with {status} for {url}");
    }
}

fn open_slack_target(target: &SlackOpenTarget) -> Result<String> {
    let url = slack_open_url(target).context("Slack result has no openable channel URL")?;
    match open_url(&url) {
        Ok(()) => Ok(url),
        Err(_err) if url.starts_with("slack://") => {
            let fallback = slack_channel_redirect(&target.team_id, &target.channel_id)
                .context("Slack result has no browser fallback URL")?;
            open_url(&fallback).with_context(|| {
                format!("failed to open native Slack URL {url}; browser fallback also failed")
            })?;
            Ok(fallback)
        }
        Err(err) => Err(err),
    }
}

/// Extract a display tag from a Slack conv_key.
/// Thread keys (`thr:CHAN-TS`) → `thr:TS` (the thread timestamp, usable with --session).
/// Session keys (`sess:CHAN-TS`) → empty (channel + turn is sufficient).
fn slack_conv_tag(conv_key: &str) -> String {
    if let Some(rest) = conv_key.strip_prefix("thr:") {
        if let Some((_chan, ts)) = rest.split_once('-') {
            return format!("thr:{ts}");
        }
    }
    String::new()
}

fn source_label(source: &str) -> &str {
    match source {
        "slack" => "slack",
        "codex" => "codex",
        "pi" => "pi",
        _ => "claude",
    }
}

fn role_label(source: &str) -> &str {
    match source {
        "codex" => "[Codex]",
        "pi" => "[Pi]",
        _ => "[Claude]",
    }
}

fn strip_body_prefix(s: &str) -> &str {
    s.strip_prefix("[User] ")
        .or_else(|| s.strip_prefix("[Claude] "))
        .or_else(|| s.strip_prefix("[Codex] "))
        .or_else(|| s.strip_prefix("[Pi] "))
        .unwrap_or(s)
}

fn indexed_turn_text(r: &SearchResult, turn_idx: usize) -> Option<String> {
    r.bodies
        .get(turn_idx + 1)
        .map(|body| strip_body_prefix(body).trim_end().to_string())
        .filter(|text| !text.trim().is_empty())
}

fn indexed_turn(r: &SearchResult, turn_idx: usize, tm: &TurnMeta) -> Option<SessionTurn> {
    Some(SessionTurn {
        role: tm.role.clone(),
        text: indexed_turn_text(r, turn_idx)?,
        timestamp: tm.timestamp.clone(),
    })
}

fn read_or_indexed_turn(r: &SearchResult, turn_idx: usize, tm: &TurnMeta) -> Option<SessionTurn> {
    read_turn_at(&r.path, &r.source, tm)
        .filter(|turn| !turn.text.trim().is_empty())
        .or_else(|| indexed_turn(r, turn_idx, tm))
}

fn can_resume_result(r: &SearchResult) -> bool {
    !r.session_id.is_empty() && matches!(r.source.as_str(), "claude" | "codex" | "pi")
}

fn first_line(text: &str) -> String {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let end = floor_char_boundary(s, max.saturating_sub(3));
        format!("{}...", &s[..end])
    }
}

fn floor_char_boundary(s: &str, index: usize) -> usize {
    let mut index = index.min(s.len());
    while !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

// --- Plain text fallback (piped output) ---

fn search_plain(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
    types: &[String],
    num_results: usize,
    dm_only: bool,
    no_dm: bool,
    unread_only: bool,
) -> Result<()> {
    let (results, search_ms) = run_search(db_name, assets, q, session, branch, exclude, since_ms, types, num_results, dm_only, no_dm, unread_only)?;

    let mut buf = Vec::new();
    let you = slack::cached_self_user()
        .map(|(_id, name)| format!("  (you: {name})"))
        .unwrap_or_default();
    writeln!(buf, "\n[[ {q} ]]{you}")?;
    writeln!(buf, "search completed in {search_ms} ms\n")?;
    for r in &results {
        writeln!(buf, "---")?;
        // match_idx 0 = header, turns[0] = first turn → turns[match_idx - 1]
        let matched_tm = if r.match_idx > 0 {
            r.turns.get(r.match_idx - 1)
        } else {
            r.turns.first()
        };
        let ts = matched_tm
            .filter(|tm| !tm.timestamp.is_empty())
            .map(|tm| format_date(&tm.timestamp))
            .unwrap_or_else(|| r.timestamp.clone());
        let source_label = source_label(&r.source);
        let filename = if r.path.ends_with(".md") {
            format!("  {}", r.path)
        } else {
            String::new()
        };
        let session_info = if r.source == "slack" {
            let tag = slack_conv_tag(&r.conv_key);
            format!("  {source_label} {tag}")
        } else if !r.session_id.is_empty() {
            let name_info = if !r.session_name.is_empty() {
                format!(" \"{}\"", r.session_name)
            } else {
                String::new()
            };
            format!("  {source_label} {}{name_info} turn {}", r.session_id, r.turn)
        } else {
            String::new()
        };
        let branch_info = if !r.branch.is_empty() {
            format!("  [{}]", r.branch)
        } else {
            String::new()
        };
        writeln!(buf, "{ts}  {}{filename}{session_info}{branch_info}", r.project)?;
        if r.source == "slack" || r.turns.is_empty() || r.path.is_empty() {
            // Slack and .md files: use indexed bodies directly
            let idx = r.match_idx;
            let start = idx.saturating_sub(1);
            let end = (idx + 3).min(r.bodies.len());
            for i in start..end {
                let prefix = if i == idx { ">>>" } else { "   " };
                for line in r.bodies[i].lines().filter(|l| !l.is_empty()) {
                    writeln!(buf, "{prefix} {line}")?;
                }
            }
        } else if !r.session_id.is_empty() {
            // Claude/Codex: read turns via byte offsets
            let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
            let ctx_start = mi.saturating_sub(1);
            let ctx_end = (mi + 2).min(r.turns.len());
            for i in ctx_start..ctx_end {
                let tm = &r.turns[i];
                let label = if tm.role == "user" {
                    "[User]"
                } else {
                    role_label(&r.source)
                };
                let prefix = if i == mi { ">>>" } else { "  " };
                writeln!(buf, "{prefix} {label} {}", format_date(&tm.timestamp))?;
                if let Some(turn) = read_or_indexed_turn(r, i, tm) {
                    for line in turn.text.lines().take(10) {
                        writeln!(buf, "{prefix}   {line}")?;
                    }
                }
            }
        }
    }
    if results.is_empty() {
        writeln!(buf, "no results")?;
    }
    std::io::stdout().write_all(&buf)?;
    Ok(())
}

/// Build styled spans for session metadata display. Returns `Span<'static>`
/// because all values are owned (format!/to_string), not borrowed from args.
fn session_meta_spans(
    date: &str,
    project: &str,
    session_id: &str,
    session_name: &str,
    source: &str,
    branch: &str,
    conv_key: &str,
) -> Vec<ratatui::text::Span<'static>> {
    use ratatui::style::{Color, Style};
    use ratatui::text::Span;

    let mut spans = vec![
        Span::styled(format!("{date} "), Style::default().fg(Color::Green)),
        Span::styled(project.to_string(), Style::default().fg(Color::Cyan)),
    ];
    if source == "slack" {
        let tag = slack_conv_tag(conv_key);
        spans.push(Span::styled(
            format!("  slack {tag}"),
            Style::default().fg(Color::Magenta),
        ));
    } else if !session_id.is_empty() {
        let source_label = source_label(source);
        let short_sid = if session_id.len() > 8 { &session_id[..8] } else { session_id };
        spans.push(Span::styled(
            format!("  {source_label} {short_sid}"),
            Style::default().fg(Color::Magenta),
        ));
    }
    if !session_name.is_empty() {
        spans.push(Span::styled(
            format!("  \"{session_name}\""),
            Style::default().fg(Color::White),
        ));
    }
    if !branch.is_empty() {
        spans.push(Span::styled(
            format!("  {branch}"),
            Style::default().fg(Color::Yellow),
        ));
    }
    spans
}

fn current_git_branch() -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if branch.is_empty() || branch == "HEAD" {
        None
    } else {
        Some(branch)
    }
}

#[derive(Clone)]
struct BranchSession {
    session_id: String,
    source: String,
    branch: String,
    cwd: String,
}


fn find_recent_sessions(db_name: &PathBuf, branch: Option<&str>) -> Result<Vec<SearchResult>> {
    let db = match DB::new_reader(db_name.clone()) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "warning: lookup-only mode could not open {}: {e}",
                db_name.display()
            );
            return Ok(Vec::new());
        }
    };
    let sql = if branch.is_some() {
        "SELECT metadata, body, MAX(date) as date
         FROM document
         WHERE json_extract(metadata, '$.session_id') IS NOT NULL
           AND json_extract(metadata, '$.session_id') != ''
           AND json_extract(metadata, '$.branch') = ?1
         GROUP BY json_extract(metadata, '$.session_id')
         ORDER BY date DESC
         LIMIT 200"
    } else {
        "SELECT metadata, body, MAX(date) as date
         FROM document
         WHERE json_extract(metadata, '$.session_id') IS NOT NULL
           AND json_extract(metadata, '$.session_id') != ''
         GROUP BY json_extract(metadata, '$.session_id')
         ORDER BY date DESC
         LIMIT 200"
    };
    let mut stmt = db.query(sql)?;
    let rows: Vec<(String, String, String)> = if let Some(br) = branch {
        stmt.query_map((br,), |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .filter_map(|r| r.ok()).collect()
    } else {
        stmt.query_map((), |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .filter_map(|r| r.ok()).collect()
    };

    let mut results = Vec::new();
    for (metadata, body, date) in &rows {
        let meta: serde_json::Value = serde_json::from_str(metadata).unwrap_or_default();
        let turns_arr: Vec<TurnMeta> = meta["turns"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|v| TurnMeta {
                        role: v["role"].as_str().unwrap_or("").to_string(),
                        timestamp: v["timestamp"].as_str().unwrap_or("").to_string(),
                        byte_offset: v["off"].as_u64().unwrap_or(0),
                        byte_len: v["len"].as_u64().unwrap_or(0),
                    })
                    .collect()
            })
            .unwrap_or_default();
        results.push(SearchResult {
            timestamp: format_date(date),
            project: meta["project"].as_str().unwrap_or("").to_string(),
            session_id: meta["session_id"].as_str().unwrap_or("").to_string(),
            session_name: meta["session_name"].as_str().unwrap_or("").to_string(),
            turn: meta["turn"].as_u64().unwrap_or(0),
            path: meta["path"].as_str().unwrap_or("").to_string(),
            cwd: meta["cwd"].as_str().unwrap_or("").to_string(),
            source: meta["source"].as_str().unwrap_or("claude").to_string(),
            branch: meta["branch"].as_str().unwrap_or("").to_string(),
            conv_key: meta["conv_key"].as_str().unwrap_or("").to_string(),
            slack_open: slack_open_target_from_metadata(&meta),
            bodies: vec![body.clone()],
            match_idx: 0,
            turns: turns_arr,
        });
    }
    Ok(results)
}

fn format_date(iso: &str) -> String {
    let month = match iso.get(5..7) {
        Some("01") => "Jan",
        Some("02") => "Feb",
        Some("03") => "Mar",
        Some("04") => "Apr",
        Some("05") => "May",
        Some("06") => "Jun",
        Some("07") => "Jul",
        Some("08") => "Aug",
        Some("09") => "Sep",
        Some("10") => "Oct",
        Some("11") => "Nov",
        Some("12") => "Dec",
        _ => "???",
    };
    let day = iso.get(8..10).unwrap_or("??");
    let time = iso.get(11..16).unwrap_or("??:??");
    format!("{month} {day} {time}")
}

fn parse_range(s: &str) -> (usize, usize) {
    if let Some((a, b)) = s.split_once('-') {
        let start = a.parse().unwrap_or(0);
        let end = b.parse().unwrap_or(usize::MAX);
        (start, end)
    } else {
        let n = s.parse().unwrap_or(0);
        (n, n)
    }
}

fn dump(db_name: &PathBuf, session_id: &str, turns_range: Option<&str>, since_ms: Option<i64>) -> Result<()> {
    use witchcraft::types::*;
    use witchcraft::sql_generator::build_filter_sql_and_params;

    let db = match DB::new_reader(db_name.clone()) {
        Ok(db) => db,
        Err(e) => {
            eprintln!(
                "warning: lookup-only mode could not open {}: {e}",
                db_name.display()
            );
            return Ok(());
        }
    };
    let name = session_id.strip_prefix('#').unwrap_or(session_id);

    // Try as session_id first, then thread conv_key, then channel_id/channel_name
    let (rows, is_channel) = {
        let (turn_start, turn_end) = turns_range.map(parse_range).unwrap_or((0, usize::MAX));
        let thread_ts = name.strip_prefix("thr:").unwrap_or(name);
        let thread_like = format!("%-{thread_ts}");

        // Helper: run a filter query and return matched rows
        fn query_filter(
            db: &DB,
            filter: &SqlStatementInternal,
            order: &str,
            turn_start: usize,
            turn_end: usize,
        ) -> Result<Vec<(String, String, i64)>> {
            let (filter_sql, params) = build_filter_sql_and_params(Some(filter))?;
            let sql = format!(
                "SELECT date, body, json_extract(metadata, '$.turn') as turn
                 FROM document WHERE {filter_sql} ORDER BY {order}"
            );
            let mut stmt = db.query(&sql)?;
            let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| &**p as &dyn rusqlite::ToSql).collect();
            let rows: Vec<(String, String, i64)> = stmt
                .query_map(param_refs.as_slice(), |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .filter_map(|r| r.ok())
                .filter(|(_, _, t)| (*t as usize) >= turn_start && (*t as usize) <= turn_end)
                .collect();
            Ok(rows)
        }

        // 1. Try as Claude/Codex session_id
        let session_match = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.session_id".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String(session_id.to_string())),
            }),
            logic: None,
            statements: None,
        };
        let rows = query_filter(&db, &session_match, "turn", turn_start, turn_end)?;
        if !rows.is_empty() {
            (rows, false)
        } else {
            // 2. Try as Slack thread TS (conv_key LIKE %-<ts>)
            let thread_match = SqlStatementInternal {
                statement_type: SqlStatementType::Condition,
                condition: Some(SqlConditionInternal {
                    key: "$.conv_key".to_string(),
                    operator: SqlOperator::Like,
                    value: Some(SqlValue::String(thread_like)),
                }),
                logic: None,
                statements: None,
            };
            let rows = query_filter(&db, &thread_match, "date", turn_start, turn_end)?;
            if !rows.is_empty() {
                (rows, true)
            } else {
            // 3. Try as channel_id/channel_name
            let channel_match = SqlStatementInternal {
                statement_type: SqlStatementType::Group,
                condition: None,
                logic: Some(SqlLogic::Or),
                statements: Some(vec![
                    SqlStatementInternal {
                        statement_type: SqlStatementType::Condition,
                        condition: Some(SqlConditionInternal {
                            key: "$.channel_id".to_string(),
                            operator: SqlOperator::Equals,
                            value: Some(SqlValue::String(name.to_string())),
                        }),
                        logic: None,
                        statements: None,
                    },
                    SqlStatementInternal {
                        statement_type: SqlStatementType::Condition,
                        condition: Some(SqlConditionInternal {
                            key: "$.channel_name".to_string(),
                            operator: SqlOperator::Equals,
                            value: Some(SqlValue::String(name.to_string())),
                        }),
                        logic: None,
                        statements: None,
                    },
                ]),
            };
            let since_filter = since_ms.map(|ms| {
                let cutoff_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64
                    - ms / 1000;
                let cutoff_iso = chrono::DateTime::from_timestamp(cutoff_secs, 0)
                    .unwrap()
                    .to_rfc3339();
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "date".to_string(),
                        operator: SqlOperator::GreaterThanOrEquals,
                        value: Some(SqlValue::String(cutoff_iso)),
                    }),
                    logic: None,
                    statements: None,
                }
            });
            let filters: Vec<SqlStatementInternal> =
                [Some(channel_match), since_filter].into_iter().flatten().collect();
            let combined = if filters.len() == 1 {
                filters.into_iter().next().unwrap()
            } else {
                SqlStatementInternal {
                    statement_type: SqlStatementType::Group,
                    condition: None,
                    logic: Some(SqlLogic::And),
                    statements: Some(filters),
                }
            };
            let rows = query_filter(&db, &combined, "turn DESC", turn_start, turn_end)?;
            (rows, true)
        }}
    };

    if rows.is_empty() {
        eprintln!("No session or channel found for {session_id}");
        std::process::exit(1);
    }

    let mut buf = Vec::new();
    for (date, body, turn) in &rows {
        writeln!(buf, "---")?;
        if is_channel {
            writeln!(buf, "conv {}  {}", *turn, format_date(date))?;
        } else {
            writeln!(buf, "turn {}  {}", *turn, format_date(date))?;
        }
        for line in body.lines().skip_while(|l| {
            !is_channel
                && l.starts_with('[')
                && !l.starts_with("[User]")
                && !l.starts_with("[Claude]")
                && !l.starts_with("[Codex]")
                && !l.starts_with("[Pi]")
        }) {
            writeln!(buf, "{line}")?;
        }
    }
    if !buf.is_empty() {
        writeln!(buf, "---")?;
    }

    use std::io::IsTerminal;
    let output = String::from_utf8(buf)?;
    if std::io::stdout().is_terminal() {
        use std::process::{Command, Stdio};
        let mut pager = Command::new("less")
            .args(["-RFX"])
            .stdin(Stdio::piped())
            .spawn()?;
        pager.stdin.take().unwrap().write_all(output.as_bytes())?;
        let _ = pager.wait();
    } else {
        print!("{output}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Warn));

    let args: Vec<String> = env::args().skip(1).collect();
    let mut session_filter: Option<String> = None;
    let mut branch_filter: Option<String> = None;
    let mut exclude_sessions: Vec<String> = Vec::new();
    let mut since_ms: Option<i64> = None;
    let mut type_filter: Vec<String> = Vec::new();
    let mut num_results: Option<usize> = None;
    let mut dump_session: Option<String> = None;
    let mut turns_range: Option<String> = None;
    let mut dm_only = false;
    let mut no_dm = false;
    let mut unread_only = false;
    let mut current = false;
    let mut exclude_current = false;
    let mut quiet = false;
    let mut query_args: Vec<&str> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                eprintln!("Usage:");
                eprintln!("  pickbrain [options] [query]");
                eprintln!("  pickbrain --dump <session-id|channel|thr:ts> [--turns N-M] [--since T]");
                eprintln!("  pickbrain --nuke");
                eprintln!();
                eprintln!("With no arguments, opens an interactive session browser.");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --branch NAME|.      filter by git branch (. = current)");
                eprintln!("  --current            search within the calling session");
                eprintln!("  --exclude UUID,...   exclude sessions from results");
                eprintln!("  --exclude-current    exclude the calling session");
                eprintln!("  --session ID         search within a session, channel, or thread");
                eprintln!("  --since 24h|7d|2w    only search recent history");
                eprintln!("  --type claude,codex,pi,slack  filter by source");
                eprintln!("  -q, --quiet          suppress ingest progress output");
                eprintln!("  -n N                 number of results (0=unlimited, default: unlimited in TUI, 20 in pipe)");
                eprintln!("  --dm                 only DMs (Slack)");
                eprintln!("  --no-dm              exclude DMs (Slack)");
                eprintln!("  --unread             only unread conversations (Slack)");
                eprintln!("  --dump ID            dump a session, channel, or thread");
                eprintln!("  --turns N-M          limit turns/conversations in dump");
                eprintln!();
                eprintln!("Environment:");
                eprintln!("  PICKBRAIN_DIR        override the pickbrain DB and state directory");
                std::process::exit(0);
            }
            "--quiet" | "-q" => {
                quiet = true;
            }
            "--nuke" => {
                let db_name = db_path();
                if db_name.exists() {
                    std::fs::remove_file(&db_name)?;
                    eprintln!("removed {}", db_name.display());
                } else {
                    eprintln!("no database to remove");
                }
                watermark::remove(&watermark::claude_path());
                watermark::remove(&watermark::codex_path());
                watermark::remove(&watermark::pi_path());
                slack::remove_watermark();
                std::process::exit(0);
            }
            "--session" | "--channel" => {
                session_filter = iter.next().cloned();
            }
            "--branch" => {
                branch_filter = iter.next().cloned();
            }
            "--exclude" => {
                if let Some(val) = iter.next() {
                    for id in val.split(',') {
                        let id = id.trim();
                        if !id.is_empty() {
                            exclude_sessions.push(id.to_string());
                        }
                    }
                }
            }
            "--since" => {
                if let Some(val) = iter.next() {
                    match parse_since(val) {
                        Some(ms) => since_ms = Some(ms),
                        None => {
                            eprintln!("invalid --since value: {val} (use e.g. 24h, 7d, 2w)");
                            std::process::exit(1);
                        }
                    }
                }
            }
            "--type" => {
                if let Some(val) = iter.next() {
                    for t in val.split(',') {
                        let t = t.trim();
                        if !t.is_empty() {
                            type_filter.push(t.to_string());
                        }
                    }
                }
            }
            "-n" => {
                if let Some(val) = iter.next() {
                    num_results = Some(val.parse().unwrap_or(0));
                }
            }
            "--dump" => {
                dump_session = iter.next().cloned();
            }
            "--turns" => {
                turns_range = iter.next().cloned();
            }
            "--current" => {
                current = true;
            }
            "--exclude-current" => {
                exclude_current = true;
            }
            "--dm" => {
                dm_only = true;
            }
            "--no-dm" => {
                no_dm = true;
            }
            "--unread" => {
                unread_only = true;
            }
            s if s.starts_with('-') => {
                eprintln!("unknown option: {s}");
                std::process::exit(1);
            }
            _ => {
                query_args.push(arg);
            }
        }
    }

    // Resolve `--branch .` to the current git branch
    if branch_filter.as_deref() == Some(".") {
        branch_filter = current_git_branch();
        if branch_filter.is_none() {
            eprintln!("error: --branch . used but not in a git repo");
            std::process::exit(1);
        }
    }

    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        eprintln!("pickbrain {} — Copyright (c) 2026 Dropbox Inc.", env!("CARGO_PKG_VERSION"));
    }

    let db_name = db_path();
    let assets = assets_path();

    // Migrate DB from old location (~/.claude/pickbrain.db)
    if !pickbrain_dir_overridden() && !db_name.exists() {
        let home = env::var("HOME").unwrap_or_default();
        let old_db = PathBuf::from(home).join(".claude/pickbrain.db");
        if old_db.exists() {
            eprintln!("migrating database from {} to {}", old_db.display(), db_name.display());
            std::fs::rename(&old_db, &db_name).ok();
        }
    }

    // Detect the calling session once — used for both ingest-skip and --current filter.
    let active_session = detect_active_session();

    if current || exclude_current {
        match &active_session {
            Some(id) => {
                if current {
                    session_filter = Some(id.clone());
                }
                if exclude_current {
                    exclude_sessions.push(id.clone());
                }
            }
            None => {
                let flag = if current { "--current" } else { "--exclude-current" };
                eprintln!("{flag}: could not detect active session");
                std::process::exit(1);
            }
        }
    }

    // Skip the active session's JSONL if its watermark is fresh (<10 min).
    // If we can't detect the active session, nothing is skipped (eager by default).
    let stale_ms = 10 * 60 * 1000;
    if needs_ingest(&db_name, active_session.as_deref(), stale_ms, &type_filter)? {
        match IngestLock::try_acquire(&db_name) {
            Ok(Some(_lock)) => {
                match ingest(&db_name, active_session.as_deref(), stale_ms, &type_filter, quiet) {
                    Ok(have_changes) => {
                        if have_changes {
                            let db_rw = DB::new(db_name.clone()).unwrap();
                            let device = witchcraft::make_device();
                            let embedder = witchcraft::Embedder::new(&device, &assets)?;
                            embed_and_index(&db_rw, &embedder, &device)?;
                        }
                    },
                    Err(e) => {
                        eprintln!("warning: ingest failed: {e}");
                        std::process::exit(1);
                    }
                }
            },
            Ok(None) => {
                warn_lookup_only("another pickbrain process is ingesting");
            }
            Err(e) => {
                warn_lookup_only(format!("the ingest lock could not be acquired: {e}"));
            }
        }
    }

    let has_branch = branch_filter.is_some();
    let has_filters = session_filter.is_some() || !exclude_sessions.is_empty() || since_ms.is_some()
        || !type_filter.is_empty() || dm_only || no_dm || unread_only || current || exclude_current
        || has_branch;

    if let Some(ref sid) = dump_session {
        dump(&db_name, sid, turns_range.as_deref(), since_ms)?;
    } else if !query_args.is_empty() || has_filters {
        let q = query_args.join(" ");
        if std::io::stdout().is_terminal() {
            let n = num_results.unwrap_or(0);
            if let Some(s) = search_tui(&db_name, &assets, Some(&q), session_filter.as_deref(), branch_filter.as_deref(), &exclude_sessions, since_ms, &type_filter, n, dm_only, no_dm, unread_only)? {
                launch_resume(&s, has_branch)?;
            }
        } else {
            let n = num_results.unwrap_or(20);
            search_plain(&db_name, &assets, &q, session_filter.as_deref(), branch_filter.as_deref(), &exclude_sessions, since_ms, &type_filter, n, dm_only, no_dm, unread_only)?;
        }
    } else {
        if let Some(s) = search_tui(&db_name, &assets, None, session_filter.as_deref(), branch_filter.as_deref(), &exclude_sessions, since_ms, &type_filter, 0, dm_only, no_dm, unread_only)? {
            launch_resume(&s, has_branch)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_date() {
        assert_eq!(format_date("2025-01-15T10:30:00Z"), "Jan 15 10:30");
        assert_eq!(format_date("bad"), "??? ?? ??:??");
    }

    #[test]
    fn test_session_meta_spans_preserves_formatted_date() {
        let spans = session_meta_spans("May 12 10:09", "DM", "", "", "slack", "", "");
        assert_eq!(spans[0].content.as_ref(), "May 12 10:09 ");
    }

    #[test]
    fn test_indexed_turn_text_strips_role_prefix() {
        let result = SearchResult {
            timestamp: String::new(),
            project: String::new(),
            session_id: String::new(),
            session_name: String::new(),
            turn: 0,
            path: String::new(),
            cwd: String::new(),
            source: "claude".to_string(),
            branch: String::new(),
            conv_key: String::new(),
            slack_open: None,
            bodies: vec![
                "[project] title\n".to_string(),
                "[User] run oxidize\nmore text\n".to_string(),
                "[Claude] done\n".to_string(),
            ],
            match_idx: 1,
            turns: Vec::new(),
        };
        assert_eq!(
            indexed_turn_text(&result, 0).as_deref(),
            Some("run oxidize\nmore text")
        );
        let tm = TurnMeta {
            role: "user".to_string(),
            timestamp: "2026-04-15T14:12:36.044Z".to_string(),
            byte_offset: 0,
            byte_len: 0,
        };
        let turn = read_or_indexed_turn(&result, 0, &tm).unwrap();
        assert_eq!(turn.role, "user");
        assert_eq!(turn.timestamp, "2026-04-15T14:12:36.044Z");
        assert_eq!(turn.text, "run oxidize\nmore text");
    }

    #[test]
    fn test_parse_range() {
        assert_eq!(parse_range("3-7"), (3, 7));
        assert_eq!(parse_range("5"), (5, 5));
    }

    #[test]
    fn test_build_sql_filter_none() {
        assert!(build_sql_filter(None, None, &[], None, &[], false, false, false).is_none());
    }

    #[test]
    fn test_build_sql_filter_branch_only() {
        use witchcraft::types::*;
        let f = build_sql_filter(None, Some("main"), &[], None, &[], false, false, false).unwrap();
        assert_eq!(f.statement_type, SqlStatementType::Condition);
        let cond = f.condition.unwrap();
        assert_eq!(cond.key, "$.branch");
    }

    #[test]
    fn test_build_sql_filter_both() {
        use witchcraft::types::*;
        let f = build_sql_filter(Some("abc"), Some("main"), &[], None, &[], false, false, false).unwrap();
        assert_eq!(f.statement_type, SqlStatementType::Group);
        assert_eq!(f.logic, Some(SqlLogic::And));
        assert_eq!(f.statements.unwrap().len(), 2);
    }

    #[test]
    fn test_slack_channel_open_url_uses_native_deep_link() {
        let meta = serde_json::json!({
            "source": "slack",
            "team_id": "T1",
            "channel_id": "C1",
            "conv_key": "sess:C1-1000.000100",
            "root_ts": "1000.000100"
        });
        let target = slack_open_target_from_metadata(&meta).unwrap();
        assert_eq!(
            slack_open_url(&target).as_deref(),
            Some("slack://channel?team=T1&id=C1&message=1000.000100")
        );
    }

    #[test]
    fn test_slack_thread_open_url_uses_https_thread_redirect() {
        let meta = serde_json::json!({
            "source": "slack",
            "team_id": "T1",
            "channel_id": "C1",
            "conv_key": "thr:C1-1000.000100",
            "root_ts": "1000.000100"
        });
        let target = slack_open_target_from_metadata(&meta).unwrap();
        assert_eq!(
            slack_open_url(&target).as_deref(),
            Some("https://app.slack.com/client/T1/C1/thread/C1-1000.000100")
        );
    }

    #[test]
    fn test_slack_open_target_falls_back_to_conv_key_parts() {
        let meta = serde_json::json!({
            "source": "slack",
            "team_id": "T1",
            "conv_key": "thr:C1-1000.000100"
        });
        let target = slack_open_target_from_metadata(&meta).unwrap();
        assert_eq!(target.channel_id, "C1");
        assert_eq!(target.root_ts, "1000.000100");
    }

    #[test]
    fn test_floor_char_boundary_matches_std_cases() {
        fn check_many(s: &str, arg: impl IntoIterator<Item = usize>, ret: usize) {
            for idx in arg {
                assert_eq!(floor_char_boundary(s, idx), ret);
            }
        }

        check_many("", [0, 1, isize::MAX as usize, usize::MAX], 0);
        check_many("x", [0], 0);
        check_many("x", [1, isize::MAX as usize, usize::MAX], 1);
        check_many("jp", [0], 0);
        check_many("jp", [1], 1);
        check_many("jp", 2..4, 2);
        check_many("ĵƥ", 0..2, 0);
        check_many("ĵƥ", 2..4, 2);
        check_many("ĵƥ", 4..6, 4);
        check_many("日本", 0..3, 0);
        check_many("日本", 3..6, 3);
        check_many("日本", 6..8, 6);
        check_many("🇯🇵", 0..4, 0);
        check_many("🇯🇵", 4..8, 4);
        check_many("🇯🇵", 8..10, 8);
    }
}
