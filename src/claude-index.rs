use anyhow::Result;
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::path::PathBuf;

use warp::DB;

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

fn db_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".claude/claude-search.sqlite")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

fn update(db_name: &PathBuf, assets: &PathBuf) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();
    let (sessions, memories) = warp::claude_code::ingest_claude_code(&mut db)?;
    if sessions + memories == 0 {
        return Ok(false);
    }
    println!("ingested {sessions} sessions, {memories} memory files");

    let device = warp::make_device();
    let embedder = warp::Embedder::new(&device, assets)?;
    let embedded = warp::embed_chunks(&db, &embedder, None)?;
    if embedded > 0 {
        println!("embedded {embedded} chunks");
        warp::full_index(&db, &device)?;
        println!("index rebuilt");
    }
    Ok(true)
}

// ANSI color helpers
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";

fn search(db_name: &PathBuf, assets: &PathBuf, q: &str) -> Result<()> {
    let device = warp::make_device();
    let embedder = warp::Embedder::new(&device, assets)?;
    let mut cache = warp::EmbeddingsCache::new(1);
    let db = DB::new_reader(db_name.clone()).unwrap();
    let results = warp::search(&db, &embedder, &mut cache, q, 0.5, 10, true, None)?;
    for (score, metadata, body, _body_idx) in &results {
        let meta: serde_json::Value = serde_json::from_str(metadata).unwrap_or_default();
        let title = meta["title"].as_str().unwrap_or("");
        let project = meta["project"].as_str().unwrap_or("");
        let source = meta["source"].as_str().unwrap_or("");
        let session_id = meta["session_id"].as_str().unwrap_or("");
        let turn = meta["turn"].as_u64().unwrap_or(0);
        let path = meta["path"].as_str().unwrap_or("");

        println!("{BOLD}{GREEN}{score:.3}{RESET}  {BOLD}{title}{RESET}");
        println!("  {CYAN}{project}{RESET}  {DIM}{source}{RESET}");
        if !session_id.is_empty() {
            println!("  {MAGENTA}{session_id}{RESET} {DIM}turn {turn}{RESET}");
        }
        if !path.is_empty() {
            println!("  {DIM}{path}{RESET}");
        }
        let preview: String = body.chars().take(300).collect();
        println!("  {YELLOW}{preview}{RESET}");
        println!();
    }
    if results.is_empty() {
        println!("no results");
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Warn));

    let args: Vec<String> = env::args().collect();
    let db_name = db_path();
    let assets = assets_path();

    if args.len() == 2 && args[1] == "update" {
        match update(&db_name, &assets)? {
            true => {}
            false => println!("up to date"),
        }
    } else if args.len() >= 3 && args[1] == "search" {
        update(&db_name, &assets)?;
        let q = args[2..].join(" ");
        search(&db_name, &assets, &q)?;
    } else {
        eprintln!("Usage: {} update | search <text>", args[0]);
    }
    Ok(())
}
