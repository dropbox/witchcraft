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
        println!("[{score:.3}] ({source}) {project} - {title}");
        let preview: String = body.chars().take(200).collect();
        println!("  {preview}");
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
