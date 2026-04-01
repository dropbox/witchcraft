use anyhow::Result;
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::io::Write;
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
    PathBuf::from(home).join(".claude/pickbrain.db")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

fn update(db_name: &PathBuf, assets: &PathBuf) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();
    let (sessions, memories, authored) = warp::claude_code::ingest_claude_code(&mut db)?;
    if sessions + memories + authored == 0 {
        return Ok(false);
    }
    println!("ingested {sessions} sessions, {memories} memory files, {authored} authored files");

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

// ANSI color helpers — disabled when stdout is not a terminal
struct Colors {
    bold: &'static str,
    dim: &'static str,
    reset: &'static str,
    cyan: &'static str,
    green: &'static str,
    yellow: &'static str,
    magenta: &'static str,
}

fn colors() -> Colors {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        Colors {
            bold: "\x1b[1m",
            dim: "\x1b[2m",
            reset: "\x1b[0m",
            cyan: "\x1b[36m",
            green: "\x1b[32m",
            yellow: "\x1b[33m",
            magenta: "\x1b[35m",
        }
    } else {
        Colors {
            bold: "",
            dim: "",
            reset: "",
            cyan: "",
            green: "",
            yellow: "",
            magenta: "",
        }
    }
}

fn search(db_name: &PathBuf, assets: &PathBuf, q: &str) -> Result<()> {
    let c = colors();
    let device = warp::make_device();
    let embedder = warp::Embedder::new(&device, assets)?;
    let mut cache = warp::EmbeddingsCache::new(1);
    let db = DB::new_reader(db_name.clone()).unwrap();
    let results = warp::search(&db, &embedder, &mut cache, q, 0.5, 10, true, None)?;

    // Render output to a buffer, then page if needed
    let (_, cols) = terminal_size();
    let separator: String = "─".repeat(cols);
    let mut buf = Vec::new();
    for (_score, metadata, bodies, sub_idx, date) in &results {
        writeln!(buf, "{}{separator}{}", c.dim, c.reset)?;
        let meta: serde_json::Value = serde_json::from_str(metadata).unwrap_or_default();
        let project = meta["project"].as_str().unwrap_or("");
        let session_id = meta["session_id"].as_str().unwrap_or("");
        let turn = meta["turn"].as_u64().unwrap_or(0);
        let path = meta["path"].as_str().unwrap_or("");
        let idx = (*sub_idx as usize).min(bodies.len().saturating_sub(1));

        let timestamp = format_date(date);
        let filename = if path.ends_with(".md") {
            format!("  {}{path}{}", c.yellow, c.reset)
        } else {
            String::new()
        };
        writeln!(buf, "{}{timestamp}{}  {}{project}{}{filename}", c.green, c.reset, c.cyan, c.reset)?;
        if !session_id.is_empty() {
            writeln!(buf, "  {}{session_id}{} {}turn {turn}{}", c.magenta, c.reset, c.dim, c.reset)?;
        }
        if idx > 0 {
            write_chunk(&mut buf, &bodies[idx - 1], "")?;
        }
        write_chunk(&mut buf, &bodies[idx], c.bold)?;
        if idx + 1 < bodies.len() {
            write_chunk(&mut buf, &bodies[idx + 1], "")?;
        }
    }
    if results.is_empty() {
        writeln!(buf, "no results")?;
    }

    use std::io::IsTerminal;
    let output = String::from_utf8(buf)?;
    if std::io::stdout().is_terminal() {
        let (term_lines, _) = terminal_size();
        let output_lines = output.lines().count();
        if output_lines + 2 > term_lines {
            use std::process::{Command, Stdio};
            let mut pager = Command::new("less")
                .args(["-RFX"])
                .stdin(Stdio::piped())
                .spawn()?;
            pager.stdin.take().unwrap().write_all(output.as_bytes())?;
            let _ = pager.wait();
            return Ok(());
        }
    }
    print!("{output}");
    Ok(())
}

fn format_date(iso: &str) -> String {
    // "2026-03-31T09:24:20.675Z" -> "Mar 31 09:24"
    let month = match iso.get(5..7) {
        Some("01") => "Jan", Some("02") => "Feb", Some("03") => "Mar",
        Some("04") => "Apr", Some("05") => "May", Some("06") => "Jun",
        Some("07") => "Jul", Some("08") => "Aug", Some("09") => "Sep",
        Some("10") => "Oct", Some("11") => "Nov", Some("12") => "Dec",
        _ => "???",
    };
    let day = iso.get(8..10).unwrap_or("??");
    let time = iso.get(11..16).unwrap_or("??:??");
    format!("{month} {day} {time}")
}

fn write_chunk(buf: &mut Vec<u8>, text: &str, style: &str) -> std::io::Result<()> {
    let reset = if style.is_empty() { "" } else { "\x1b[0m" };
    for line in text.lines().filter(|l| !l.is_empty()) {
        writeln!(buf, "  {style}{line}{reset}")?;
    }
    Ok(())
}

fn terminal_size() -> (usize, usize) {
    #[repr(C)]
    struct Winsize { ws_row: u16, ws_col: u16, _xpixel: u16, _ypixel: u16 }
    extern "C" { fn ioctl(fd: i32, request: u64, ...) -> i32; }
    const TIOCGWINSZ: u64 = 0x40087468;
    unsafe {
        let mut ws = std::mem::zeroed::<Winsize>();
        if ioctl(1, TIOCGWINSZ, &mut ws) == 0 && ws.ws_row > 0 && ws.ws_col > 0 {
            (ws.ws_row as usize, ws.ws_col as usize)
        } else {
            (24, 80)
        }
    }
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Warn));

    let args: Vec<String> = env::args().skip(1).collect();
    let do_update = args.first().is_some_and(|a| a == "--update");
    let query_args: Vec<&str> = args.iter()
        .filter(|a| *a != "--update")
        .map(|s| s.as_str())
        .collect();

    let db_name = db_path();
    let assets = assets_path();

    if do_update {
        match update(&db_name, &assets)? {
            true => {}
            false => if query_args.is_empty() { println!("up to date") },
        }
    }

    if !query_args.is_empty() {
        if !do_update {
            if !db_name.exists() {
                eprintln!("No database found. Run: pickbrain --update");
                std::process::exit(1);
            }
            if let Ok(meta) = std::fs::metadata(&db_name) {
                if let Ok(modified) = meta.modified() {
                    let age = std::time::SystemTime::now().duration_since(modified).unwrap_or_default();
                    if age.as_secs() > 86400 {
                        let hours = age.as_secs() / 3600;
                        eprintln!("Database is {hours}h old. Consider: pickbrain --update <query>");
                    }
                }
            }
        }
        let q = query_args.join(" ");
        search(&db_name, &assets, &q)?;
    } else if !do_update {
        eprintln!("Usage: pickbrain [--update] <query>");
    }
    Ok(())
}
