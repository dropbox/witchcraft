use anyhow::Result;
use log::{Level, LevelFilter, Metadata, Record};
use std::env;
use std::io::Write;
use std::path::PathBuf;

mod claude_code;
mod codex;
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

fn db_path() -> PathBuf {
    let home = env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".claude/pickbrain.db")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

fn ingest(db_name: &PathBuf) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();
    let (sessions, memories, authored) = claude_code::ingest_claude_code(&mut db)?;
    let codex_sessions = codex::ingest_codex(&mut db)?;
    let total = sessions + memories + authored + codex_sessions;
    if total == 0 {
        return Ok(false);
    }
    eprintln!(
        "ingested {sessions} claude sessions, {codex_sessions} codex sessions, {memories} memory files, {authored} authored files"
    );
    Ok(true)
}

fn embed_and_index(db: &DB, embedder: &Embedder, device: &candle_core::Device) -> Result<()> {
    let embedded = witchcraft::embed_chunks(db, embedder, None)?;
    if embedded > 0 {
        eprintln!("embedded {embedded} chunks");
        witchcraft::index_chunks(db, device)?;
        eprintln!("index rebuilt");
    }
    Ok(())
}

// --- Search result data ---

struct SearchResult {
    timestamp: String,
    project: String,
    session_id: String,
    turn: u64,
    path: String,
    cwd: String,
    source: String,
    bodies: Vec<String>,
    match_idx: usize,
}

// A turn from the original JSONL session file
struct SessionTurn {
    role: String,
    text: String,
    timestamp: String,
}

fn load_session_turns(jsonl_path: &str, source: &str) -> Vec<SessionTurn> {
    if source == "codex" {
        return load_codex_session_turns(jsonl_path);
    }
    load_claude_session_turns(jsonl_path)
}

fn load_claude_session_turns(jsonl_path: &str) -> Vec<SessionTurn> {
    let raw = match std::fs::read_to_string(jsonl_path) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    #[derive(serde::Deserialize)]
    struct Entry {
        #[serde(rename = "type")]
        entry_type: String,
        timestamp: Option<String>,
        message: Option<Msg>,
        #[serde(default, rename = "isMeta")]
        is_meta: bool,
    }
    #[derive(serde::Deserialize)]
    struct Msg {
        role: Option<String>,
        content: Option<MsgContent>,
    }
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum MsgContent {
        Text(String),
        Blocks(Vec<Block>),
    }
    #[derive(serde::Deserialize)]
    struct Block {
        #[serde(rename = "type")]
        block_type: String,
        #[serde(default)]
        text: Option<String>,
    }

    let mut turns = Vec::new();
    for line in raw.lines() {
        let entry: Entry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.entry_type != "user" && entry.entry_type != "assistant" {
            continue;
        }
        if entry.is_meta {
            continue;
        }
        let msg = match entry.message {
            Some(m) => m,
            None => continue,
        };
        let role = match msg.role {
            Some(r) if r == "user" || r == "assistant" => r,
            _ => continue,
        };
        let text = match msg.content {
            Some(MsgContent::Text(t)) => t,
            Some(MsgContent::Blocks(blocks)) => {
                blocks
                    .iter()
                    .filter(|b| b.block_type == "text")
                    .filter_map(|b| b.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            None => continue,
        };
        let trimmed = text.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("<command-")
            || trimmed.starts_with("<local-command-")
        {
            continue;
        }
        let clean = regex::Regex::new(r"<[^>]+>")
            .map(|re| re.replace_all(&text, "").to_string())
            .unwrap_or(text);
        let clean = clean.trim().to_string();
        if clean.is_empty() {
            continue;
        }
        turns.push(SessionTurn {
            role,
            text: clean,
            timestamp: entry.timestamp.unwrap_or_default(),
        });
    }
    turns
}

fn load_codex_session_turns(jsonl_path: &str) -> Vec<SessionTurn> {
    let raw = match std::fs::read_to_string(jsonl_path) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let re_xml = regex::Regex::new(r"<[^>]+>").unwrap();
    let mut turns = Vec::new();

    for line in raw.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let timestamp = v.get("timestamp").and_then(|t| t.as_str()).unwrap_or_default().to_string();
        let entry_type = v.get("type").and_then(|t| t.as_str()).unwrap_or_default();
        let payload = match v.get("payload") {
            Some(p) => p,
            None => continue,
        };

        if entry_type == "response_item" {
            let ptype = payload.get("type").and_then(|t| t.as_str()).unwrap_or_default();
            if ptype == "message" && payload.get("role").and_then(|r| r.as_str()) == Some("user") {
                let content = match payload.get("content").and_then(|c| c.as_array()) {
                    Some(c) => c,
                    None => continue,
                };
                let text: String = content
                    .iter()
                    .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_text"))
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                let clean = re_xml.replace_all(&text, "").trim().to_string();
                if clean.is_empty() {
                    continue;
                }
                turns.push(SessionTurn {
                    role: "user".to_string(),
                    text: clean,
                    timestamp,
                });
            }
        } else if entry_type == "event_msg" {
            let ptype = payload.get("type").and_then(|t| t.as_str()).unwrap_or_default();
            if ptype == "agent_reasoning" {
                if let Some(text) = payload.get("text").and_then(|t| t.as_str()) {
                    let clean = text.trim().to_string();
                    if !clean.is_empty() {
                        turns.push(SessionTurn {
                            role: "assistant".to_string(),
                            text: clean,
                            timestamp,
                        });
                    }
                }
            }
        }
    }
    turns
}

fn run_search(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
) -> Result<(Vec<SearchResult>, u128)> {
    use witchcraft::types::*;
    let device = witchcraft::make_device();
    let embedder = witchcraft::Embedder::new(&device, assets)?;

    {
        let db_rw = DB::new(db_name.clone()).unwrap();
        embed_and_index(&db_rw, &embedder, &device)?;
    }

    let mut cache = witchcraft::EmbeddingsCache::new(1);
    let db = DB::new_reader(db_name.clone()).unwrap();
    let sql_filter = session.map(|id| SqlStatementInternal {
        statement_type: SqlStatementType::Condition,
        condition: Some(SqlConditionInternal {
            key: "$.session_id".to_string(),
            operator: SqlOperator::Equals,
            value: Some(SqlValue::String(id.to_string())),
        }),
        logic: None,
        statements: None,
    });
    let now = std::time::Instant::now();
    let results = witchcraft::search(
        &db,
        &embedder,
        &mut cache,
        q,
        0.5,
        10,
        true,
        sql_filter.as_ref(),
    )?;
    let search_ms = now.elapsed().as_millis();

    let out: Vec<SearchResult> = results
        .into_iter()
        .map(|(_score, metadata, bodies, sub_idx, date)| {
            let meta: serde_json::Value = serde_json::from_str(&metadata).unwrap_or_default();
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            SearchResult {
                timestamp: format_date(&date),
                project: meta["project"].as_str().unwrap_or("").to_string(),
                session_id: meta["session_id"].as_str().unwrap_or("").to_string(),
                turn: meta["turn"].as_u64().unwrap_or(0),
                path: meta["path"].as_str().unwrap_or("").to_string(),
                cwd: meta["cwd"].as_str().unwrap_or("").to_string(),
                source: meta["source"].as_str().unwrap_or("claude").to_string(),
                bodies,
                match_idx: idx,
            }
        })
        .collect();
    Ok((out, search_ms))
}

// --- TUI ---

enum View {
    List,
    Detail(usize),
}

fn search_tui(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
) -> Result<Option<(String, String, String)>> {
    let (results, search_ms) = run_search(db_name, assets, q, session)?;
    if results.is_empty() {
        eprintln!("no results");
        return Ok(None);
    }

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
    let mut scroll_offset: u16 = 0;
    let mut resume_session: Option<(String, String, String)> = None;
    let mut confirm_resume: Option<(String, String, String, String)> = None;
    // Cache loaded session turns so we don't re-read the JSONL on every frame
    let mut detail_cache: Option<(usize, Vec<SessionTurn>)> = None;

    loop {
        terminal.draw(|f| {
            let area = f.area();
            let show_footer = confirm_resume.is_some() && matches!(view, View::Detail(_));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(if show_footer {
                    vec![Constraint::Length(2), Constraint::Min(0), Constraint::Length(1)]
                } else {
                    vec![Constraint::Length(2), Constraint::Min(0), Constraint::Length(0)]
                })
                .split(area);

            // Header
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("[[ {q} ]]"),
                    Style::default()
                        .fg(Color::Rgb(0, 255, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {search_ms} ms  "),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    match view {
                        View::List => "↑↓ navigate  ⏎ open  q quit",
                        View::Detail(idx) if !results[idx].session_id.is_empty() => {
                            "↑↓ scroll  r resume session  esc back  q quit"
                        }
                        View::Detail(_) => "↑↓ scroll  esc back  q quit",
                    },
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
            f.render_widget(header, chunks[0]);

            // Footer: resume confirmation
            if show_footer {
                let cwd = confirm_resume.as_ref()
                    .map(|(_, _, c, _)| c.as_str())
                    .unwrap_or("?");
                let sid = confirm_resume.as_ref()
                    .map(|(s, _, _, _)| s.as_str())
                    .unwrap_or("?");
                let src = confirm_resume.as_ref()
                    .map(|(_, _, _, s)| s.as_str())
                    .unwrap_or("claude");
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
                f.render_widget(footer, chunks[2]);
            }

            match view {
                View::List => {
                    let width = chunks[1].width as usize;
                    let items: Vec<ratatui::widgets::ListItem> = results
                        .iter()
                        .map(|r| {
                            let preview_idx = r
                                .bodies
                                .iter()
                                .position(|b| b.starts_with("[User]"))
                                .or_else(|| {
                                    if r.match_idx == 0 && r.bodies.len() > 1 {
                                        Some(1)
                                    } else {
                                        Some(r.match_idx)
                                    }
                                })
                                .unwrap_or(0);
                            let raw_preview = first_line(&r.bodies[preview_idx]);
                            let preview = raw_preview
                                .strip_prefix("[User] ")
                                .or_else(|| raw_preview.strip_prefix("[Claude] "))
                                .or_else(|| raw_preview.strip_prefix("[Codex] "))
                                .unwrap_or(&raw_preview);
                            let mut meta_spans = vec![
                                Span::styled(
                                    format!("{} ", r.timestamp),
                                    Style::default().fg(Color::Green),
                                ),
                                Span::styled(&r.project, Style::default().fg(Color::Cyan)),
                            ];
                            if r.path.ends_with(".md") {
                                meta_spans.push(Span::styled(
                                    format!("  {}", r.path),
                                    Style::default().fg(Color::Yellow),
                                ));
                            }
                            if !r.session_id.is_empty() {
                                let source_label = if r.source == "codex" {
                                    "codex"
                                } else {
                                    "claude"
                                };
                                let short_sid = if r.session_id.len() > 8 {
                                    &r.session_id[..8]
                                } else {
                                    &r.session_id
                                };
                                meta_spans.push(Span::styled(
                                    format!("  {source_label} {short_sid}"),
                                    Style::default().fg(Color::Magenta),
                                ));
                                meta_spans.push(Span::styled(
                                    format!("  turn {}", r.turn),
                                    Style::default().fg(Color::DarkGray),
                                ));
                            }
                            ratatui::widgets::ListItem::new(vec![
                                Line::from(meta_spans),
                                Line::styled(
                                    format!("  {}", truncate(&preview, width.saturating_sub(4))),
                                    Style::default(),
                                ),
                                Line::from(""),
                            ])
                        })
                        .collect();

                    let list = ratatui::widgets::List::new(items).highlight_style(
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    );
                    f.render_stateful_widget(list, chunks[1], &mut list_state);
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
                        lines.push(Line::from(vec![
                            Span::styled(&r.session_id, Style::default().fg(Color::Magenta)),
                            Span::styled(
                                format!("  turn {}", r.turn),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    }
                    lines.push(Line::from(""));

                    // If we have a JSONL path and a session, show the real conversation
                    let turns = detail_cache
                        .as_ref()
                        .filter(|(ci, _)| *ci == idx)
                        .map(|(_, t)| t.as_slice());

                    if let Some(turns) = turns {
                        let (matched_start, matched_end) =
                            interaction_turn_range(turns, r.turn as usize);
                        for (i, turn) in turns.iter().enumerate() {
                            let role_style = if turn.role == "user" {
                                Style::default()
                                    .fg(Color::Rgb(0, 255, 0))
                                    .add_modifier(Modifier::BOLD)
                            } else {
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD)
                            };
                            let is_matched_turn = i >= matched_start && i < matched_end;
                            lines.push(Line::from(vec![
                                Span::styled(
                                    if turn.role == "user" {
                                        "[User] "
                                    } else if r.source == "codex" {
                                        "[Codex] "
                                    } else {
                                        "[Claude] "
                                    },
                                    role_style,
                                ),
                                Span::styled(
                                    format_date(&turn.timestamp),
                                    Style::default().fg(Color::DarkGray),
                                ),
                            ]));
                            let text_style = if is_matched_turn {
                                Style::default()
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
                        .scroll((scroll_offset, 0));
                    f.render_widget(detail, chunks[1]);
                }
            }
        })?;

        if let Event::Key(key) = event::read()? {
            match (&view, key.code, key.modifiers) {
                (_, KeyCode::Char('q') | KeyCode::Esc, _) if confirm_resume.is_some() => {
                    confirm_resume = None;
                }
                (View::Detail(_), KeyCode::Char('q') | KeyCode::Esc, _) => {
                    view = View::List;
                }
                (View::List, KeyCode::Char('q') | KeyCode::Esc, _) => break,
                (_, KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                (_, KeyCode::Char('z'), KeyModifiers::CONTROL) => {
                    disable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        LeaveAlternateScreen,
                        crossterm::cursor::Show
                    )?;
                    unsafe { libc::raise(libc::SIGTSTP); }
                    // When resumed (fg), re-enter TUI
                    enable_raw_mode()?;
                    crossterm::execute!(
                        std::io::stdout(),
                        EnterAlternateScreen,
                        crossterm::cursor::Hide
                    )?;
                    // Force full redraw
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
                    let r = &results[selected];
                    if !r.session_id.is_empty() && !r.path.is_empty() {
                        let turns = load_session_turns(&r.path, &r.source);
                        // Scroll to the matched interaction
                        let (matched_start, _) =
                            interaction_turn_range(&turns, r.turn as usize);
                        let mut line_count: u16 = 3; // session header lines
                        for (i, turn) in turns.iter().enumerate() {
                            if i >= matched_start {
                                break;
                            }
                            line_count += 1; // role header
                            line_count += turn.text.lines().count() as u16;
                            line_count += 1; // blank line
                        }
                        scroll_offset = line_count.saturating_sub(2);
                        detail_cache = Some((selected, turns));
                    } else {
                        scroll_offset = 0;
                        detail_cache = None;
                    }
                    view = View::Detail(selected);
                }

                // Detail view
                (View::Detail(_), KeyCode::Down | KeyCode::Char('j'), _) => {
                    scroll_offset = scroll_offset.saturating_add(1);
                }
                (View::Detail(_), KeyCode::Up | KeyCode::Char('k'), _) => {
                    scroll_offset = scroll_offset.saturating_sub(1);
                }
                (View::Detail(_), KeyCode::PageDown | KeyCode::Char(' '), _) => {
                    scroll_offset = scroll_offset.saturating_add(20);
                }
                (View::Detail(_), KeyCode::PageUp, _) => {
                    scroll_offset = scroll_offset.saturating_sub(20);
                }
                (View::Detail(_), KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                    let half = terminal.size().map(|s| s.height / 2).unwrap_or(10);
                    scroll_offset = scroll_offset.saturating_add(half);
                }
                (View::Detail(_), KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                    let half = terminal.size().map(|s| s.height / 2).unwrap_or(10);
                    scroll_offset = scroll_offset.saturating_sub(half);
                }
                (View::Detail(idx), KeyCode::Char('r'), _) => {
                    let r = &results[*idx];
                    if !r.session_id.is_empty() {
                        let cwd = if !r.cwd.is_empty() {
                            r.cwd.clone()
                        } else {
                            read_cwd_from_jsonl(&r.path, &r.source)
                                .unwrap_or_else(|| "?".to_string())
                        };
                        confirm_resume = Some((r.session_id.clone(), r.path.clone(), cwd, r.source.clone()));
                    }
                }
                (View::Detail(_), KeyCode::Char('y') | KeyCode::Enter, _) => {
                    if let Some((ref sid, ref path, _, ref source)) = confirm_resume {
                        resume_session = Some((sid.clone(), path.clone(), source.clone()));
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
        } else {
            // Claude: cwd is a top-level field
            if let Some(cwd) = v.get("cwd").and_then(|c| c.as_str()) {
                return Some(cwd.to_string());
            }
        }
    }
    None
}

fn launch_resume(session_id: &str, jsonl_path: &str, source: &str) -> Result<()> {
    use std::os::unix::process::CommandExt;
    if let Some(cwd) = read_cwd_from_jsonl(jsonl_path, source) {
        let _ = std::env::set_current_dir(&cwd);
    }
    if source == "codex" {
        eprintln!("Resuming codex session {session_id}...");
        let err = std::process::Command::new("codex")
            .args(["resume", session_id])
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

fn interaction_turn_range(turns: &[SessionTurn], interaction_idx: usize) -> (usize, usize) {
    let user_starts: Vec<usize> = turns
        .iter()
        .enumerate()
        .filter(|(_, t)| t.role == "user")
        .map(|(i, _)| i)
        .collect();
    let start = user_starts.get(interaction_idx).copied().unwrap_or(0);
    let end = user_starts
        .get(interaction_idx + 1)
        .copied()
        .unwrap_or(turns.len());
    (start, end)
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
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

// --- Plain text fallback (piped output) ---

fn search_plain(
    db_name: &PathBuf,
    assets: &PathBuf,
    q: &str,
    session: Option<&str>,
) -> Result<()> {
    let (results, search_ms) = run_search(db_name, assets, q, session)?;

    let mut buf = Vec::new();
    writeln!(buf, "\n[[ {q} ]]")?;
    writeln!(buf, "search completed in {search_ms} ms\n")?;
    for r in &results {
        writeln!(buf, "---")?;
        let filename = if r.path.ends_with(".md") {
            format!("  {}", r.path)
        } else {
            String::new()
        };
        writeln!(buf, "{}  {}{filename}", r.timestamp, r.project)?;
        if !r.session_id.is_empty() {
            writeln!(buf, "  {} turn {}", r.session_id, r.turn)?;
        }
        let idx = r.match_idx;
        if idx > 0 {
            for line in r.bodies[idx - 1].lines().filter(|l| !l.is_empty()) {
                writeln!(buf, "  {line}")?;
            }
        }
        for line in r.bodies[idx].lines().filter(|l| !l.is_empty()) {
            writeln!(buf, "  {line}")?;
        }
        if idx + 1 < r.bodies.len() {
            for line in r.bodies[idx + 1].lines().filter(|l| !l.is_empty()) {
                writeln!(buf, "  {line}")?;
            }
        }
    }
    if results.is_empty() {
        writeln!(buf, "no results")?;
    }
    std::io::stdout().write_all(&buf)?;
    Ok(())
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

fn dump(db_name: &PathBuf, session_id: &str, turns_range: Option<&str>) -> Result<()> {
    let db = DB::new_reader(db_name.clone()).unwrap();

    let (turn_start, turn_end) = turns_range.map(parse_range).unwrap_or((0, usize::MAX));

    let mut stmt = db.query(
        "SELECT date, body, json_extract(metadata, '$.turn') as turn
         FROM document
         WHERE json_extract(metadata, '$.session_id') = ?1
         ORDER BY turn",
    )?;
    let rows: Vec<(String, String, i64)> = stmt
        .query_map((session_id,), |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    if rows.is_empty() {
        eprintln!("No session found for {session_id}");
        std::process::exit(1);
    }

    let mut buf = Vec::new();
    for (date, body, turn) in &rows {
        let t = *turn as usize;
        if t < turn_start || t > turn_end {
            continue;
        }
        writeln!(buf, "---")?;
        writeln!(buf, "turn {t}  {}", format_date(date))?;
        for line in body.lines().skip_while(|l| {
            l.starts_with('[') && !l.starts_with("[User]") && !l.starts_with("[Claude]")
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
    let do_update = args.iter().any(|a| a == "--update");
    let mut session_filter: Option<String> = None;
    let mut dump_session: Option<String> = None;
    let mut turns_range: Option<String> = None;
    let mut query_args: Vec<&str> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--update" => {}
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
                std::process::exit(0);
            }
            "--session" => {
                session_filter = iter.next().cloned();
            }
            "--dump" => {
                dump_session = iter.next().cloned();
            }
            "--turns" => {
                turns_range = iter.next().cloned();
            }
            _ => {
                query_args.push(arg);
            }
        }
    }

    let db_name = db_path();
    let assets = assets_path();

    // Always ingest first — cheap filesystem walk with mtime watermarks
    let has_query = !query_args.is_empty();
    let has_dump = dump_session.is_some();
    if has_query || has_dump || do_update {
        match ingest(&db_name) {
            Ok(false) => {
                if do_update && !has_query && !has_dump {
                    println!("up to date");
                }
            }
            Ok(true) => {
                if do_update && !has_query {
                    let device = witchcraft::make_device();
                    let embedder = Embedder::new(&device, &assets)?;
                    let db_rw = DB::new(db_name.clone()).unwrap();
                    embed_and_index(&db_rw, &embedder, &device)?;
                }
            }
            Err(e) => {
                if !db_name.exists() {
                    eprintln!("No database found. Run: pickbrain --update");
                    std::process::exit(1);
                }
                eprintln!("warning: ingest failed: {e}");
            }
        }
    }

    if let Some(ref sid) = dump_session {
        dump(&db_name, sid, turns_range.as_deref())?;
    } else if has_query {
        let q = query_args.join(" ");
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            if let Some((sid, path, source)) = search_tui(&db_name, &assets, &q, session_filter.as_deref())? {
                launch_resume(&sid, &path, &source)?;
            }
        } else {
            search_plain(&db_name, &assets, &q, session_filter.as_deref())?;
        }
    } else if !do_update {
        eprintln!("Usage: pickbrain [--update] [--session UUID] <query>");
        eprintln!("       pickbrain --dump <UUID> [--turns N-M]");
        eprintln!("       pickbrain --nuke");
    }
    Ok(())
}
