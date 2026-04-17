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
    let dir = PathBuf::from(home).join(".pickbrain");
    std::fs::create_dir_all(&dir).ok();
    dir.join("pickbrain.db")
}

fn assets_path() -> PathBuf {
    PathBuf::from(env::var("WARP_ASSETS").unwrap_or_else(|_| "assets".into()))
}

fn ingest(db_name: &PathBuf) -> Result<bool> {
    let mut db = DB::new(db_name.clone()).unwrap();
    let (sessions, memories, authored, configs) = claude_code::ingest_claude_code(&mut db)?;
    let codex_sessions = codex::ingest_codex(&mut db)?;
    let total = sessions + memories + authored + configs + codex_sessions;
    if total == 0 {
        eprintln!("No new sessions to ingest.");
        return Ok(false);
    }
    eprintln!(
        "ingested {sessions} claude sessions, {codex_sessions} codex sessions, {memories} memory files, {authored} authored files, {configs} config files"
    );
    Ok(true)
}

fn embed_and_index(db: &DB, embedder: &Embedder, device: &candle_core::Device) -> Result<()> {
    let embedded = witchcraft::embed_chunks(db, embedder, None)?;
    if embedded > 0 {
        witchcraft::index_chunks(db, device)?;
    }
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
    bodies: Vec<String>,
    match_idx: usize,
    turns: Vec<TurnMeta>,
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

fn read_turn_at(path: &str, source: &str, tm: &TurnMeta) -> Option<SessionTurn> {
    let line = read_jsonl_line(path, tm.byte_offset, tm.byte_len)?;
    let v: serde_json::Value = serde_json::from_str(&line).ok()?;

    let text = if source == "codex" {
        let payload = v.get("payload")?;
        let ptype = payload.get("type")?.as_str()?;
        if ptype == "message" && payload.get("role")?.as_str()? == "user" {
            let content = payload.get("content")?.as_array()?;
            let texts: Vec<&str> = content
                .iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("input_text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect();
            texts.join("\n")
        } else if ptype == "agent_reasoning" {
            payload.get("text")?.as_str()?.to_string()
        } else {
            return None;
        }
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
) -> Option<witchcraft::types::SqlStatementInternal> {
    use witchcraft::types::*;
    let mut conditions: Vec<SqlStatementInternal> = Vec::new();
    if let Some(id) = session {
        conditions.push(SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.session_id".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String(id.to_string())),
            }),
            logic: None,
            statements: None,
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
) -> Result<(Vec<SearchResult>, u128)> {
    let device = witchcraft::make_device();
    let embedder = witchcraft::Embedder::new(&device, assets)?;
    run_search_with(&DB::new_reader(db_name.clone()).unwrap(), &embedder, q, session, branch, exclude, since_ms)
}

fn run_search_with(
    db: &DB,
    embedder: &Embedder,
    q: &str,
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
) -> Result<(Vec<SearchResult>, u128)> {
    let mut cache = witchcraft::EmbeddingsCache::new(1);
    let sql_filter = build_sql_filter(session, branch, exclude, since_ms);
    let now = std::time::Instant::now();
    let results = witchcraft::search(
        db,
        embedder,
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
    q: &str,
    session: Option<&str>,
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
) -> Result<Option<(String, String, String, String, String)>> {
    let device = witchcraft::make_device();
    let embedder = witchcraft::Embedder::new(&device, assets)?;
    let db = DB::new_reader(db_name.clone()).unwrap();
    let (mut results, mut search_ms) = run_search_with(&db, &embedder, q, session, branch, exclude, since_ms)?;
    if results.is_empty() {
        eprintln!("no results");
        return Ok(None);
    }
    let mut active_query = q.to_string();

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
    let mut resume_session: Option<(String, String, String, String, String)> = None;
    let mut confirm_resume: Option<(String, String, String, String, String)> = None;
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
                "type query  ⏎ search  esc cancel"
            } else {
                match view {
                    View::List => "↑↓/jk navigate  ⏎ open  / search  q quit",
                    View::Detail(idx) if !results[idx].session_id.is_empty() => {
                        "↑↓/jk scroll  r resume  / search  esc back  q quit"
                    }
                    View::Detail(_) => "↑↓/jk scroll  / search  esc back  q quit",
                }
            };
            let header = Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("[[ {} ]]", active_query),
                    Style::default()
                        .fg(Color::Rgb(0, 255, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {} results  {} ms  ", results.len(), search_ms),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(help_text, Style::default().fg(Color::DarkGray)),
            ]));
            f.render_widget(header, chunks[0]);

            // Search input bar
            if show_search {
                let search_bar = Paragraph::new(Line::from(vec![
                    Span::styled("/ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        &search_filter,
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                    if searching {
                        Span::styled("_", Style::default().fg(Color::DarkGray))
                    } else {
                        Span::raw("")
                    },
                ]));
                f.render_widget(search_bar, chunks[1]);
            }

            let content_area = chunks[2];
            let footer_area = chunks[3];

            // Footer: resume confirmation
            if show_footer {
                let cwd = confirm_resume.as_ref()
                    .map(|(_, _, c, _, _)| c.as_str())
                    .unwrap_or("?");
                let sid = confirm_resume.as_ref()
                    .map(|(s, _, _, _, _)| s.as_str())
                    .unwrap_or("?");
                let src = confirm_resume.as_ref()
                    .map(|(_, _, _, s, _)| s.as_str())
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
                                &ts, &r.project, &r.session_id, &r.session_name, &r.source, &r.branch,
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
                            let role_prefix = if match_role == "user" {
                                "[User] "
                            } else if match_role == "assistant" {
                                if r.source == "codex" { "[Codex] " } else { "[Claude] " }
                            } else {
                                ""
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

                    let list = ratatui::widgets::List::new(items).highlight_style(
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    );
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
                        // Restore pre-search state
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
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                    (KeyCode::Backspace, _) => { search_filter.pop(); }
                    (KeyCode::Char(c), _) => { search_filter.push(c); }
                    _ => {}
                }
                // Live search: update results as the user types (>= 3 chars)
                if search_filter.chars().count() >= 3 {
                    if let Ok((new_results, ms)) = run_search_with(
                        &db, &embedder, &search_filter, session, branch, exclude, since_ms,
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
                            for tm in &r.turns {
                                if let Some(turn) = read_turn_at(&r.path, &r.source, tm) {
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
                    if !r.session_id.is_empty() {
                        let cwd = if !r.cwd.is_empty() {
                            r.cwd.clone()
                        } else {
                            read_cwd_from_jsonl(&r.path, &r.source)
                                .unwrap_or_else(|| "?".to_string())
                        };
                        confirm_resume = Some((r.session_id.clone(), r.path.clone(), cwd, r.source.clone(), r.branch.clone()));
                    }
                }
                (View::Detail(_), KeyCode::Char('y') | KeyCode::Enter, _) => {
                    if let Some((ref sid, ref _path, ref cwd, ref source, ref branch)) = confirm_resume {
                        resume_session = Some((sid.clone(), source.clone(), branch.clone(), cwd.clone(), _path.clone()));
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

fn launch_resume(session_id: &str, source: &str, branch: &str, cwd: &str) -> Result<()> {
    use std::os::unix::process::CommandExt;
    if !cwd.is_empty() {
        let _ = std::env::set_current_dir(cwd);
    }
    if !branch.is_empty() {
        let current = current_git_branch().unwrap_or_default();
        if current != branch {
            if !git_working_tree_clean() {
                eprintln!(
                    "warning: working tree is dirty, staying on '{current}' \
                     (stash or commit before switching to '{branch}')"
                );
            } else {
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


fn strip_body_prefix(s: &str) -> &str {
    s.strip_prefix("[User] ")
        .or_else(|| s.strip_prefix("[Claude] "))
        .or_else(|| s.strip_prefix("[Codex] "))
        .unwrap_or(s)
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
    branch: Option<&str>,
    exclude: &[String],
    since_ms: Option<i64>,
) -> Result<()> {
    let (results, search_ms) = run_search(db_name, assets, q, session, branch, exclude, since_ms)?;

    let mut buf = Vec::new();
    writeln!(buf, "\n[[ {q} ]]")?;
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
        let source_label = if r.source == "codex" { "codex" } else { "claude" };
        let filename = if r.path.ends_with(".md") {
            format!("  {}", r.path)
        } else {
            String::new()
        };
        let name_info = if !r.session_name.is_empty() {
            format!(" \"{}\"", r.session_name)
        } else {
            String::new()
        };
        let session_info = if !r.session_id.is_empty() {
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
        if !r.session_id.is_empty() && !r.path.is_empty() && !r.turns.is_empty() {
            // Read only the matched turn + neighbors via byte offsets
            let mi = if r.match_idx > 0 { r.match_idx - 1 } else { 0 };
            let ctx_start = mi.saturating_sub(1);
            let ctx_end = (mi + 2).min(r.turns.len());
            for i in ctx_start..ctx_end {
                let tm = &r.turns[i];
                let label = if tm.role == "user" {
                    "[User]"
                } else if r.source == "codex" {
                    "[Codex]"
                } else {
                    "[Claude]"
                };
                let prefix = if i == mi { ">>>" } else { "  " };
                writeln!(buf, "{prefix} {label} {}", format_date(&tm.timestamp))?;
                if let Some(turn) = read_turn_at(&r.path, &r.source, tm) {
                    for line in turn.text.lines().take(10) {
                        writeln!(buf, "{prefix}   {line}")?;
                    }
                }
            }
        } else {
            // Fallback for .md files etc: use indexed bodies
            let idx = r.match_idx;
            for line in r.bodies[idx].lines().filter(|l| !l.is_empty()) {
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

/// Build styled spans for session metadata display. Returns `Span<'static>`
/// because all values are owned (format!/to_string), not borrowed from args.
fn session_meta_spans(
    date: &str,
    project: &str,
    session_id: &str,
    session_name: &str,
    source: &str,
    branch: &str,
) -> Vec<ratatui::text::Span<'static>> {
    use ratatui::style::{Color, Style};
    use ratatui::text::Span;

    let mut spans = vec![
        Span::styled(format!("{} ", format_date(date)), Style::default().fg(Color::Green)),
        Span::styled(project.to_string(), Style::default().fg(Color::Cyan)),
    ];
    if !session_id.is_empty() {
        let source_label = if source == "codex" { "codex" } else { "claude" };
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
    session_name: String,
    source: String,
    path: String,
    project: String,
    branch: String,
    cwd: String,
    date: String,
    title: String,
}

/// Convert search results into BranchSessions for the resume picker,
/// deduplicating by session_id (keeping the first/best match per session).
fn results_to_sessions(results: &[SearchResult]) -> Vec<BranchSession> {
    let mut seen = std::collections::HashSet::new();
    results
        .iter()
        .filter(|r| !r.session_id.is_empty())
        .filter(|r| seen.insert(r.session_id.clone()))
        .map(|r| {
            let title = r.bodies.get(r.match_idx)
                .map(|b| first_line(b))
                .map(|l| strip_body_prefix(&l).to_string())
                .unwrap_or_default()
                .chars()
                .take(80)
                .collect::<String>();
            BranchSession {
                session_id: r.session_id.clone(),
                session_name: r.session_name.clone(),
                source: r.source.clone(),
                path: r.path.clone(),
                project: r.project.clone(),
                branch: r.branch.clone(),
                cwd: r.cwd.clone(),
                date: r.timestamp.clone(),
                title,
            }
        })
        .collect()
}

fn find_sessions_for_branch(db_name: &PathBuf, branch: &str) -> Result<Vec<BranchSession>> {
    let db = DB::new_reader(db_name.clone()).unwrap();
    let mut stmt = db.query(
        "SELECT metadata, body, date FROM document
         WHERE json_extract(metadata, '$.branch') = ?1
         ORDER BY date DESC",
    )?;
    let rows: Vec<(String, String, String)> = stmt
        .query_map((branch,), |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    // Deduplicate by session_id, keeping the most recent turn per session
    let mut seen = std::collections::HashSet::new();
    let mut sessions = Vec::new();
    for (metadata, body, date) in &rows {
        let meta: serde_json::Value = serde_json::from_str(metadata).unwrap_or_default();
        let sid = meta["session_id"].as_str().unwrap_or("").to_string();
        if sid.is_empty() || !seen.insert(sid.clone()) {
            continue;
        }
        // Extract title from the first [User] line in the body
        let title = body
            .lines()
            .find(|l| l.starts_with("[User]"))
            .map(|l| l.strip_prefix("[User] ").unwrap_or(l))
            .unwrap_or("")
            .chars()
            .take(80)
            .collect::<String>();
        sessions.push(BranchSession {
            session_id: sid,
            session_name: meta["session_name"].as_str().unwrap_or("").to_string(),
            source: meta["source"].as_str().unwrap_or("claude").to_string(),
            path: meta["path"].as_str().unwrap_or("").to_string(),
            project: meta["project"].as_str().unwrap_or("").to_string(),
            branch: meta["branch"].as_str().unwrap_or("").to_string(),
            cwd: meta["cwd"].as_str().unwrap_or("").to_string(),
            date: date.clone(),
            title,
        });
    }
    Ok(sessions)
}

fn pick_and_resume(
    sessions: Vec<BranchSession>,
    context: &str,
    db_name: &PathBuf,
    assets: &PathBuf,
    branch_filter: Option<&str>,
) -> Result<()> {
    if sessions.is_empty() {
        eprintln!("No sessions found {context}");
        std::process::exit(1);
    }
    if sessions.len() == 1 {
        let s = &sessions[0];
        if !std::path::Path::new(&s.path).exists() {
            anyhow::bail!("session file no longer exists: {}", s.path);
        }
        launch_resume(&s.session_id, &s.source, &s.branch, &s.cwd)?;
        return Ok(());
    }

    // Lazy-load embedder+db only when user first presses `/` to search
    let mut search_engine: Option<(Embedder, DB)> = None;

    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::ListState;
    use ratatui::Terminal;

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut active_sessions = sessions.clone();
    let mut query = String::new();
    let mut searching = false;
    let mut selected: usize = 0;
    let mut list_state = ListState::default();
    list_state.select(Some(0));
    let mut chosen: Option<usize> = None;

    loop {
        // Clamp selection
        if selected >= active_sessions.len() {
            selected = active_sessions.len().saturating_sub(1);
        }
        list_state.select(if active_sessions.is_empty() { None } else { Some(selected) });

        terminal.draw(|f| {
            let area = f.area();
            let show_search = searching || !query.is_empty();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),  // header
                    Constraint::Length(if show_search { 1 } else { 0 }),  // search bar
                    Constraint::Min(0),     // list
                ])
                .split(area);

            // Header
            let count_label = if query.is_empty() {
                format!("{} sessions", active_sessions.len())
            } else {
                format!("{} results", active_sessions.len())
            };
            let help_text = if searching {
                "type query  ⏎ search  esc cancel"
            } else {
                "↑↓/jk navigate  ⏎ resume  / filter  q quit"
            };
            let header = Line::from(vec![
                Span::styled(
                    format!("[[ resume {context} ]]  "),
                    Style::default()
                        .fg(Color::Rgb(0, 255, 0))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{count_label}  "),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(help_text, Style::default().fg(Color::DarkGray)),
            ]);
            f.render_widget(ratatui::widgets::Paragraph::new(header), chunks[0]);

            // Search bar
            if show_search {
                let search_line = Line::from(vec![
                    Span::styled("/ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        &query,
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                    if searching {
                        Span::styled("_", Style::default().fg(Color::DarkGray))
                    } else {
                        Span::raw("")
                    },
                ]);
                f.render_widget(ratatui::widgets::Paragraph::new(search_line), chunks[1]);
            }

            let width = chunks[2].width as usize;
            let items: Vec<ratatui::widgets::ListItem> = active_sessions
                .iter()
                .map(|s| {
                    let meta_spans = session_meta_spans(
                        &s.date, &s.project, &s.session_id, &s.session_name, &s.source, &s.branch,
                    );
                    ratatui::widgets::ListItem::new(vec![
                        Line::from(meta_spans),
                        Line::from(vec![
                            Span::raw("  "),
                            Span::raw(truncate(&s.title, width.saturating_sub(4))),
                        ]),
                        Line::from(""),
                    ])
                })
                .collect();

            let list = ratatui::widgets::List::new(items).highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            );
            f.render_stateful_widget(list, chunks[2], &mut list_state);
        })?;

        if let Event::Key(key) = event::read()? {
            // Search mode: live-search as the user types
            if searching {
                match (key.code, key.modifiers) {
                    (KeyCode::Esc, _) => {
                        searching = false;
                        query.clear();
                        active_sessions = sessions.clone();
                        selected = 0;
                        continue;
                    }
                    (KeyCode::Enter, _) => {
                        searching = false;
                        continue;
                    }
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                    (KeyCode::Backspace, _) => { query.pop(); }
                    (KeyCode::Char(c), _) => { query.push(c); }
                    _ => {}
                }
                // Live search: update results as the user types (>= 3 chars)
                if query.chars().count() >= 3 {
                    let engine = search_engine.get_or_insert_with(|| {
                        let device = witchcraft::make_device();
                        let embedder = witchcraft::Embedder::new(&device, assets).unwrap();
                        let db = DB::new_reader(db_name.clone()).unwrap();
                        (embedder, db)
                    });
                    if let Ok((search_results, _)) = run_search_with(
                        &engine.1, &engine.0, &query, None, branch_filter, &[], None,
                    ) {
                        active_sessions = results_to_sessions(&search_results);
                        selected = 0;
                    }
                } else if query.is_empty() {
                    active_sessions = sessions.clone();
                    selected = 0;
                }
                continue;
            }

            match (key.code, key.modifiers) {
                (KeyCode::Char('q'), _) => break,
                (KeyCode::Esc, _) => {
                    if !query.is_empty() {
                        query.clear();
                        active_sessions = sessions.clone();
                        selected = 0;
                    } else {
                        break;
                    }
                }
                (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                (KeyCode::Char('f'), KeyModifiers::CONTROL) |
                (KeyCode::Char('/'), _) => {
                    searching = true;
                }
                (KeyCode::Down | KeyCode::Char('j'), _) => {
                    if !active_sessions.is_empty() && selected + 1 < active_sessions.len() {
                        selected += 1;
                        list_state.select(Some(selected));
                    }
                }
                (KeyCode::Up | KeyCode::Char('k'), _) => {
                    if selected > 0 {
                        selected -= 1;
                        list_state.select(Some(selected));
                    }
                }
                (KeyCode::Enter, _) => {
                    if selected < active_sessions.len() {
                        chosen = Some(selected);
                    }
                    break;
                }
                _ => {}
            }
        }
    }

    disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), LeaveAlternateScreen)?;

    if let Some(idx) = chosen {
        let s = &active_sessions[idx];
        if !std::path::Path::new(&s.path).exists() {
            anyhow::bail!("session file no longer exists: {}", s.path);
        }
        launch_resume(&s.session_id, &s.source, &s.branch, &s.cwd)?;
    }
    Ok(())
}

fn resume_for_branch(db_name: &PathBuf, assets: &PathBuf, branch: &str) -> Result<()> {
    let sessions = find_sessions_for_branch(db_name, branch)?;
    pick_and_resume(sessions, &format!("on branch '{branch}'"), db_name, assets, Some(branch))
}

fn find_sessions_for_project(db_name: &PathBuf, cwd: &str) -> Result<Vec<BranchSession>> {
    let db = DB::new_reader(db_name.clone()).unwrap();
    let mut stmt = db.query(
        "SELECT metadata, body, date FROM document
         WHERE json_extract(metadata, '$.session_id') IS NOT NULL
         ORDER BY date DESC",
    )?;
    let rows: Vec<(String, String, String)> = stmt
        .query_map((), |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    let mut seen = std::collections::HashSet::new();
    let mut sessions = Vec::new();

    // Claude Code stores the full cwd (sans leading /) as "project".
    // Codex stores just the directory basename as "project" but has a
    // separate "cwd" field with the full path. Match against both.
    let cwd_canonical = cwd.trim_start_matches('/');

    for (metadata, body, date) in &rows {
        let meta: serde_json::Value = serde_json::from_str(metadata).unwrap_or_default();
        let sid = meta["session_id"].as_str().unwrap_or("").to_string();
        if sid.is_empty() || !seen.insert(sid.clone()) {
            continue;
        }
        let project = meta["project"].as_str().unwrap_or("");
        let meta_cwd = meta["cwd"].as_str().unwrap_or("").trim_start_matches('/');
        if project != cwd_canonical && meta_cwd != cwd_canonical {
            continue;
        }
        let title = body
            .lines()
            .find(|l| l.starts_with("[User]"))
            .map(|l| l.strip_prefix("[User] ").unwrap_or(l))
            .unwrap_or("")
            .chars()
            .take(80)
            .collect::<String>();
        sessions.push(BranchSession {
            session_id: sid,
            session_name: meta["session_name"].as_str().unwrap_or("").to_string(),
            source: meta["source"].as_str().unwrap_or("claude").to_string(),
            path: meta["path"].as_str().unwrap_or("").to_string(),
            project: meta["project"].as_str().unwrap_or("").to_string(),
            branch: meta["branch"].as_str().unwrap_or("").to_string(),
            cwd: meta["cwd"].as_str().unwrap_or("").to_string(),
            date: date.clone(),
            title,
        });
    }
    Ok(sessions)
}

fn resume_for_project(db_name: &PathBuf, assets: &PathBuf, cwd: &str) -> Result<()> {
    let sessions = find_sessions_for_project(db_name, cwd)?;
    pick_and_resume(sessions, &format!("in '{cwd}'"), db_name, assets, None)
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
    let mut session_filter: Option<String> = None;
    let mut branch_filter: Option<String> = None;
    let mut exclude_sessions: Vec<String> = Vec::new();
    let mut since_ms: Option<i64> = None;
    let mut dump_session: Option<String> = None;
    let mut turns_range: Option<String> = None;
    let mut do_resume = false;
    let mut query_args: Vec<&str> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
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
            "--dump" => {
                dump_session = iter.next().cloned();
            }
            "--turns" => {
                turns_range = iter.next().cloned();
            }
            "--resume" => {
                do_resume = true;
            }
            _ => {
                query_args.push(arg);
            }
        }
    }

    use std::io::IsTerminal;
    if std::io::stderr().is_terminal() {
        eprintln!("pickbrain {} — Copyright (c) 2026 Dropbox Inc.", env!("CARGO_PKG_VERSION"));
    }

    let db_name = db_path();
    let assets = assets_path();

    // Migrate DB from old location (~/.claude/pickbrain.db)
    if !db_name.exists() {
        let home = env::var("HOME").unwrap_or_default();
        let old_db = PathBuf::from(home).join(".claude/pickbrain.db");
        if old_db.exists() {
            eprintln!("migrating database from {} to {}", old_db.display(), db_name.display());
            std::fs::rename(&old_db, &db_name).ok();
        }
    }

    match ingest(&db_name) {
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

    if let Some(ref sid) = dump_session {
        dump(&db_name, sid, turns_range.as_deref())?;
    } else if do_resume {
        // --resume: find sessions by branch (or by project dir if not in a git repo)
        if let Some(ref br) = branch_filter {
            resume_for_branch(&db_name, &assets, br)?;
        } else if let Some(br) = current_git_branch() {
            resume_for_branch(&db_name, &assets, &br)?;
        } else {
            // Not in a git repo — fall back to sessions for the current project directory
            let cwd = std::env::current_dir()?.to_string_lossy().to_string();
            resume_for_project(&db_name, &assets, &cwd)?;
        }
    } else if !query_args.is_empty() {
        let q = query_args.join(" ");
        if std::io::stdout().is_terminal() {
            if let Some((sid, source, branch, cwd, _path)) = search_tui(&db_name, &assets, &q, session_filter.as_deref(), branch_filter.as_deref(), &exclude_sessions, since_ms)? {
                launch_resume(&sid, &source, &branch, &cwd)?;
            }
        } else {
            search_plain(&db_name, &assets, &q, session_filter.as_deref(), branch_filter.as_deref(), &exclude_sessions, since_ms)?;
        }
    } else {
        eprintln!("Usage: pickbrain [--branch NAME] [--session UUID] [--exclude UUID,...] [--since 24h|7d|2w] <query>");
        eprintln!("       pickbrain --resume [--branch NAME]");
        eprintln!("       pickbrain --dump <UUID> [--turns N-M]");
        eprintln!("       pickbrain --nuke");
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
    fn test_parse_range() {
        assert_eq!(parse_range("3-7"), (3, 7));
        assert_eq!(parse_range("5"), (5, 5));
    }

    #[test]
    fn test_build_sql_filter_none() {
        assert!(build_sql_filter(None, None, &[], None).is_none());
    }

    #[test]
    fn test_build_sql_filter_branch_only() {
        use witchcraft::types::*;
        let f = build_sql_filter(None, Some("main"), &[], None).unwrap();
        assert_eq!(f.statement_type, SqlStatementType::Condition);
        let cond = f.condition.unwrap();
        assert_eq!(cond.key, "$.branch");
    }

    #[test]
    fn test_build_sql_filter_both() {
        use witchcraft::types::*;
        let f = build_sql_filter(Some("abc"), Some("main"), &[], None).unwrap();
        assert_eq!(f.statement_type, SqlStatementType::Group);
        assert_eq!(f.logic, Some(SqlLogic::And));
        assert_eq!(f.statements.unwrap().len(), 2);
    }

    #[test]
    fn test_results_to_sessions_deduplicates() {
        let results = vec![
            SearchResult {
                timestamp: "Jan 15 10:30".to_string(),
                project: "server".to_string(),
                session_id: "abc-123".to_string(),
                session_name: "debug".to_string(),
                turn: 0,
                path: "/tmp/test.jsonl".to_string(),
                cwd: "/home/user/src/server".to_string(),
                source: "claude".to_string(),
                branch: "main".to_string(),
                bodies: vec!["header".to_string(), "[User] first message".to_string()],
                match_idx: 1,
                turns: vec![],
            },
            SearchResult {
                timestamp: "Jan 15 10:35".to_string(),
                project: "server".to_string(),
                session_id: "abc-123".to_string(), // same session
                session_name: "debug".to_string(),
                turn: 1,
                path: "/tmp/test.jsonl".to_string(),
                cwd: "/home/user/src/server".to_string(),
                source: "claude".to_string(),
                branch: "main".to_string(),
                bodies: vec!["header".to_string(), "[User] second message".to_string()],
                match_idx: 1,
                turns: vec![],
            },
            SearchResult {
                timestamp: "Jan 15 11:00".to_string(),
                project: "server".to_string(),
                session_id: "def-456".to_string(), // different session
                session_name: "refactor".to_string(),
                turn: 0,
                path: "/tmp/test2.jsonl".to_string(),
                cwd: "/home/user/src/server".to_string(),
                source: "claude".to_string(),
                branch: "main".to_string(),
                bodies: vec!["header".to_string(), "[User] other work".to_string()],
                match_idx: 1,
                turns: vec![],
            },
        ];
        let sessions = results_to_sessions(&results);
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].session_id, "abc-123");
        assert_eq!(sessions[0].title, "first message"); // stripped [User] prefix
        assert_eq!(sessions[1].session_id, "def-456");
    }

    #[test]
    fn test_results_to_sessions_skips_empty_session_id() {
        let results = vec![
            SearchResult {
                timestamp: "Jan 15 10:30".to_string(),
                project: "server".to_string(),
                session_id: "".to_string(), // e.g. .md file
                session_name: "".to_string(),
                turn: 0,
                path: "/tmp/test.md".to_string(),
                cwd: "".to_string(),
                source: "claude".to_string(),
                branch: "".to_string(),
                bodies: vec!["some content".to_string()],
                match_idx: 0,
                turns: vec![],
            },
        ];
        let sessions = results_to_sessions(&results);
        assert!(sessions.is_empty());
    }
}
