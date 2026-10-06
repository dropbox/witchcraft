use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

mod registration;
pub use registration::register;

struct Options {
    pickbrain: OsString,
    delay: Duration,
    max_delay: Duration,
    service_config: Option<PathBuf>,
}

fn options(mut args: impl Iterator<Item = OsString>) -> Result<Option<Options>> {
    let mut delay = Duration::from_secs(30);
    let mut max_delay = Duration::from_secs(300);
    let mut service_config = None;
    while let Some(arg) = args.next() {
        if arg == "--watch" || arg == "--register" {
            continue;
        } else if arg == "--help" || arg == "-h" {
            println!("Usage: pickbrain --watch [--delay SECONDS] [--max-delay SECONDS]\n\
                      \x20      pickbrain --register [--delay SECONDS] [--max-delay SECONDS]\n\
                      Watch local sessions and spawn pickbrain --ingest-only --quiet.\n\
                      Register or update the current user's background watcher with --register.\n\
                      Defaults: 30 seconds quiet, 300 seconds maximum delay.");
            return Ok(None);
        } else if arg == "--service-config" {
            service_config = Some(PathBuf::from(args.next().context("missing service configuration path")?));
        } else if arg == "--delay" || arg == "--max-delay" {
            let seconds = args.next().context("missing delay in seconds")?;
            let seconds = seconds.to_str().context("delay must be a positive integer")?
                .parse::<u64>().context("delay must be a positive integer")?;
            anyhow::ensure!(seconds > 0, "delay must be positive");
            let duration = Duration::from_secs(seconds);
            Instant::now().checked_add(duration).context("delay is too large")?;
            if arg == "--delay" { delay = duration; } else { max_delay = duration; }
        } else {
            bail!("unexpected watch argument: {}", arg.to_string_lossy());
        }
    }
    anyhow::ensure!(max_delay >= delay, "maximum delay must be at least the quiet delay");
    let pickbrain = env::current_exe().context("locate the running Pickbrain executable")?.into_os_string();
    Ok(Some(Options { pickbrain, delay, max_delay, service_config }))
}

fn normalize(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() { path.to_path_buf() } else {
        env::current_dir().expect("read current directory").join(path)
    };
    // Resolve existing ancestors so paths remain comparable across /tmp symlinks and missing directories.
    for ancestor in absolute.ancestors() {
        if let Ok(resolved) = ancestor.canonicalize() {
            return resolved.join(absolute.strip_prefix(ancestor).unwrap());
        }
    }
    absolute
}

struct Sources {
    roots: Vec<PathBuf>,
    state: PathBuf,
    slack: Option<PathBuf>,
}

impl Sources {
    #[cfg(target_os = "macos")]
    fn snapshot(&self) -> BTreeMap<PathBuf, (u64, i64, i64)> {
        use std::os::unix::fs::MetadataExt;
        let mut files = BTreeMap::new();
        // Restrict the fallback to session trees, avoiding plugin caches and other
        // unrelated files under the agent directories.
        let mut directories: Vec<_> = self.roots.iter()
            .flat_map(|root| [root.join("sessions"), root.join("projects")]).collect();
        while let Some(directory) = directories.pop() {
            let Ok(entries) = std::fs::read_dir(directory) else { continue; };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(kind) = entry.file_type() else { continue; };
                if kind.is_dir() {
                    directories.push(path);
                } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "jsonl" || ext == "md") {
                    if let Ok(metadata) = entry.metadata() {
                        files.insert(path, (metadata.len(), metadata.ctime(), metadata.ctime_nsec()));
                    }
                }
            }
        }
        files
    }

    fn from_env() -> Result<Self> {
        let home = env::var_os("HOME").filter(|home| !home.is_empty())
            .or_else(|| env::var_os("USERPROFILE"))
            .context("HOME or USERPROFILE must be set")?;
        let home = PathBuf::from(home);
        anyhow::ensure!(home.is_dir(), "home directory does not exist: {}", home.display());
        let mut roots = Vec::new();
        for (default, extra) in [(".claude", "EXTRA_CLAUDE_DIRS"),
            (".codex", "EXTRA_CODEX_DIRS"), (".pi/agent", "EXTRA_PI_DIRS")] {
            roots.push(normalize(&home.join(default)));
            if let Some(extra) = env::var_os(extra) {
                roots.extend(env::split_paths(&extra).filter(|path| !path.as_os_str().is_empty())
                    .map(|path| normalize(&path)));
            }
        }
        let slack = if cfg!(target_os = "macos") {
            Some(normalize(&home.join("Library/Application Support/Slack/IndexedDB")))
        } else { None };
        roots.extend(slack.iter().cloned());
        roots.sort();
        roots.dedup();
        let state = env::var_os("PICKBRAIN_DIR").filter(|path| !path.is_empty())
            .map(PathBuf::from).unwrap_or_else(|| home.join(".pickbrain"));
        Ok(Self { roots, state: normalize(&state), slack })
    }

    fn relevant(&self, event: &Event) -> bool {
        if matches!(event.kind, EventKind::Access(_)) { return false; }
        event.paths.iter().any(|path| self.relevant_path(&normalize(path)))
    }

    fn relevant_path(&self, path: &Path) -> bool {
        // Explicit remote source directories may live under ~/.pickbrain; do not exclude those.
        if path.starts_with(&self.state) && !self.roots.iter().any(|root|
            root != &self.state && root.starts_with(&self.state) && path.starts_with(root)) {
            return false;
        }
        if path.file_name().is_some_and(|name| name == "pickbrain.watermark") {
            return false;
        }
        if self.roots.iter().any(|root| root.starts_with(path)) { return true; }
        if !self.roots.iter().any(|root| path.starts_with(root)) { return false; }
        if self.slack.as_ref().is_some_and(|root| path.starts_with(root)) {
            return path.components().any(|part| part.as_os_str().to_string_lossy().ends_with(".indexeddb.blob"));
        }
        path.is_dir() || path.extension().is_some_and(|ext| ext == "jsonl" || ext == "md")
            || path.file_name().is_some_and(|name| name == "pickbrain.remote")
            || path.extension().is_none()
    }

    fn watches(&self) -> BTreeMap<PathBuf, bool> {
        let mut paths = BTreeMap::new();
        for root in &self.roots {
            // FSEvents can coalesce parent creation into descendant events, so missing
            // sources need a recursive ancestor stream on macOS (no per-file handles).
            if let Some(path) = root.ancestors().find(|path| path.is_dir()) {
                let recursive = path == root || cfg!(target_os = "macos");
                paths.entry(path.to_path_buf()).and_modify(|old| *old |= recursive).or_insert(recursive);
            }
        }
        paths
    }
}

fn refresh_watches(watcher: &mut RecommendedWatcher, sources: &Sources,
    current: &mut BTreeMap<PathBuf, bool>, reset: bool) -> Result<()> {
    let desired = sources.watches();
    for (path, recursive) in current.iter() {
        if reset || desired.get(path) != Some(recursive) {
            let _ = watcher.unwatch(path);
        }
    }
    for (path, recursive) in &desired {
        if reset || current.get(path) != Some(recursive) {
            watcher.watch(path, if *recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive })
                .with_context(|| format!("watch {}", path.display()))?;
        }
    }
    *current = desired;
    Ok(())
}

#[derive(Default)]
struct Pending {
    first: Option<Instant>,
    last: Option<Instant>,
}

impl Pending {
    fn mark(&mut self, now: Instant) {
        self.first.get_or_insert(now);
        self.last = Some(now);
    }

    fn deadline(&self, options: &Options) -> Option<Instant> {
        Some((self.last? + options.delay).min(self.first? + options.max_delay))
    }
}

enum Wake {
    Changed,
    Finished(std::io::Result<ExitStatus>),
}

fn spawn_ingest(options: &Options, tx: mpsc::SyncSender<Wake>) -> Result<()> {
    let mut command = Command::new(&options.pickbrain);
    command.args(["--ingest-only", "--quiet"]).stdin(Stdio::null()).stdout(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let mut child = command.spawn().with_context(|| format!("start {}", options.pickbrain.to_string_lossy()))?;
    std::thread::spawn(move || { let _ = tx.send(Wake::Finished(child.wait())); });
    Ok(())
}

fn run(options: Options, sources: Sources) -> Result<()> {
    let sources = Arc::new(sources);
    let dirty = Arc::new(AtomicBool::new(false));
    let reset = Arc::new(AtomicBool::new(false));
    // A burst of filesystem events occupies one slot; dirty also preserves events when the slot is full.
    let (tx, rx) = mpsc::sync_channel(1);
    let callback_tx = tx.clone();
    let callback_sources = sources.clone();
    let callback_dirty = dirty.clone();
    let callback_reset = reset.clone();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
        let relevant = match event {
            Ok(event) => {
                if event.need_rescan() { callback_reset.store(true, Ordering::SeqCst); }
                event.need_rescan() || callback_sources.relevant(&event)
            },
            Err(error) => {
                eprintln!("filesystem watcher: {error}; rescanning");
                callback_reset.store(true, Ordering::SeqCst);
                true
            },
        };
        if relevant {
            callback_dirty.store(true, Ordering::SeqCst);
            let _ = callback_tx.try_send(Wake::Changed);
        }
    })?;
    let mut watches = BTreeMap::new();
    refresh_watches(&mut watcher, &sources, &mut watches, false)?;
    // FSEvents may defer writes to a file until its writer closes it. Codex keeps
    // rollout files open, so metadata checks cover updates absent from that stream.
    #[cfg(target_os = "macos")]
    let (mut snapshot, mut check_at) = (sources.snapshot(), Instant::now() + options.delay);
    let mut pending = Pending::default();
    pending.mark(Instant::now()); // Catch up on files changed while the supervisor was stopped.
    let mut running = false;
    loop {
        #[cfg(target_os = "macos")]
        if Instant::now() >= check_at {
            let next = sources.snapshot();
            if next != snapshot {
                pending.mark(Instant::now());
                refresh_watches(&mut watcher, &sources, &mut watches, false)?;
                snapshot = next;
            }
            check_at = Instant::now() + options.delay;
        }
        if dirty.swap(false, Ordering::SeqCst) {
            pending.mark(Instant::now());
            refresh_watches(&mut watcher, &sources, &mut watches, reset.swap(false, Ordering::SeqCst))?;
        }
        let deadline = if running { None } else { pending.deadline(&options) };
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            match spawn_ingest(&options, tx.clone()) {
                Ok(()) => { running = true; pending = Pending::default(); },
                Err(error) => {
                    eprintln!("{error:#}; retrying after the quiet delay");
                    pending = Pending::default();
                    pending.mark(Instant::now());
                },
            }
            continue;
        }
        #[cfg(target_os = "macos")]
        let deadline = Some(deadline.map_or(check_at, |deadline| deadline.min(check_at)));
        let wake = match deadline {
            Some(deadline) => match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(wake) => wake,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("watcher disconnected"),
            },
            None => rx.recv().context("watcher disconnected")?,
        };
        if let Wake::Finished(status) = wake {
            running = false;
            match status {
                Ok(status) if status.success() => {},
                status => {
                    eprintln!("ingestion failed ({status:?}); retrying after the quiet delay");
                    pending = Pending::default();
                    pending.mark(Instant::now());
                },
            }
        }
    }
}

pub fn watch(args: impl Iterator<Item = OsString>) -> Result<()> {
    let Some(options) = options(args)? else { return Ok(()); };
    if let Some(path) = &options.service_config {
        registration::restore(path)?;
    }
    run(options, Sources::from_env()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watch_mode_uses_current_executable_and_rejects_external_binary_paths() {
        let options = options(["--watch", "--delay", "1", "--max-delay", "3"]
            .into_iter().map(OsString::from)).unwrap().unwrap();
        assert_eq!(PathBuf::from(options.pickbrain), env::current_exe().unwrap());
        assert!(super::options(["--watch", "/other/pickbrain"].into_iter().map(OsString::from)).is_err());
    }

    #[test]
    fn debounce_extends_quiet_time_but_bounds_continuous_writes() {
        let options = Options { pickbrain: "unused".into(), delay: Duration::from_secs(30), max_delay: Duration::from_secs(300), service_config: None };
        let start = Instant::now();
        let mut pending = Pending::default();
        assert!(pending.deadline(&options).is_none());
        pending.mark(start);
        assert_eq!(pending.deadline(&options), Some(start + options.delay));
        pending.mark(start + Duration::from_secs(290));
        assert_eq!(pending.deadline(&options), Some(start + options.max_delay));
        pending = Pending::default();
        pending.mark(start + Duration::from_secs(310));
        assert_eq!(pending.deadline(&options), Some(start + Duration::from_secs(340)));
    }

    #[test]
    fn source_events_exclude_state_watermarks_and_reads_but_allow_remote_sources() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let state = root.join(".pickbrain");
        let remote = state.join("remote-codex");
        let sources = Sources { roots: vec![root.join(".codex"), remote.clone()], state: state.clone(), slack: None };
        assert!(sources.relevant_path(&root.join(".codex/sessions/new.jsonl")));
        assert!(sources.relevant_path(&remote.join("sessions/new.jsonl")));
        assert!(!sources.relevant_path(&state.join("pickbrain.db")));
        assert!(!sources.relevant_path(&root.join(".codex/pickbrain.watermark")));
        assert!(!sources.relevant_path(&root.join(".codex/auth.json")));
        assert!(!sources.relevant(&Event::new(EventKind::Access(notify::event::AccessKind::Any))
            .add_path(root.join(".codex/sessions/new.jsonl"))));
    }

    #[test]
    fn missing_sources_watch_ancestors_then_switch_to_the_source() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let source = root.join("missing/sessions");
        let sources = Sources { roots: vec![source.clone()], state: root.join("state"), slack: None };
        assert_eq!(sources.watches(), BTreeMap::from([(root.clone(), cfg!(target_os = "macos"))]));
        assert!(sources.relevant_path(&root.join("missing")));
        std::fs::create_dir_all(&source).unwrap();
        assert_eq!(sources.watches(), BTreeMap::from([(source.clone(), true)]));
        std::fs::remove_dir_all(&root.join("missing")).unwrap();
        assert_eq!(sources.watches(), BTreeMap::from([(root, cfg!(target_os = "macos"))]));
    }
}

#[cfg(all(test, unix))]
mod supervisor_tests;
