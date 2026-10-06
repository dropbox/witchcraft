use super::{run, Options, Sources};

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

struct Supervisor(Child);

impl Drop for Supervisor {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start(root: &Path, script: &str) -> Supervisor {
    let executable = root.join("mock-pickbrain");
    fs::write(&executable, script).unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
    fs::create_dir(root.join(".pickbrain")).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "supervisor_tests::watcher_process_entry", "--nocapture"])
        .env("PICKBRAIN_WATCH_TEST_ROOT", root)
        .env("HOME", root).env("PICKBRAIN_DIR", root.join(".pickbrain"))
        .env_remove("EXTRA_CLAUDE_DIRS").env_remove("EXTRA_CODEX_DIRS").env_remove("EXTRA_PI_DIRS")
        .env_remove("PRE_INGEST_COMMAND")
        .stdin(Stdio::null()).stdout(Stdio::null()).spawn().unwrap();
    Supervisor(child)
}

fn lines(path: &Path) -> usize {
    fs::read_to_string(path).unwrap_or_default().lines().count()
}

fn wait_for(path: &Path, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while lines(path) < count {
        assert!(Instant::now() < deadline, "timed out waiting for {} entries in {}", count, path.display());
        sleep(Duration::from_millis(10));
    }
}

#[test]
fn watches_new_sources_coalesces_writes_and_never_overlaps_children() {
    let root = tempfile::tempdir().unwrap();
    let mut supervisor = start(root.path(), r#"#!/bin/sh
set -eu
test "$*" = "--ingest-only --quiet"
mkdir "$PICKBRAIN_DIR/running" || { touch "$PICKBRAIN_DIR/overlap"; exit 1; }
echo start >> "$PICKBRAIN_DIR/starts"
sleep 2
rmdir "$PICKBRAIN_DIR/running"
echo done >> "$PICKBRAIN_DIR/ends"
"#);
    let state = root.path().join(".pickbrain");
    wait_for(&state.join("starts"), 1);
    let sessions = root.path().join(".pi/agent/sessions");
    fs::create_dir_all(&sessions).unwrap();
    for change in 0..100 {
        fs::write(sessions.join("session.jsonl"), change.to_string()).unwrap();
    }
    wait_for(&state.join("ends"), 2);
    fs::write(state.join("watcher.log"), "do not trigger myself").unwrap();
    fs::write(root.path().join(".pi/agent/pickbrain.watermark"), "").unwrap();
    sleep(Duration::from_secs(2));
    assert_eq!(lines(&state.join("starts")), 2);
    assert!(!state.join("overlap").exists());
    assert!(supervisor.0.try_wait().unwrap().is_none());
}

#[test]
fn notices_appends_to_an_open_codex_session() {
    use std::io::Write;
    let root = tempfile::tempdir().unwrap();
    let sessions = root.path().join(".codex/sessions/2026/10/05");
    fs::create_dir_all(&sessions).unwrap();
    let path = sessions.join("rollout-session.jsonl");
    let mut session = fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
    writeln!(session, "initial").unwrap();
    session.flush().unwrap();
    let _supervisor = start(root.path(), r#"#!/bin/sh
echo start >> "$PICKBRAIN_DIR/starts"
"#);
    let starts = root.path().join(".pickbrain/starts");
    wait_for(&starts, 1);
    sleep(Duration::from_secs(2));
    let baseline = lines(&starts);
    writeln!(session, "updated while the writer keeps the file open").unwrap();
    session.flush().unwrap();
    wait_for(&starts, baseline + 1);
    writeln!(session, "another update through the same open handle").unwrap();
    session.flush().unwrap();
    wait_for(&starts, baseline + 2);
}

#[test]
fn failed_child_retries_without_another_filesystem_event() {
    let root = tempfile::tempdir().unwrap();
    let mut supervisor = start(root.path(), r#"#!/bin/sh
set -eu
echo start >> "$PICKBRAIN_DIR/starts"
if test ! -f "$PICKBRAIN_DIR/failed"; then
    touch "$PICKBRAIN_DIR/failed"
    exit 1
fi
touch "$PICKBRAIN_DIR/success"
"#);
    let state = root.path().join(".pickbrain");
    wait_for(&state.join("starts"), 2);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !state.join("success").exists() {
        assert!(Instant::now() < deadline);
        sleep(Duration::from_millis(10));
    }
    sleep(Duration::from_secs(2));
    assert_eq!(lines(&state.join("starts")), 2);
    assert!(supervisor.0.try_wait().unwrap().is_none());
}

#[test]
#[ignore = "subprocess entry point for supervisor tests"]
fn watcher_process_entry() {
    let root = std::path::PathBuf::from(std::env::var_os("PICKBRAIN_WATCH_TEST_ROOT").unwrap());
    let options = Options {
        pickbrain: root.join("mock-pickbrain").into_os_string(),
        delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(3),
        service_config: None,
    };
    run(options, Sources::from_env().unwrap()).unwrap();
}
