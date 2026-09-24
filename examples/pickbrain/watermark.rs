use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub fn source_dirs(default: &str, extra_env: &str) -> Vec<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut dirs = vec![PathBuf::from(home).join(default)];
    if let Ok(extra) = std::env::var(extra_env) {
        append_extra_dirs(&mut dirs, &extra);
    }
    dirs
}

fn append_extra_dirs(dirs: &mut Vec<PathBuf>, extra: &str) {
    for part in extra.split(':').filter(|part| !part.is_empty()) {
        let dir = PathBuf::from(part);
        if !dirs.contains(&dir) { dirs.push(dir); }
    }
}

fn dirs_state_path(source: &str) -> PathBuf {
    crate::pickbrain_dir().join(format!("{source}.source_dirs"))
}

fn dirs_state(dirs: &[PathBuf]) -> String {
    dirs.iter().map(|dir| dir_state(dir)).collect::<Vec<_>>().join("\n")
}

fn dir_state(dir: &Path) -> String {
    let mut state = dir.to_string_lossy().to_string();
    if let Ok(origin) = fs::read_to_string(dir.join("pickbrain.remote")) {
        state.push('\t');
        state.push_str(origin.trim());
    }
    state
}

pub fn record_dirs(source: &str, dirs: &[PathBuf]) {
    let _ = fs::write(dirs_state_path(source), dirs_state(dirs));
}

pub fn mtime_for_dir(source: &str, dirs: &[PathBuf], dir: &Path, path: &Path) -> i64 {
    let saved = fs::read_to_string(dirs_state_path(source));
    if dir_state_is_current(saved.as_deref().ok(), dirs, dir) { mtime_ms(path) } else { 0 }
}

fn dir_state_is_current(saved: Option<&str>, dirs: &[PathBuf], dir: &Path) -> bool {
    match saved {
        Some(saved) => saved.lines().any(|line| line == dir_state(dir)),
        None => dirs.first().is_some_and(|default| default == dir),
    }
}

fn watermark_path(agent_dir: &str) -> PathBuf {
    if crate::pickbrain_dir_overridden() {
        let name = agent_dir.trim_start_matches('.');
        return crate::pickbrain_dir().join(format!("{name}.watermark"));
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(agent_dir)
        .join("pickbrain.watermark")
}

pub fn claude_path() -> PathBuf {
    watermark_path(".claude")
}

pub fn codex_path() -> PathBuf {
    watermark_path(".codex")
}

pub fn pi_path() -> PathBuf {
    watermark_path(".pi/agent")
}

pub fn mtime_ms(path: &Path) -> i64 {
    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn touch(path: &Path) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, "");
}

pub fn remove(path: &Path) {
    let _ = fs::remove_file(path);
}

pub fn is_fresh(path: &Path, max_age_ms: i64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let wm = mtime_ms(path);
    wm > 0 && (now - wm) < max_age_ms
}

pub fn file_newer_than(file: &Path, watermark: i64) -> bool {
    let Ok(metadata) = fs::metadata(file) else { return false; };
    let modified_ms = metadata.modified().ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0);
    if modified_ms > watermark { return true; }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let changed_ms = metadata.ctime() * 1000 + i64::from(metadata.ctime_nsec()) / 1_000_000;
        changed_ms > watermark
    }
    #[cfg(not(unix))]
    {
        metadata.created().ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .is_some_and(|duration| duration.as_millis() as i64 > watermark)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copied_file_with_old_mtime_is_new_work() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let watermark = mtime_ms(file.path()) - 1_000;
        let times = fs::FileTimes::new().set_modified(UNIX_EPOCH);
        file.as_file().set_times(times).unwrap();
        assert!(file_newer_than(file.path(), watermark));
    }

    #[test]
    fn extra_dirs_are_colon_separated_and_deduplicated() {
        let mut dirs = vec![PathBuf::from("/default")];
        append_extra_dirs(&mut dirs, "/one::/two:/one:/default");
        assert_eq!(dirs, vec!["/default", "/one", "/two"].into_iter().map(PathBuf::from).collect::<Vec<_>>());
    }

    #[test]
    fn remote_marker_changes_directory_state() {
        let default = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let dirs = vec![default.path().to_path_buf(), dir.path().to_path_buf()];
        let before = dirs_state(&dirs);
        fs::write(dir.path().join("pickbrain.remote"), "remote.example\n").unwrap();
        assert_ne!(dirs_state(&dirs), before);
        assert!(dir_state_is_current(Some(&before), &dirs, default.path()));
        assert!(!dir_state_is_current(Some(&before), &dirs, dir.path()));
    }
}
