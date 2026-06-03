use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

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
    mtime_ms(file) > watermark
}
