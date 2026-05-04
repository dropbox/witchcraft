mod v8;

use rusty_leveldb::{LdbIterator, Options, DB};
use std::cmp::Ordering;
use std::path::Path;
use std::rc::Rc;
use std::fs;

/// Stub comparator that satisfies LevelDB's name check for Chrome's IndexedDB.
/// The actual ordering doesn't matter for a sequential dump — we just need the
/// id to match the one stored in the MANIFEST so LevelDB will open the DB.
struct IdbCmp1;

impl rusty_leveldb::Cmp for IdbCmp1 {
    fn cmp(&self, a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }
    fn id(&self) -> &'static str {
        "idb_cmp1"
    }
    fn find_shortest_sep(&self, a: &[u8], _b: &[u8]) -> Vec<u8> {
        a.to_vec()
    }
    fn find_short_succ(&self, a: &[u8]) -> Vec<u8> {
        a.to_vec()
    }
}

/// Open a Chrome IndexedDB LevelDB directory and dump every value to JSON.
///
/// Returns a JSON array of objects, one per IndexedDB record. Each object has
/// `key` (hex-encoded raw LevelDB key) and `value` (the deserialized V8
/// structured-clone payload, or null if decoding failed).
pub fn dump(path: &Path, debug_raw: bool) -> Result<Vec<serde_json::Value>, String> {
    let mut opts = Options::default();
    // Default compressor list already has NoneCompressor (0) and SnappyCompressor (1).
    opts.compressor = 0;
    opts.cmp = Rc::new(Box::new(IdbCmp1));
    opts.create_if_missing = false;

    let mut db = DB::open(path, opts).map_err(|e| format!("Failed to open DB: {}", e))?;

    let mut results = Vec::new();
    let mut iter = db.new_iter().map_err(|e| format!("Failed to create iterator: {}", e))?;

    loop {
        let entry = iter.next();
        if entry.is_none() {
            break;
        }
        let (key, val) = entry.unwrap();

        // Try to decode the value as a V8 structured clone.
        let decoded = v8::deserialize(&val);

        if debug_raw && decoded.is_none() && val.len() > 10 {
            let preview: String = val.iter().take(30).map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
            eprintln!("FAIL key={} len={} preview={}", hex_encode(&key).chars().take(40).collect::<String>(), val.len(), preview);
        }

        let mut entry = serde_json::json!({
            "key": hex_encode(&key),
            "value": decoded.unwrap_or(serde_json::Value::Null),
        });
        if debug_raw {
            entry["raw_value_hex"] = serde_json::Value::String(hex_encode(&val));
            entry["raw_value_len"] = serde_json::Value::Number(val.len().into());
        }
        results.push(entry);
    }

    Ok(results)
}

/// Read blob files from a Chrome IndexedDB `.indexeddb.blob` directory.
///
/// Recursively walks the directory, deserializes each blob file as a V8
/// structured-clone payload, and returns JSON entries with `blob_path`,
/// `value`, and `blob_size`.
pub fn dump_blobs(blob_dir: &Path, debug_raw: bool) -> Result<Vec<serde_json::Value>, String> {
    let mut results = Vec::new();
    let mut files = Vec::new();
    collect_files(blob_dir, &mut files)
        .map_err(|e| format!("Failed to walk blob dir: {}", e))?;
    files.sort();

    for file_path in &files {
        let data = fs::read(file_path)
            .map_err(|e| format!("Failed to read blob {}: {}", file_path.display(), e))?;

        let rel_path = file_path
            .strip_prefix(blob_dir)
            .unwrap_or(file_path)
            .to_string_lossy()
            .into_owned();

        let decoded = v8::deserialize_blob(&data);

        if debug_raw && decoded.is_none() && data.len() > 10 {
            let preview: String = data
                .iter()
                .take(30)
                .map(|b| format!("{:02x}", b))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "BLOB FAIL path={} len={} preview={}",
                rel_path,
                data.len(),
                preview
            );
        }

        let mut entry = serde_json::json!({
            "blob_path": rel_path,
            "blob_size": data.len(),
            "value": decoded.unwrap_or(serde_json::Value::Null),
        });
        if debug_raw {
            let preview_hex: String = data
                .iter()
                .take(64)
                .map(|b| format!("{:02x}", b))
                .collect();
            entry["raw_preview_hex"] = serde_json::Value::String(preview_hex);
        }
        results.push(entry);
    }

    Ok(results)
}

fn collect_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
