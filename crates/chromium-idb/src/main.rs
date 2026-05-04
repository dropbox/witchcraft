use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let debug_raw = args.iter().any(|a| a == "--debug");
    let blobs_only = args.iter().any(|a| a == "--blobs-only");
    let path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from);

    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("Usage: idb-dump [--debug] [--blobs-only] <path-to-indexeddb-leveldb-dir>");
            eprintln!();
            eprintln!("Dumps Chrome IndexedDB records (LevelDB + blob files) to JSON.");
            eprintln!("Automatically reads sibling .indexeddb.blob directory if present.");
            std::process::exit(1);
        }
    };

    let mut all_records = Vec::new();

    // Dump LevelDB records (unless --blobs-only)
    if !blobs_only {
        match chromium_idb::dump(&path, debug_raw) {
            Ok(records) => {
                all_records.extend(records);
            }
            Err(e) => {
                eprintln!("Error reading LevelDB: {}", e);
                if blobs_only {
                    // Continue to blobs
                } else {
                    std::process::exit(1);
                }
            }
        }
    }

    // Auto-detect sibling .indexeddb.blob directory
    let blob_dir = find_blob_dir(&path);
    if let Some(ref blob_path) = blob_dir {
        eprintln!("Reading blobs from: {}", blob_path.display());
        match chromium_idb::dump_blobs(blob_path, debug_raw) {
            Ok(blobs) => {
                all_records.extend(blobs);
            }
            Err(e) => {
                eprintln!("Error reading blobs: {}", e);
            }
        }
    }

    let json = serde_json::to_string_pretty(&all_records).expect("JSON serialization failed");
    println!("{}", json);
}

/// Given a `.indexeddb.leveldb` directory path, find the sibling `.indexeddb.blob` directory.
/// e.g. `/path/to/https_foo_0.indexeddb.leveldb` → `/path/to/https_foo_0.indexeddb.blob`
fn find_blob_dir(leveldb_path: &PathBuf) -> Option<PathBuf> {
    let name = leveldb_path.file_name()?.to_str()?;
    let blob_name = name.replace(".indexeddb.leveldb", ".indexeddb.blob");
    if blob_name == name {
        // Didn't contain the expected suffix — try the path as-is with .blob appended
        return None;
    }
    let blob_path = leveldb_path.with_file_name(blob_name);
    if blob_path.is_dir() {
        Some(blob_path)
    } else {
        None
    }
}
