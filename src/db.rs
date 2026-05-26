use super::types::SqlStatementInternal;
use iso8601_timestamp::Timestamp;
use log::{error, warn};
use rusqlite::{params_from_iter, Connection, OpenFlags, Result as SQLResult, Statement};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use super::sql_generator::build_filter_sql_and_params;

const HASH_CHARS: usize = 32; // we'll use sha256 truncated at 128 bits/32 characters
const APP_ID: i32 = 0x07DB_DA55;
const SCHEMA_VERSION: i32 = 10;

pub struct DB {
    db_fn: PathBuf,
    connection: Option<Connection>,
    remove_on_shutdown: bool,
    recreated: bool,
    read_only: bool,
}

impl DB {
    fn conn(&self) -> &Connection {
        self.connection.as_ref().expect("Connection should exist")
    }

    pub fn was_recreated(&self) -> bool {
        self.recreated
    }

    pub fn path(&self) -> &PathBuf {
        &self.db_fn
    }

    fn sidecar_path(db_fn: &Path, suffix: &str) -> PathBuf {
        let mut path = db_fn.as_os_str().to_os_string();
        path.push(suffix);
        PathBuf::from(path)
    }

    fn bucket_data_prefix(db_fn: &Path) -> String {
        let base = db_fn
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "warp.sqlite".into());
        format!("{base}.buckets.")
    }

    fn db_parent(db_fn: &Path) -> &Path {
        match db_fn.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        }
    }

    fn remove_sidecars(db_fn: &Path) {
        let _ = std::fs::remove_file(Self::sidecar_path(db_fn, "-wal"));
        let _ = std::fs::remove_file(Self::sidecar_path(db_fn, "-shm"));
    }

    fn remove_bucket_data_sidecars(db_fn: &Path) {
        let parent = Self::db_parent(db_fn);
        let prefix = Self::bucket_data_prefix(db_fn);
        let old_prefix = Self::old_residuals_prefix(db_fn);
        let Ok(entries) = std::fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) || name.starts_with(&old_prefix) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    fn old_residuals_prefix(db_fn: &Path) -> String {
        let base = db_fn
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "warp.sqlite".into());
        format!("{base}.residuals.")
    }

    fn configure(connection: &Connection) -> SQLResult<()> {
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        connection.pragma_update(None, "mmap_size", 512 * 1024 * 1024)?;
        Ok(())
    }

    fn schema_matches(connection: &Connection) -> bool {
        let app_id: SQLResult<i32> =
            connection.query_row("PRAGMA application_id;", [], |r| r.get(0));
        let user_version: SQLResult<i32> =
            connection.query_row("PRAGMA user_version;", [], |r| r.get(0));
        matches!(
            (app_id, user_version),
            (Ok(a), Ok(v)) if a == APP_ID && v == SCHEMA_VERSION
        )
    }

    fn create_schema(connection: &Connection) -> SQLResult<()> {
        connection.execute_batch(&format!(
            "PRAGMA application_id = {APP_ID};
             PRAGMA user_version = {SCHEMA_VERSION};

             CREATE TABLE document(
                 uuid TEXT NOT NULL PRIMARY KEY,
                 date TEXT NOT NULL,
                 metadata JSON,
                 hash TEXT CHECK (length(hash) = {HASH_CHARS}),
                 body TEXT,
                 lens TEXT);

             CREATE INDEX document_index ON document(hash);

             CREATE VIRTUAL TABLE document_fts
                 USING fts5(body, content='document', content_rowid='rowid');
             INSERT INTO document_fts(document_fts) VALUES('rebuild');

             CREATE TRIGGER document_fts_insert AFTER INSERT ON document
             BEGIN
                 INSERT INTO document_fts(rowid, body) VALUES (new.rowid, new.body);
             END;

             CREATE TRIGGER document_fts_delete AFTER DELETE ON document
             BEGIN
                 INSERT INTO document_fts(document_fts, rowid, body)
                     VALUES('delete', old.rowid, old.body);
             END;

             CREATE TRIGGER document_fts_update AFTER UPDATE ON document
             BEGIN
                 INSERT INTO document_fts(document_fts, rowid, body)
                     VALUES('delete', old.rowid, old.body);
                 INSERT INTO document_fts(rowid, body) VALUES (new.rowid, new.body);
             END;

             CREATE TABLE chunk(
                 hash TEXT PRIMARY KEY CHECK (length(hash) = {HASH_CHARS}),
                 model TEXT,
                 embeddings BLOB NOT NULL,
                 counts TEXT NOT NULL,
                 embedding_count INTEGER NOT NULL);

             CREATE TRIGGER document_after_delete AFTER DELETE ON document
             BEGIN
                 DELETE FROM chunk
                     WHERE hash = OLD.hash
                     AND NOT EXISTS (SELECT 1 FROM document WHERE hash = OLD.hash);
             END;

             CREATE TRIGGER document_after_update AFTER UPDATE ON document
             BEGIN
                 DELETE FROM chunk
                     WHERE hash = OLD.hash
                     AND NOT EXISTS (SELECT 1 FROM document WHERE hash = OLD.hash);
             END;

             CREATE TABLE generation(
                 id INTEGER PRIMARY KEY,
                 level INTEGER NOT NULL,
                 num_embeddings INTEGER NOT NULL,
                 min_chunk_rowid INTEGER NOT NULL,
                 max_chunk_rowid INTEGER NOT NULL,
                 bucket_data_file TEXT NOT NULL);"
        ))?;
        Ok(())
    }

    pub fn new_reader(db_fn: PathBuf) -> SQLResult<Self> {
        let connection =
            Connection::open_with_flags(db_fn.clone(), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        connection.pragma_update(None, "mmap_size", 512 * 1024 * 1024)?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(Self {
            db_fn,
            connection: Some(connection),
            remove_on_shutdown: false,
            recreated: false,
            read_only: true,
        })
    }

    fn integrity_ok(connection: &Connection) -> bool {
        let status: SQLResult<String> =
            connection.query_row("PRAGMA quick_check;", [], |row| row.get(0));
        matches!(status, Ok(text) if text.trim().eq_ignore_ascii_case("ok"))
    }

    fn open_internal(db_fn: PathBuf, fast: bool) -> SQLResult<Self> {
        let mut first_creation = !db_fn.exists();
        let mut connection = Connection::open(&db_fn)?;

        let db_ok = fast || first_creation || Self::integrity_ok(&connection);
        let schema_ok = first_creation || Self::schema_matches(&connection);

        let mut recreated = false;
        if !db_ok || !schema_ok {
            warn!(
                "warp database {} {}, recreating!",
                db_fn.display(),
                if !db_ok { "corrupted" } else { "schema mismatch" }
            );
            recreated = !first_creation;
            connection.close().map_err(|(_conn, e)| e)?;
            std::fs::remove_file(&db_fn)
                .map_err(|_e| rusqlite::Error::InvalidPath(db_fn.clone()))?;
            Self::remove_sidecars(&db_fn);
            Self::remove_bucket_data_sidecars(&db_fn);
            connection = Connection::open(&db_fn)?;
            first_creation = true;
        }

        Self::configure(&connection)?;

        if first_creation {
            Self::create_schema(&connection)?;
        }

        Ok(Self {
            db_fn,
            connection: Some(connection),
            remove_on_shutdown: false,
            recreated,
            read_only: false,
        })
    }

    /// Open with integrity check — safe for in-process use.
    pub fn new(db_fn: PathBuf) -> SQLResult<Self> {
        Self::open_internal(db_fn, false)
    }

    /// Open without integrity check — for CLI batch operations where startup
    /// latency on large databases is prohibitive.
    pub fn new_fast(db_fn: PathBuf) -> SQLResult<Self> {
        Self::open_internal(db_fn, true)
    }

    fn clear_inner(&mut self) -> SQLResult<()> {
        self.execute("DELETE FROM document")?;
        self.execute("DELETE FROM chunk")?;
        self.execute("DELETE FROM generation")?;
        self.remove_all_bucket_data_sidecars();
        self.execute("VACUUM")?;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.remove_on_shutdown = true;
        let _ = self.clear_inner();
    }

    pub fn delete_with_filter(&mut self, sql_filter: &SqlStatementInternal) -> SQLResult<()> {
        let (filter_sql, params) = build_filter_sql_and_params(Some(sql_filter))
            .map_err(|err| rusqlite::Error::ToSqlConversionFailure(err.into()))?;

        if filter_sql.trim().is_empty() {
            return self.clear_inner();
        }

        let delete_sql = format!("DELETE FROM document WHERE {filter_sql}");
        let mut statement = self.conn().prepare(&delete_sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|param| param.as_ref() as &dyn rusqlite::ToSql)
            .collect();
        statement.execute(params_from_iter(param_refs))?;
        Ok(())
    }

    /// Internal helper to checkpoint and truncate the WAL.
    fn checkpoint_internal(connection: &rusqlite::Connection, log_errors: bool) {
        if let Err(e) = connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)") {
            if log_errors {
                warn!("wal_checkpoint failed: {e}");
            }
        }
    }

    pub fn shutdown(&mut self) {
        if let Some(connection) = self.connection.take() {
            // Checkpoint and truncate the WAL file so the main .sqlite file is
            // self-contained on exit (no stale -wal / -shm files left behind).
            if !self.read_only {
                Self::checkpoint_internal(&connection, false);
            }
            match connection.close() {
                Ok(_) => {}
                Err((conn, e)) => {
                    error!("failed to close db connection: {e}");
                    drop(conn);
                }
            };
        }

        if self.remove_on_shutdown {
            // Remove main database file
            match std::fs::remove_file(&self.db_fn) {
                Ok(()) => {
                    self.remove_on_shutdown = false;
                }
                Err(v) => {
                    warn!(
                        "unable to remove database file {}: {v}",
                        self.db_fn.display()
                    );
                }
            };

            // Also remove WAL and SHM files if they exist
            Self::remove_sidecars(&self.db_fn);
            Self::remove_bucket_data_sidecars(&self.db_fn);
        }
    }

    /// Checkpoint and truncate the WAL into the main database file.
    /// Safe to call at any point when no statements are active on this connection.
    pub fn checkpoint(&self) {
        if !self.read_only {
            if let Some(connection) = self.connection.as_ref() {
                Self::checkpoint_internal(connection, true);
            }
        }
    }

    pub fn file_size(&self) -> std::io::Result<u64> {
        std::fs::metadata(&self.db_fn).map(|meta| meta.len())
    }

    pub fn bucket_data_file_name(&self, generation_id: i64) -> String {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        format!("{}{generation_id}.{nonce}", Self::bucket_data_prefix(&self.db_fn))
    }

    pub fn bucket_data_path(&self, file_name: &str) -> PathBuf {
        let path = Path::new(file_name);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            Self::db_parent(&self.db_fn).join(path)
        }
    }

    pub fn remove_bucket_data_file(&self, file_name: &str) {
        let path = self.bucket_data_path(file_name);
        if let Err(e) = std::fs::remove_file(&path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("unable to remove bucket data sidecar {}: {e}", path.display());
            }
        }
    }

    pub fn remove_bucket_data_files(&self, file_names: &[String]) {
        for file_name in file_names {
            self.remove_bucket_data_file(file_name);
        }
    }

    pub fn remove_all_bucket_data_sidecars(&self) {
        Self::remove_bucket_data_sidecars(&self.db_fn);
    }

    pub fn execute(&self, sql: &str) -> SQLResult<()> {
        match self.conn().execute(sql, ()) {
            Ok(_v) => Ok(()),
            Err(v) => {
                error!("failed to execute SQL {v}");
                Err(v)
            }
        }
    }

    pub fn query(&self, sql: &str) -> SQLResult<Statement<'_>> {
        self.conn().prepare(sql)
    }

    pub fn begin_transaction(&self) -> SQLResult<()> {
        self.conn().execute("BEGIN", ())?;
        Ok(())
    }

    pub fn commit_transaction(&self) -> SQLResult<()> {
        self.conn().execute("COMMIT", ())?;
        Ok(())
    }

    pub fn rollback_transaction(&self) -> SQLResult<()> {
        self.conn().execute("ROLLBACK", ())?;
        Ok(())
    }

    pub fn add_doc(
        &mut self,
        uuid: &Uuid,
        date: Option<Timestamp>,
        metadata: &str,
        body: &str,
        lens: Option<Vec<usize>>,
    ) -> SQLResult<()> {
        self.add_docs_batch(&[(*uuid, date, metadata, body, lens)])?;
        Ok(())
    }

    /// Batch-add documents in a single transaction. Prepares the statement once
    /// and reuses it for all inserts. Much faster than individual add_doc calls.
    pub fn add_docs_batch(
        &mut self,
        docs: &[(Uuid, Option<Timestamp>, &str, &str, Option<Vec<usize>>)],
    ) -> SQLResult<usize> {
        if docs.is_empty() {
            return Ok(0);
        }
        self.conn().execute("BEGIN", ())?;
        let mut stmt = self.conn().prepare(
            "INSERT INTO document VALUES(?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(uuid) DO UPDATE SET
                date = ?2, metadata = ?3, hash = ?4, body = ?5, lens = ?6",
        )?;

        let mut count = 0;
        for (uuid, date, metadata, body, lens) in docs {
            let lens = match lens {
                Some(lens) => lens.clone(),
                None => vec![body.chars().count()],
            };
            let total: usize = lens.iter().copied().sum();
            if total != body.chars().count() {
                warn!("bad length: [{} vs {}]", total, body.chars().count());
            }
            let lens_str = lens
                .iter()
                .map(|len| len.to_string())
                .collect::<Vec<_>>()
                .join(",");

            let mut hasher = Sha256::new();
            hasher.update(body.as_bytes());
            hasher.update(lens_str.as_bytes());
            let hash: String = hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let hash = &hash[..HASH_CHARS];

            let date = date.unwrap_or_else(Timestamp::now_utc);
            stmt.execute((
                &uuid.to_string(),
                date.to_string(),
                *metadata,
                hash,
                *body,
                &lens_str,
            ))?;
            count += 1;
        }
        drop(stmt);
        self.conn().execute("COMMIT", ())?;
        self.remove_on_shutdown = false;
        Ok(count)
    }

    pub fn remove_doc(&mut self, uuid: &Uuid) -> SQLResult<()> {
        self.conn()
            .execute("DELETE FROM document WHERE uuid = ?1", (uuid.to_string(),))?;
        Ok(())
    }

    pub fn add_chunk(
        &self,
        hash: &str,
        model: &str,
        embeddings: &Vec<u8>,
        counts: &str,
        embedding_count: usize,
    ) -> SQLResult<()> {
        self.conn().execute(
            "INSERT OR IGNORE INTO chunk(hash, model, embeddings, counts, embedding_count)
             VALUES(?1, ?2, ?3, ?4, ?5)",
            (&hash, &model, embeddings, counts, embedding_count as i64),
        )?;
        Ok(())
    }

    pub fn add_generation(
        &self,
        level: u32,
        num_embeddings: u64,
        min_chunk_rowid: i64,
        max_chunk_rowid: i64,
    ) -> SQLResult<i64> {
        self.conn().execute(
            "INSERT INTO generation(level, num_embeddings, min_chunk_rowid, max_chunk_rowid, bucket_data_file)
             VALUES(?1, ?2, ?3, ?4, '')",
            (level, num_embeddings as i64, min_chunk_rowid, max_chunk_rowid),
        )?;
        Ok(self.conn().last_insert_rowid())
    }

    pub fn set_generation_bucket_data_file(
        &self,
        generation_id: i64,
        bucket_data_file: &str,
    ) -> SQLResult<()> {
        self.conn().execute(
            "UPDATE generation SET bucket_data_file = ?1 WHERE id = ?2",
            (bucket_data_file, generation_id),
        )?;
        Ok(())
    }
}

impl Drop for DB {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            if !self.read_only {
                Self::checkpoint_internal(&connection, false);
            }
            match connection.close() {
                Ok(_) => {}
                Err((conn, e)) => {
                    error!("failed to close db connection in Drop: {e}");
                    drop(conn);
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DB;
    use std::path::{Path, PathBuf};

    #[test]
    fn db_parent_uses_current_dir_for_bare_relative_path() {
        assert_eq!(DB::db_parent(Path::new("mydb.sqlite")), Path::new("."));
        assert_eq!(DB::db_parent(Path::new("data/mydb.sqlite")), Path::new("data"));
    }

    #[test]
    fn removes_sidecars_for_bare_relative_path() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = format!("warp-sidecar-test-{nonce}.sqlite");
        let db_path = PathBuf::from(&base);
        let sidecar = PathBuf::from(format!("{base}.buckets.1.test"));
        let old_sidecar = PathBuf::from(format!("{base}.residuals.1.test"));

        std::fs::write(&sidecar, b"bucket").unwrap();
        std::fs::write(&old_sidecar, b"residual").unwrap();
        DB::remove_bucket_data_sidecars(&db_path);

        assert!(!sidecar.exists());
        assert!(!old_sidecar.exists());
    }
}
