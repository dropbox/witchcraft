use super::types::SqlStatementInternal;
use super::app_id::APP_ID;
use super::document_cache_hash;
use iso8601_timestamp::Timestamp;
use log::{error, warn};
use rusqlite::{
    params_from_iter, Connection, OpenFlags, OptionalExtension, Result as SQLResult, Statement,
};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use super::sql_generator::build_filter_sql_and_params;

const SCHEMA_VERSION: i32 = 13;
const HASH_CHARS: usize = 32;
const MAX_SQLITE_ROWID: u64 = i64::MAX as u64;

fn invalid_input_error(message: String) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message,
    )))
}

fn sqlite_rowid_from_u64(rowid: u64) -> SQLResult<i64> {
    if rowid == 0 || rowid > MAX_SQLITE_ROWID {
        return Err(invalid_input_error(format!(
            "document rowid {rowid} must be between 1 and {MAX_SQLITE_ROWID}"
        )));
    }
    Ok(rowid as i64)
}

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

    fn rowids_prefix(db_fn: &Path) -> String {
        let base = db_fn
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "warp.sqlite".into());
        format!("{base}.rowids.")
    }

    fn rowids_buffer_file(db_fn: &Path) -> String {
        let base = db_fn
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "warp.sqlite".into());
        format!("{base}.rowids.buffer")
    }

    fn index_manifest_file(db_fn: &Path) -> String {
        let base = db_fn
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "warp.sqlite".into());
        format!("{base}.index")
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
        let base = db_fn
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "warp.sqlite".into());
        let prefix = Self::bucket_data_prefix(db_fn);
        let rowids_prefix = Self::rowids_prefix(db_fn);
        let rowids_buffer = Self::rowids_buffer_file(db_fn);
        let manifest = Self::index_manifest_file(db_fn);
        let pruning_prefix = format!("{base}.gatebpef");
        let old_prefix = Self::old_residuals_prefix(db_fn);
        let Ok(entries) = std::fs::read_dir(parent) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix)
                || name.starts_with(&rowids_prefix)
                || name == rowids_buffer
                || name == manifest
                || Self::is_pruning_index_sidecar(&name, &pruning_prefix)
                || name.starts_with(&old_prefix)
            {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    fn is_pruning_index_sidecar(name: &str, pruning_prefix: &str) -> bool {
        name.starts_with(pruning_prefix)
            && (name.contains(".buckets.")
                || name.contains(".rowids.")
                || name.ends_with(".index")
                || name.contains(".residuals."))
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
                 hash TEXT CHECK (hash IS NULL OR length(hash) = {HASH_CHARS}),
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
                 model TEXT NOT NULL,
                 embeddings BLOB NOT NULL,
                 counts TEXT NOT NULL,
                 embedding_count INTEGER NOT NULL,
                 metadata TEXT);

             CREATE TRIGGER document_after_delete AFTER DELETE ON document
             BEGIN
                 DELETE FROM chunk
                     WHERE hash = old.hash
                     AND NOT EXISTS (SELECT 1 FROM document WHERE hash = old.hash);
             END;

             CREATE TRIGGER document_after_update AFTER UPDATE ON document
             BEGIN
                 DELETE FROM chunk
                     WHERE hash = old.hash
                     AND NOT EXISTS (SELECT 1 FROM document WHERE hash = old.hash);
             END;

             "
        ))?;
        Ok(())
    }

    fn has_column(connection: &Connection, table: &str, column: &str) -> SQLResult<bool> {
        let mut query = connection.prepare(&format!("PRAGMA table_info({table})"))?;
        let columns = query.query_map((), |row| row.get::<_, String>(1))?;
        for result in columns {
            if result? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn backfill_document_hashes(connection: &Connection) -> SQLResult<()> {
        let rows = {
            let mut query = connection.prepare(
                "SELECT rowid, IFNULL(body, ''), IFNULL(lens, '')
                 FROM document
                 WHERE hash IS NULL OR length(hash) != ?1",
            )?;
            let rows = query
                .query_map((HASH_CHARS as i64,), |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .collect::<SQLResult<Vec<_>>>()?;
            rows
        };

        let mut update = connection.prepare("UPDATE document SET hash = ?1 WHERE rowid = ?2")?;
        for (rowid, body, lens) in rows {
            let hash = document_cache_hash(&body, &lens);
            update.execute((&hash, rowid))?;
        }
        Ok(())
    }

    fn backfill_chunk_embedding_counts(connection: &Connection) -> SQLResult<()> {
        let rows = {
            let mut query = connection.prepare(
                "SELECT hash, counts FROM chunk WHERE embedding_count = 0",
            )?;
            let rows = query
                .query_map((), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<SQLResult<Vec<_>>>()?;
            rows
        };

        let mut update = connection.prepare("UPDATE chunk SET embedding_count = ?1 WHERE hash = ?2")?;
        for (hash, counts) in rows {
            let embedding_count: usize = counts
                .split(',')
                .filter_map(|count| count.parse::<usize>().ok())
                .sum();
            update.execute((embedding_count as i64, hash))?;
        }
        Ok(())
    }

    fn ensure_chunk_schema(connection: &Connection) -> SQLResult<()> {
        if !Self::has_column(connection, "document", "hash")? {
            connection.execute_batch(&format!(
                "ALTER TABLE document
                 ADD COLUMN hash TEXT CHECK (hash IS NULL OR length(hash) = {HASH_CHARS});",
            ))?;
        }
        Self::backfill_document_hashes(connection)?;

        connection.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS document_index ON document(hash);

             CREATE TABLE IF NOT EXISTS chunk(
                 hash TEXT PRIMARY KEY CHECK (length(hash) = {HASH_CHARS}),
                 model TEXT NOT NULL,
                 embeddings BLOB NOT NULL,
                 counts TEXT NOT NULL,
                 embedding_count INTEGER NOT NULL DEFAULT 0,
                 metadata TEXT);

             CREATE TRIGGER IF NOT EXISTS document_after_delete AFTER DELETE ON document
             BEGIN
                 DELETE FROM chunk
                     WHERE hash = old.hash
                     AND NOT EXISTS (SELECT 1 FROM document WHERE hash = old.hash);
             END;

             CREATE TRIGGER IF NOT EXISTS document_after_update AFTER UPDATE ON document
             BEGIN
                 DELETE FROM chunk
                     WHERE hash = old.hash
                     AND NOT EXISTS (SELECT 1 FROM document WHERE hash = old.hash);
             END;",
        ))?;

        if !Self::has_column(connection, "chunk", "embedding_count")? {
            connection.execute_batch(
                "ALTER TABLE chunk
                 ADD COLUMN embedding_count INTEGER NOT NULL DEFAULT 0;",
            )?;
        }
        if !Self::has_column(connection, "chunk", "metadata")? {
            connection.execute_batch(
                "ALTER TABLE chunk
                 ADD COLUMN metadata TEXT;",
            )?;
        }
        Self::backfill_chunk_embedding_counts(connection)?;
        Ok(())
    }

    fn ensure_document_index_tombstone_schema(connection: &Connection) -> SQLResult<()> {
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS document_index_tombstone(
                 rowid INTEGER NOT NULL PRIMARY KEY
             );

             CREATE TRIGGER IF NOT EXISTS document_index_tombstone_delete
             AFTER DELETE ON document
             BEGIN
                 INSERT OR IGNORE INTO document_index_tombstone(rowid)
                     VALUES(old.rowid);
             END;

             CREATE TRIGGER IF NOT EXISTS document_index_tombstone_rowid_update
             AFTER UPDATE ON document
             WHEN old.rowid != new.rowid
             BEGIN
                 INSERT OR IGNORE INTO document_index_tombstone(rowid)
                     VALUES(old.rowid);
             END;",
        )?;
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
        Self::ensure_chunk_schema(&connection)?;
        Self::ensure_document_index_tombstone_schema(&connection)?;

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
        log::info!("new fast!");
        Self::open_internal(db_fn, true)
    }

    fn clear_inner(&mut self) -> SQLResult<()> {
        self.execute("DELETE FROM document")?;
        self.execute("DELETE FROM document_index_tombstone")?;
        self.execute("DELETE FROM chunk")?;
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
        rowid: Option<u64>,
        uuid: &Uuid,
        date: Option<Timestamp>,
        metadata: &str,
        body: &str,
        lens: Option<Vec<usize>>,
    ) -> SQLResult<()> {
        self.add_docs_batch(&[(rowid, *uuid, date, metadata, body, lens)])?;
        Ok(())
    }

    /// Batch-add documents in a single transaction. Prepares the statement once
    /// and reuses it for all inserts. Much faster than individual add_doc calls.
    pub fn add_docs_batch(
        &mut self,
        docs: &[(Option<u64>, Uuid, Option<Timestamp>, &str, &str, Option<Vec<usize>>)],
    ) -> SQLResult<usize> {
        if docs.is_empty() {
            return Ok(0);
        }
        let rowids = self.resolve_document_rowids(docs)?;
        self.conn().execute("BEGIN", ())?;
        let mut stmt = self.conn().prepare(
            "INSERT INTO document(rowid, uuid, date, metadata, hash, body, lens)
            VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(uuid) DO UPDATE SET
                rowid = ?1, date = ?3, metadata = ?4, hash = ?5, body = ?6, lens = ?7",
        )?;

        let mut count = 0;
        for ((_, uuid, date, metadata, body, lens), rowid) in docs.iter().zip(rowids.iter()) {
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
            let hash = document_cache_hash(body, &lens_str);

            let date = date.unwrap_or_else(Timestamp::now_utc);
            stmt.execute((
                rowid,
                &uuid.to_string(),
                date.to_string(),
                *metadata,
                &hash,
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

    fn resolve_document_rowids(
        &self,
        docs: &[(Option<u64>, Uuid, Option<Timestamp>, &str, &str, Option<Vec<usize>>)],
    ) -> SQLResult<Vec<i64>> {
        let mut previous = self.max_known_document_rowid()?;
        let mut rowid_query = self
            .conn()
            .prepare("SELECT rowid FROM document WHERE uuid = ?1")?;
        let mut rowids = Vec::with_capacity(docs.len());

        for (rowid, uuid, _, _, _, _) in docs {
            let rowid = match rowid {
                Some(rowid) => {
                    let rowid = sqlite_rowid_from_u64(*rowid)?;
                    if rowid <= previous {
                        return Err(invalid_input_error(format!(
                            "document rowid {rowid} must be greater than previous rowid {previous}"
                        )));
                    }
                    previous = rowid;
                    rowid
                }
                None => match rowid_query
                    .query_row((uuid.to_string(),), |row| row.get::<_, i64>(0))
                    .optional()?
                {
                    Some(existing) => existing,
                    None => {
                        previous = previous.checked_add(1).ok_or_else(|| {
                            invalid_input_error("document rowid space exhausted".to_string())
                        })?;
                        previous
                    }
                },
            };
            rowids.push(rowid);
        }

        Ok(rowids)
    }

    fn max_known_document_rowid(&self) -> SQLResult<i64> {
        self.conn()
            .query_row("SELECT IFNULL(MAX(rowid), 0) FROM document", (), |row| {
                row.get(0)
            })
    }

    pub fn remove_doc(&mut self, uuid: &Uuid) -> SQLResult<()> {
        self.conn()
            .execute("DELETE FROM document WHERE uuid = ?1", (uuid.to_string(),))?;
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
        let rowids = PathBuf::from(format!("{base}.rowids.1.test"));
        let rowids_buffer = PathBuf::from(format!("{base}.rowids.buffer"));
        let manifest = PathBuf::from(format!("{base}.index"));
        let old_sidecar = PathBuf::from(format!("{base}.residuals.1.test"));
        let pruning_sidecar = PathBuf::from(format!("{base}.gatebpef2500.buckets.1.test"));
        let pruning_rowids = PathBuf::from(format!("{base}.gatebpef2500.rowids.1.test"));
        let pruning_buffer = PathBuf::from(format!("{base}.gatebpef2500.rowids.buffer"));
        let pruning_manifest = PathBuf::from(format!("{base}.gatebpef2500.index"));
        let pruning_residuals = PathBuf::from(format!("{base}.gatebpef2500.residuals.1.test"));

        std::fs::write(&sidecar, b"bucket").unwrap();
        std::fs::write(&rowids, b"rowids").unwrap();
        std::fs::write(&rowids_buffer, b"buffer").unwrap();
        std::fs::write(&manifest, b"manifest").unwrap();
        std::fs::write(&old_sidecar, b"residual").unwrap();
        std::fs::write(&pruning_sidecar, b"bucket").unwrap();
        std::fs::write(&pruning_rowids, b"rowids").unwrap();
        std::fs::write(&pruning_buffer, b"buffer").unwrap();
        std::fs::write(&pruning_manifest, b"manifest").unwrap();
        std::fs::write(&pruning_residuals, b"residual").unwrap();
        DB::remove_bucket_data_sidecars(&db_path);

        assert!(!sidecar.exists());
        assert!(!rowids.exists());
        assert!(!rowids_buffer.exists());
        assert!(!manifest.exists());
        assert!(!old_sidecar.exists());
        assert!(!pruning_sidecar.exists());
        assert!(!pruning_rowids.exists());
        assert!(!pruning_buffer.exists());
        assert!(!pruning_manifest.exists());
        assert!(!pruning_residuals.exists());
    }
}
