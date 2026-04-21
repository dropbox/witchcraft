use super::types::SqlStatementInternal;
use iso8601_timestamp::Timestamp;
use log::{error, warn};
use rusqlite::{params_from_iter, Connection, OpenFlags, Result as SQLResult, Statement};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use uuid::Uuid;

use super::sql_generator::build_filter_sql_and_params;

const HASH_CHARS: usize = 32; // we'll use sha256 truncated at 128 bits/32 characters

pub struct DB {
    db_fn: PathBuf,
    connection: Option<Connection>,
    remove_on_shutdown: bool,
}

impl DB {
    fn conn(&self) -> &Connection {
        self.connection.as_ref().expect("Connection should exist")
    }

    pub fn new_reader(db_fn: PathBuf) -> SQLResult<Self> {
        let connection =
            Connection::open_with_flags(db_fn.clone(), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Self {
            db_fn,
            connection: Some(connection),
            remove_on_shutdown: false,
        })
    }

    pub fn new(db_fn: PathBuf) -> SQLResult<Self> {
        const APP_ID: i32 = 0x07DB_DA55;
        const EXPECTED_VERSION: i32 = 7;

        let mut first_creation = !db_fn.exists();
        let connection = Connection::open(&db_fn)?;

        let status: SQLResult<String> =
            connection.query_row("PRAGMA quick_check;", [], |row| row.get(0));
        let db_ok = match status {
            Ok(text) => text.trim().eq_ignore_ascii_case("ok"),
            Err(_e) => false,
        };

        let schema_ok = if first_creation {
            true
        } else {
            let app_id: SQLResult<i32> =
                connection.query_row("PRAGMA application_id;", [], |r| r.get(0));
            let user_version: SQLResult<i32> =
                connection.query_row("PRAGMA user_version;", [], |r| r.get(0));
            matches!((app_id, user_version),
                (Ok(a), Ok(v)) if a == APP_ID && v == EXPECTED_VERSION && a != 0 && v != 0
            )
        };

        let connection = if db_ok && schema_ok {
            connection
        } else {
            warn!(
                "warp database {} corrupted or schema mismatch, recreating it!",
                db_fn.display()
            );
            connection.close().map_err(|(_conn, e)| e)?;
            std::fs::remove_file(&db_fn)
                .map_err(|_e| rusqlite::Error::InvalidPath(db_fn.clone()))?;
            let _ = std::fs::remove_file(db_fn.with_extension("wal"));
            let _ = std::fs::remove_file(db_fn.with_extension("shm"));
            first_creation = true;

            Connection::open(&db_fn)?
        };

        if first_creation {
            connection.execute_batch(&format!(
                "PRAGMA application_id = {APP_ID}; PRAGMA user_version = {EXPECTED_VERSION}"
            ))?;
        }

        // Enable WAL mode for better concurrency and performance
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;

        let query = format!(
            "CREATE TABLE IF NOT EXISTS document(uuid TEXT NOT NULL PRIMARY KEY,
            date TEXT NOT NULL,
            metadata JSON, hash TEXT
            CHECK (length(hash) = {HASH_CHARS}),
            body TEXT,
            lens TEXT)"
        );
        connection.execute(&query, ())?;

        let query = "CREATE INDEX IF NOT EXISTS document_index ON document(hash)";
        connection.execute(query, ())?;

        let query = "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(body, content='document', content_rowid='rowid')";
        connection.execute(query, ())?;

        let query = "INSERT INTO document_fts(document_fts) VALUES('rebuild')";
        connection.execute(query, ())?;

        let query = format!(
            "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY
            CHECK (length(hash) = {HASH_CHARS}),
            model TEXT,
            embeddings BLOB NOT NULL,
            counts TEXT NOT NULL)"
        );
        connection.execute(&query, ())?;


        let query = "CREATE TRIGGER IF NOT EXISTS document_after_delete
            AFTER DELETE ON document
            BEGIN
              DELETE FROM chunk
              WHERE hash = OLD.hash
                AND NOT EXISTS (SELECT 1 FROM document WHERE hash = OLD.hash);
            END";
        connection.execute(query, ())?;

        let query = "CREATE TRIGGER IF NOT EXISTS document_after_update
            AFTER UPDATE ON document
            BEGIN
              DELETE FROM chunk
              WHERE hash = OLD.hash
                AND NOT EXISTS (SELECT 1 FROM document WHERE hash = OLD.hash);
            END";
        connection.execute(query, ())?;

        let query = "CREATE TABLE IF NOT EXISTS bucket(id INTEGER PRIMARY KEY,
            center BLOB NOT NULL, indices BLOB NOT NULL, residuals BLOB NOT NULL)";
        connection.execute(query, ())?;

        let query =
            "CREATE TABLE IF NOT EXISTS indexed_chunk(chunkid INTEGER PRIMARY KEY NOT NULL)";
        connection.execute(query, ())?;
        Ok(Self {
            db_fn,
            connection: Some(connection),
            remove_on_shutdown: false,
        })
    }

    fn clear_inner(&mut self) -> SQLResult<()> {
        self.execute("DELETE FROM document")?;
        self.execute("DELETE FROM chunk")?;
        self.execute("DELETE FROM bucket")?;
        self.execute("DELETE FROM indexed_chunk")?;
        self.execute("VACUUM")?;
        Ok(())
    }

    pub fn refresh_ft(&mut self) -> SQLResult<()> {
        self.execute("INSERT INTO document_fts(document_fts) VALUES('rebuild')")?;
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
            Self::checkpoint_internal(&connection, false);
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
            let _ = std::fs::remove_file(self.db_fn.with_extension("wal"));
            let _ = std::fs::remove_file(self.db_fn.with_extension("shm"));
        }
    }

    /// Checkpoint and truncate the WAL into the main database file.
    /// Safe to call at any point when no statements are active on this connection.
    pub fn checkpoint(&self) {
        if let Some(connection) = self.connection.as_ref() {
            Self::checkpoint_internal(connection, true);
        }
    }

    pub fn file_size(&self) -> std::io::Result<u64> {
        std::fs::metadata(&self.db_fn).map(|meta| meta.len())
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
            let hash = format!("{:x}", hasher.finalize());
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
    ) -> SQLResult<()> {
        self.conn().execute(
            "INSERT OR IGNORE INTO chunk VALUES(?1, ?2, ?3, ?4)",
            (&hash, &model, embeddings, counts),
        )?;
        Ok(())
    }

    pub fn add_bucket(
        &self,
        id: u32,
        center: &Vec<u8>,
        indices: &Vec<u8>,
        residuals: &Vec<u8>,
    ) -> SQLResult<()> {
        self.conn().execute(
            "INSERT OR REPLACE INTO bucket VALUES(?1, ?2, ?3, ?4)",
            (id, center, indices, residuals),
        )?;
        Ok(())
    }

    pub fn add_indexed_chunk(&self, chunkid: u32) -> SQLResult<()> {
        self.conn().execute(
            "INSERT OR REPLACE INTO indexed_chunk VALUES(?1)",
            (chunkid,),
        )?;
        Ok(())
    }
}

impl Drop for DB {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take() {
            Self::checkpoint_internal(&connection, false);
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
