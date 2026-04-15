use super::layer::{Layer, LayerChain, LayerError, LayerStatus};
use super::types::SqlStatementInternal;
use iso8601_timestamp::Timestamp;
use log::{error, warn};
use rusqlite::{params, params_from_iter, Connection, OpenFlags, Result as SQLResult, Statement};
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
        const EXPECTED_VERSION: i32 = 6;

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

        let query = "CREATE INDEX IF NOT EXISTS document_uuid_index ON document(uuid)";
        connection.execute(query, ())?;

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

        let query = "CREATE INDEX IF NOT EXISTS chunk_index ON chunk(hash)";
        connection.execute(query, ())?;

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

        // Layer tree for overlay-aware retrieval
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS layer(
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id   INTEGER REFERENCES layer(id),
                name        TEXT NOT NULL,
                metadata    JSON,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'sealed', 'compacted'))
            );
            CREATE INDEX IF NOT EXISTS layer_parent_index ON layer(parent_id);
            CREATE INDEX IF NOT EXISTS layer_status_index ON layer(status);",
        )?;

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
        self.execute("DELETE FROM layer")?;
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
        let lens = match lens {
            Some(lens) => lens,
            None => [body.chars().count()].to_vec(),
        };

        let total: usize = lens.iter().copied().sum();
        if total != body.chars().count() {
            warn!("bad length: [{} vs {}]", total, body.chars().count());
        }

        let lens = lens
            .iter()
            .map(|len| len.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let mut hasher = Sha256::new();
        hasher.update(body);
        hasher.update(&lens);
        let hash = format!("{:x}", hasher.finalize());
        let hash = &hash[..HASH_CHARS];

        let date = date.unwrap_or_else(Timestamp::now_utc);

        self.conn().execute(
            "INSERT INTO document VALUES(?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(uuid) DO UPDATE SET
                date = ?2, metadata = ?3, hash = ?4, body = ?5, lens = ?6",
            (
                &uuid.to_string(),
                date.to_string(),
                metadata,
                &hash,
                &body,
                &lens,
            ),
        )?;
        self.remove_on_shutdown = false;
        Ok(())
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

// ---------------------------------------------------------------------------
// Layer CRUD
// ---------------------------------------------------------------------------
impl DB {
    fn read_layer(row: &rusqlite::Row) -> rusqlite::Result<Layer> {
        let status_str: String = row.get(5)?;
        let status: LayerStatus = match status_str.parse() {
            Ok(s) => s,
            Err(e) => {
                warn!("unknown layer status '{}': {}", status_str, e);
                LayerStatus::Active
            }
        };
        Ok(Layer {
            id: row.get(0)?,
            parent_id: row.get(1)?,
            name: row.get(2)?,
            metadata: row.get(3)?,
            created_at: row.get(4)?,
            status,
        })
    }

    /// Create a new layer. Returns the layer ID.
    pub fn create_layer(
        &self,
        parent_id: Option<i64>,
        name: &str,
        metadata: Option<&str>,
    ) -> Result<i64, LayerError> {
        let now = Timestamp::now_utc().to_string();
        self.conn().execute(
            "INSERT INTO layer(parent_id, name, metadata, created_at) VALUES(?1, ?2, ?3, ?4)",
            params![parent_id, name, metadata, now],
        )?;
        Ok(self.conn().last_insert_rowid())
    }

    /// Seal a layer (mark immutable). Advisory — not enforced at the
    /// SQL level, but the application should check before writing.
    pub fn seal_layer(&self, layer_id: i64) -> Result<(), LayerError> {
        let changed = self.conn().execute(
            "UPDATE layer SET status = 'sealed' WHERE id = ?1 AND status = 'active'",
            params![layer_id],
        )?;
        if changed == 0 {
            // Distinguish not-found from not-active
            return match self.get_layer(layer_id) {
                Ok(_) => Err(LayerError::NotActive(layer_id)),
                Err(_) => Err(LayerError::NotFound(layer_id)),
            };
        }
        Ok(())
    }

    /// Get a single layer by ID.
    pub fn get_layer(&self, layer_id: i64) -> Result<Layer, LayerError> {
        self.conn()
            .query_row(
                "SELECT id, parent_id, name, metadata, created_at, status FROM layer WHERE id = ?1",
                params![layer_id],
                Self::read_layer,
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => LayerError::NotFound(layer_id),
                other => LayerError::Sqlite(other),
            })
    }

    /// Walk parent pointers from `layer_id` to root.
    /// Returns a LayerChain with (layer_id, depth) pairs, depth 0 = the queried layer.
    pub fn layer_chain(&self, layer_id: i64) -> Result<LayerChain, LayerError> {
        let mut stmt = self.conn().prepare(
            "WITH RECURSIVE chain(id, parent_id, depth) AS (
                SELECT id, parent_id, 0 FROM layer WHERE id = ?1
                UNION ALL
                SELECT l.id, l.parent_id, c.depth + 1
                FROM chain c
                JOIN layer l ON l.id = c.parent_id
            )
            SELECT id, depth FROM chain ORDER BY depth",
        )?;
        let chain: Vec<(i64, u32)> = stmt
            .query_map(params![layer_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, u32>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        if chain.is_empty() {
            return Err(LayerError::NotFound(layer_id));
        }
        Ok(LayerChain { chain })
    }

    /// Direct children of a layer.
    pub fn layer_children(&self, layer_id: i64) -> Result<Vec<Layer>, LayerError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, parent_id, name, metadata, created_at, status
             FROM layer WHERE parent_id = ?1 ORDER BY id",
        )?;
        let layers = stmt
            .query_map(params![layer_id], Self::read_layer)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(layers)
    }

    /// All layers in the tree.
    pub fn layer_tree(&self) -> Result<Vec<Layer>, LayerError> {
        let mut stmt = self.conn().prepare(
            "SELECT id, parent_id, name, metadata, created_at, status FROM layer ORDER BY id",
        )?;
        let layers = stmt
            .query_map((), Self::read_layer)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(layers)
    }

    /// Delete a layer. Fails if the layer has children
    /// (caller must reparent or delete children first).
    pub fn delete_layer(&self, layer_id: i64) -> Result<(), LayerError> {
        let child_count: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM layer WHERE parent_id = ?1",
            params![layer_id],
            |row| row.get(0),
        )?;
        if child_count > 0 {
            return Err(LayerError::HasChildren(layer_id));
        }
        let changed = self
            .conn()
            .execute("DELETE FROM layer WHERE id = ?1", params![layer_id])?;
        if changed == 0 {
            return Err(LayerError::NotFound(layer_id));
        }
        Ok(())
    }

    /// Reparent a layer to a new parent. Private — only used by compact_layers
    /// to avoid exposing an API that could create cycles.
    fn reparent_layer(
        &self,
        layer_id: i64,
        new_parent_id: Option<i64>,
    ) -> Result<(), LayerError> {
        let changed = self.conn().execute(
            "UPDATE layer SET parent_id = ?1 WHERE id = ?2",
            params![new_parent_id, layer_id],
        )?;
        if changed == 0 {
            return Err(LayerError::NotFound(layer_id));
        }
        Ok(())
    }

    /// Compact: merge the chain from `layer_id` up to (but not including)
    /// `stop_at` into a single new layer. Reparent children of merged layers
    /// to the new layer. Mark compacted layers as `compacted`.
    /// Returns the new layer's ID.
    ///
    /// Chunk data migration is added in PR 2.
    pub fn compact_layers(
        &self,
        layer_id: i64,
        stop_at: Option<i64>,
    ) -> Result<i64, LayerError> {
        // Walk the chain from layer_id up to stop_at
        let full_chain = self.layer_chain(layer_id)?;

        // Validate that stop_at is actually in the chain
        if let Some(stop) = stop_at {
            if !full_chain.chain.iter().any(|&(id, _)| id == stop) {
                return Err(LayerError::NotInChain {
                    stop_at: stop,
                    layer_id,
                });
            }
        }

        let mut to_compact: Vec<i64> = Vec::new();
        for &(id, _depth) in &full_chain.chain {
            if Some(id) == stop_at {
                break;
            }
            to_compact.push(id);
        }
        if to_compact.is_empty() {
            // layer_id == stop_at: nothing to compact
            return Err(LayerError::NotInChain {
                stop_at: stop_at.unwrap(),
                layer_id,
            });
        }

        // The new layer's parent is either stop_at or the parent of the
        // last layer in the compaction chain (the next entry after it
        // in the depth-ordered chain).
        let new_parent = stop_at.or_else(|| {
            let last = *to_compact.last().unwrap();
            let last_pos = full_chain
                .chain
                .iter()
                .position(|&(id, _)| id == last)
                .unwrap();
            full_chain.chain.get(last_pos + 1).map(|&(id, _)| id)
        });

        self.begin_transaction()
            .map_err(|e| LayerError::Sqlite(e))?;

        let result = (|| -> Result<i64, LayerError> {
            // Create the compacted layer
            let leaf = self.get_layer(layer_id)?;
            let compacted_id = self.create_layer(
                new_parent,
                &format!("compacted:{}", leaf.name),
                None,
            )?;

            // Reparent children of all compacted layers to the new layer
            // (excluding the compacted layers themselves)
            for &id in &to_compact {
                let children = self.layer_children(id)?;
                for child in children {
                    if !to_compact.contains(&child.id) {
                        self.reparent_layer(child.id, Some(compacted_id))?;
                    }
                }
            }

            // Mark compacted layers
            for &id in &to_compact {
                self.conn().execute(
                    "UPDATE layer SET status = 'compacted' WHERE id = ?1",
                    params![id],
                )?;
            }

            Ok(compacted_id)
        })();

        match &result {
            Ok(_) => {
                self.commit_transaction()
                    .map_err(|e| LayerError::Sqlite(e))?;
            }
            Err(_) => {
                if let Err(e) = self.rollback_transaction() {
                    error!("rollback failed during compact_layers: {e}");
                }
            }
        }

        result
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
