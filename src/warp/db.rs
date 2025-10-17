use iso8601_timestamp::Timestamp;
use log::warn;
use rusqlite::{Connection, OpenFlags, Result as SQLResult, Statement};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use std::path::PathBuf;

const HASH_CHARS: usize = 32; // we'll use sha256 truncated at 128 bits/32 characters

pub struct DB {
    db_fn: PathBuf,
    connection: Connection,
    remove_on_shutdown: bool,
}

impl DB {
    pub fn new_reader(db_fn: PathBuf) -> SQLResult<Self> {
        let connection = Connection::open_with_flags(db_fn.clone(), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Self {
            db_fn: db_fn,
            connection: connection,
            remove_on_shutdown: false,
        })
    }


    pub fn new(db_fn: PathBuf) -> SQLResult<Self> {
        const APP_ID: i32 = 0x07DB_DA55;
        const EXPECTED_VERSION: i32 = 1;

        let connection = Connection::open(&db_fn)?;

        let status: SQLResult<String> =
            connection.query_row("PRAGMA quick_check;", [], |row| row.get(0));
        let db_ok = match status {
            Ok(text) => text.trim().eq_ignore_ascii_case("ok"),
            Err(_e) => false,
        };

        // check app_id and user_version
        let app_id: SQLResult<i32> = connection.query_row("PRAGMA application_id;", [], |r| r.get(0));
        let user_version: SQLResult<i32> =
            connection.query_row("PRAGMA user_version;", [], |r| r.get(0));

        let schema_ok = matches!((app_id, user_version),
            (Ok(a), Ok(v)) if a == APP_ID && v == EXPECTED_VERSION && a != 0 && v != 0
        );

        let connection = if db_ok && schema_ok {
            connection
        } else {
            warn!("warp database corrupted or schema mismatch, recreating it!");
            connection.close().map_err(|(_conn, e)| e)?;
            std::fs::remove_file(&db_fn)
                .map_err(|_e| rusqlite::Error::InvalidPath(db_fn.clone()))?;
            let _ = std::fs::remove_file(db_fn.with_extension("wal"));
            let _ = std::fs::remove_file(db_fn.with_extension("shm"));
            let connection = Connection::open(&db_fn)?;
            connection.execute_batch(&format!("PRAGMA application_id = {APP_ID}; PRAGMA user_version = {EXPECTED_VERSION}"))?;
            connection
        };

        //connection
        //.pragma_update(None, "journal_mode", &"WAL")
        //?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;

        let query = format!(
            "CREATE TABLE IF NOT EXISTS document(uuid TEXT NOT NULL PRIMARY KEY,
            date TEXT NOT NULL,
            metadata JSON, hash TEXT
            CHECK (length(hash) = {HASH_CHARS}),
            body TEXT)"
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
            embeddings BLOB NOT NULL)"
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
            generation INTEGER NOT NULL,
            center BLOB NOT NULL, indices BLOB NOT NULL, residuals BLOB NOT NULL)";
        connection.execute(query, ())?;

        let query = "CREATE INDEX IF NOT EXISTS bucket_index ON bucket(generation, id)";
        connection.execute(query, ())?;

        let query = "CREATE TABLE IF NOT EXISTS indexed_chunk(chunkid INTEGER PRIMARY KEY NOT NULL, generation INTEGER NOT NULL)";
        connection.execute(query, ())?;

        let query =
            "CREATE UNIQUE INDEX IF NOT EXISTS indexed_chunk_index ON indexed_chunk(chunkid, generation)";
        connection.execute(query, ())?;
        Ok(Self {
            db_fn: db_fn,
            connection: connection,
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

    pub fn shutdown(&mut self) {
        if self.remove_on_shutdown {
            match std::fs::remove_file(&self.db_fn) {
                Ok(()) => {
                    self.remove_on_shutdown = false;
                }
                Err(v) => {
                    warn!("unable to remove database file {v}");
                }
            };
        }
    }

    pub fn execute(self: &Self, sql: &str) -> SQLResult<()> {
        self.connection.execute(sql, ()).unwrap();
        Ok(())
    }

    pub fn query(self: &Self, sql: &str) -> SQLResult<Statement<'_>> {
        self.connection.prepare(&sql)
    }

    pub fn begin_transaction(&self) -> SQLResult<()> {
        self.connection.execute("BEGIN", ())?;
        Ok(())
    }

    pub fn commit_transaction(&self) -> SQLResult<()> {
        self.connection.execute("COMMIT", ())?;
        Ok(())
    }

    pub fn rollback_transaction(&self) -> SQLResult<()> {
        self.connection.execute("ROLLBACK", ())?;
        Ok(())
    }

    pub fn add_doc(
        self: &mut Self,
        uuid: &Uuid,
        date: Option<Timestamp>,
        metadata: &str,
        body: &str,
    ) -> SQLResult<()> {
        let mut hasher = Sha256::new();
        hasher.update(&body);
        let hash = format!("{:x}", hasher.finalize());
        let hash = &hash[..HASH_CHARS];
        let date = date.unwrap_or_else(Timestamp::now_utc);

        self.connection.execute(
            "INSERT INTO document VALUES(?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(uuid) DO UPDATE SET
                date = ?2, metadata = ?3, hash = ?4, body = ?5",
            (&uuid.to_string(), date.to_string(), metadata, &hash, &body),
        )?;
        self.remove_on_shutdown = false;
        Ok(())
    }

    pub fn remove_doc(self: &mut Self, uuid: &Uuid) -> SQLResult<()> {
        self.connection
            .execute("DELETE FROM document WHERE uuid = ?1", (uuid.to_string(),))?;
        Ok(())
    }

    pub fn add_chunk(self: &Self, hash: &str, model: &str, embeddings: &Vec<u8>) -> SQLResult<()> {
        self.connection.execute(
            "INSERT OR IGNORE INTO chunk VALUES(?1, ?2, ?3)",
            (&hash, &model, embeddings),
        )?;
        Ok(())
    }

    pub fn add_bucket(
        self: &Self,
        id: u32,
        generation: u32,
        center: &Vec<u8>,
        indices: &Vec<u8>,
        residuals: &Vec<u8>,
    ) -> SQLResult<()> {
        self.connection.execute(
            "INSERT OR REPLACE INTO bucket VALUES(?1, ?2, ?3, ?4, ?5)",
            (id, generation, center, indices, residuals),
        )?;
        Ok(())
    }

    pub fn add_indexed_chunk(self: &Self, chunkid: u32, generation: u32) -> SQLResult<()> {
        self.connection.execute(
            "INSERT OR REPLACE INTO indexed_chunk VALUES(?1, ?2)",
            (chunkid, generation),
        )?;
        Ok(())
    }
}
