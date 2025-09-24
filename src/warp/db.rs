use iso8601_timestamp::Timestamp;
use rusqlite::{Connection, OpenFlags, Result as SQLResult, Statement};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const HASH_CHARS: usize = 32; // we'll use sha256 truncated at 128 bits/32 characters

pub struct DB {
    db_fn: String,
    connection: Connection,
    remove_on_shutdown: bool,
}

impl DB {
    pub fn new_reader(db_fn: &str) -> Self {
        let connection =
            Connection::open_with_flags(db_fn, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        Self {
            db_fn: db_fn.to_string(),
            connection: connection,
            remove_on_shutdown: false,
        }
    }

    pub fn new(db_fn: &str) -> Self {
        let mut connection = Connection::open(db_fn).unwrap();

        let status: SQLResult<String> =
            connection.query_row("PRAGMA quick_check;", [], |row| row.get(0));
        let db_ok = match status {
            Ok(text) => text.trim().eq_ignore_ascii_case("ok"),
            Err(_e) => false,
        };
        if db_ok == false {
            println!("warp database corrupted, recreating it!");
            drop(connection);
            let _ = std::fs::remove_file(&db_fn);
            connection = Connection::open(db_fn).unwrap();
        }

        //connection
        //.pragma_update(None, "journal_mode", &"WAL")
        //.unwrap();
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let query = format!(
            "CREATE TABLE IF NOT EXISTS document(uuid TEXT NOT NULL PRIMARY KEY,
            date TEXT NOT NULL,
            metadata JSON, hash TEXT
            CHECK (length(hash) = {HASH_CHARS}),
            body TEXT)"
        );
        connection.execute(&query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS document_uuid_index ON document(uuid)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS document_index ON document(hash)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(body, content='document', content_rowid='rowid')";
        connection.execute(query, ()).unwrap();

        let query = "INSERT INTO document_fts(document_fts) VALUES('rebuild')";
        connection.execute(query, ()).unwrap();

        let query = format!(
            "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY
            CHECK (length(hash) = {HASH_CHARS}),
            model TEXT,
            embeddings BLOB NOT NULL)"
        );
        connection.execute(&query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS chunk_index ON chunk(hash)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TRIGGER IF NOT EXISTS document_after_delete
            AFTER DELETE ON document
            BEGIN
              DELETE FROM chunk
              WHERE hash = OLD.hash
                AND NOT EXISTS (SELECT 1 FROM document WHERE hash = OLD.hash);
            END";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS bucket(id INTEGER PRIMARY KEY,
            generation INTEGER NOT NULL,
            center BLOB NOT NULL, indices BLOB NOT NULL, residuals BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS bucket_index ON bucket(generation, id)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS indexed_chunk(chunkid INTEGER PRIMARY KEY NOT NULL, generation INTEGER NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query =
            "CREATE UNIQUE INDEX IF NOT EXISTS indexed_chunk_index ON indexed_chunk(chunkid, generation)";
        connection.execute(query, ()).unwrap();
        Self {
            db_fn: db_fn.to_string(),
            connection: connection,
            remove_on_shutdown: false,
        }
    }

    pub fn clear(&mut self) {
        self.execute("DELETE FROM document").unwrap();
        self.execute("DELETE FROM chunk").unwrap();
        self.execute("DELETE FROM bucket").unwrap();
        self.execute("DELETE FROM indexed_chunk").unwrap();
        self.execute("VACUUM").unwrap();
        self.remove_on_shutdown = true;
    }

    pub fn shutdown(&mut self) {
        if self.remove_on_shutdown {
            let _ = std::fs::remove_file(&self.db_fn);
            self.remove_on_shutdown = false;
        }
    }

    pub fn execute(self: &Self, sql: &str) -> SQLResult<()> {
        self.connection.execute(sql, ()).unwrap();
        Ok(())
    }

    pub fn query(self: &Self, sql: &str) -> Statement<'_> {
        self.connection.prepare(&sql).unwrap()
    }

    pub fn begin_transaction(&self) {
        self.connection.execute("BEGIN", ()).unwrap();
    }

    pub fn commit_transaction(&self) {
        self.connection.execute("COMMIT", ()).unwrap();
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
            "INSERT OR IGNORE INTO document VALUES(?1, ?2, ?3, ?4, ?5)",
            (&uuid.to_string(), date.to_string(), metadata, &hash, &body),
        )?;
        self.remove_on_shutdown = false;
        Ok(())
    }

    pub fn remove_doc(self: &mut Self, uuid: &Uuid) -> SQLResult<()> {
        self.connection
            .execute("DELETE FROM document WHERE uuid = ?1)", (uuid.to_string(),))?;
        Ok(())
    }

    pub fn add_chunk(
        self: &Self,
        hash: &str,
        model: &str,
        embeddings: &Vec<u8>,
    ) {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO chunk VALUES(?1, ?2, ?3)",
                (&hash, &model, embeddings),
            )
            .unwrap();
    }

    pub fn add_bucket(
        self: &Self,
        id: u32,
        generation: u32,
        center: &Vec<u8>,
        indices: &Vec<u8>,
        residuals: &Vec<u8>,
    ) {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO bucket VALUES(?1, ?2, ?3, ?4, ?5)",
                (id, generation, center, indices, residuals),
            )
            .unwrap();
    }

    pub fn add_indexed_chunk(self: &Self, chunkid: u32, generation: u32) {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO indexed_chunk VALUES(?1, ?2)",
                (chunkid, generation),
            )
            .unwrap();
    }
}
