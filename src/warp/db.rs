use rusqlite::{Connection, OpenFlags, Result as SQLResult, Statement};
use sha2::{Digest, Sha256};

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
        let connection = Connection::open(db_fn).unwrap();

        //connection
        //.pragma_update(None, "journal_mode", &"WAL")
        //.unwrap();
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .unwrap();

        let query = format!(
            "CREATE TABLE IF NOT EXISTS document(metadata JSON, hash TEXT
            CHECK (length(hash) = {HASH_CHARS}),
            body TEXT, UNIQUE(metadata, hash))"
        );
        connection.execute(&query, ()).unwrap();

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

    pub fn begin_transaction(&self) -> SQLResult<()> {
        self.connection.execute("BEGIN", ()).unwrap();
        Ok(())
    }

    pub fn commit_transaction(&self) -> SQLResult<()> {
        self.connection.execute("COMMIT", ()).unwrap();
        Ok(())
    }

    pub fn add_doc(self: &mut Self, metadata: &str, body: &str) -> SQLResult<()> {
        let mut hasher = Sha256::new();
        hasher.update(&body);
        let hash = format!("{:x}", hasher.finalize());
        let hash = &hash[..HASH_CHARS];
        self.connection.execute(
            "INSERT OR IGNORE INTO document VALUES(?1, ?2, ?3)",
            (&metadata, &hash, &body),
        )?;
        self.remove_on_shutdown = false;
        Ok(())
    }

    pub fn add_chunk(self: &Self, hash: &str, model: &str, embeddings: &Vec<u8>) -> SQLResult<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO chunk VALUES(?1, ?2, ?3)",
                (&hash, &model, embeddings),
            )
            .unwrap();
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
        self.connection
            .execute(
                "INSERT OR REPLACE INTO bucket VALUES(?1, ?2, ?3, ?4, ?5)",
                (id, generation, center, indices, residuals),
            )
            .unwrap();
        Ok(())
    }

    pub fn add_indexed_chunk(self: &Self, chunkid: u32, generation: u32) -> SQLResult<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO indexed_chunk VALUES(?1, ?2)",
                (chunkid, generation),
            )
            .unwrap();
        Ok(())
    }
}
