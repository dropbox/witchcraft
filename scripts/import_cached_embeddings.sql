-- Import cached embedding blobs from another Warp SQLite database into the
-- currently-open database.
--
-- Usage:
--   sqlite3 mydb.sqlite \
--     ".parameter set @source '/path/to/existing.sqlite'" \
--     ".read scripts/import_cached_embeddings.sql"
--
-- Run this after mydb.sqlite has been created by the latest binary and after
-- its document rows have been loaded, so document.hash is populated. The import
-- only copies cache rows whose hash is referenced by a document in mydb.sqlite.
--
-- The source database may be older and lack chunk.embedding_count; this script
-- recomputes embedding_count from chunk.counts.

.bail on
.timeout 5000

ATTACH DATABASE @source AS source_cache;

BEGIN IMMEDIATE;

CREATE INDEX IF NOT EXISTS main.document_nonempty_hash_index
    ON document(hash)
    WHERE length(body) > 0;

CREATE INDEX IF NOT EXISTS main.chunk_hash_model_embedding_count_index
    ON chunk(hash, model, embedding_count);

WITH RECURSIVE
source_chunks(hash, model, embeddings, counts) AS (
    SELECT c.hash, c.model, c.embeddings, c.counts
    FROM source_cache.chunk AS c
    WHERE c.hash IS NOT NULL
      AND length(c.hash) = 32
      AND EXISTS (
          SELECT 1
          FROM main.document AS d
          WHERE d.hash = c.hash
      )
),
count_parts(hash, model, embeddings, counts, rest, embedding_count) AS (
    SELECT hash, model, embeddings, counts, counts || ',', 0
    FROM source_chunks

    UNION ALL

    SELECT hash,
           model,
           embeddings,
           counts,
           substr(rest, instr(rest, ',') + 1),
           embedding_count + COALESCE(
               CAST(NULLIF(trim(substr(rest, 1, instr(rest, ',') - 1)), '') AS INTEGER),
               0
           )
    FROM count_parts
    WHERE rest <> ''
      AND instr(rest, ',') > 0
),
computed_chunks AS (
    SELECT hash,
           model,
           embeddings,
           counts,
           MAX(embedding_count) AS embedding_count
    FROM count_parts
    GROUP BY hash
)
INSERT INTO main.chunk(hash, model, embeddings, counts, embedding_count)
SELECT hash, model, embeddings, counts, embedding_count
FROM computed_chunks
WHERE 1
ON CONFLICT(hash) DO UPDATE SET
    model = excluded.model,
    embeddings = excluded.embeddings,
    counts = excluded.counts,
    embedding_count = excluded.embedding_count;

COMMIT;

DETACH DATABASE source_cache;

PRAGMA optimize;
