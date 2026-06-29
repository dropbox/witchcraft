use crate::file_index::{sort_dedup_rowid_records, FileBackedIndex, RowidRecord};
use crate::progress_reporter::ProgressReporter;
use crate::sql_generator::build_filter_sql_and_params;
use crate::{
    cached_embeddings_for_index, clear_generations_cache, document_cache_hash,
    hybrid_reciprocal_rank_fusion, index_buffered_embeddings_with_options, load_cached_embeddings,
    load_or_compute_cached_embeddings, match_centroids_raw,
    split_by_codepoints, CachedEmbeddings, DB, DocPtr, Embedder, EmbeddingCache, EmbeddingsCache,
    IndexOptions, SqlStatementInternal,
};
use anyhow::Result;
use candle_core::Tensor;
use log::{debug, info};
use rusqlite::OptionalExtension;
use std::collections::HashMap;

struct SqliteEmbeddingCache<'a> {
    db: &'a DB,
}

impl<'a> SqliteEmbeddingCache<'a> {
    fn new(db: &'a DB) -> Self {
        Self { db }
    }
}

impl EmbeddingCache for SqliteEmbeddingCache<'_> {
    fn get(&self, hash: &str) -> Result<Option<CachedEmbeddings>> {
        let mut query = self.db.query(
            "SELECT model, counts, embedding_count, embeddings, metadata
             FROM chunk
             WHERE hash = ?1",
        )?;
        let cached = query
            .query_row((hash,), |row| {
                Ok(CachedEmbeddings {
                    model: row.get(0)?,
                    counts: row.get(1)?,
                    embedding_count: row.get::<_, i64>(2)?.try_into().map_err(|err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Integer,
                            Box::new(err),
                        )
                    })?,
                    embeddings: row.get(3)?,
                    metadata: cached_metadata_from_sql(row.get(4)?, 4)?,
                })
            })
            .optional()?;
        Ok(cached)
    }

    fn get_for_document(&self, rowid: u64, hash: &str) -> Result<Option<CachedEmbeddings>> {
        if !hash.is_empty() {
            return self.get(hash);
        }

        let rowid: i64 = rowid.try_into()?;
        let mut query = self.db.query(
            "SELECT chunk.model, chunk.counts, chunk.embedding_count, chunk.embeddings, chunk.metadata
             FROM document, chunk
             WHERE document.hash = chunk.hash
             AND document.rowid = ?1",
        )?;
        let cached = query
            .query_row((rowid,), |row| {
                Ok(CachedEmbeddings {
                    model: row.get(0)?,
                    counts: row.get(1)?,
                    embedding_count: row.get::<_, i64>(2)?.try_into().map_err(|err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Integer,
                            Box::new(err),
                        )
                    })?,
                    embeddings: row.get(3)?,
                    metadata: cached_metadata_from_sql(row.get(4)?, 4)?,
                })
            })
            .optional()?;
        Ok(cached)
    }

    fn put(&self, hash: &str, embeddings: &CachedEmbeddings) -> Result<()> {
        let metadata = embeddings
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let mut statement = self.db.query(
            "INSERT INTO chunk(hash, model, embeddings, counts, embedding_count, metadata)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(hash) DO UPDATE SET
                 model = excluded.model,
                 embeddings = excluded.embeddings,
                 counts = excluded.counts,
                 embedding_count = excluded.embedding_count,
                 metadata = excluded.metadata",
        )?;
        statement.execute((
            hash,
            &embeddings.model,
            &embeddings.embeddings,
            &embeddings.counts,
            i64::try_from(embeddings.embedding_count)?,
            metadata,
        ))?;
        Ok(())
    }
}

fn cached_metadata_from_sql(
    value: Option<String>,
    column: usize,
) -> rusqlite::Result<Option<crate::CachedEmbeddingMetadata>> {
    value
        .map(|value| {
            serde_json::from_str(&value).map_err(|err| {
                rusqlite::Error::FromSqlConversionFailure(
                    column,
                    rusqlite::types::Type::Text,
                    Box::new(err),
                )
            })
        })
        .transpose()
}

/// DB-backed wrapper: searches only a query-ready semantic index.
pub fn match_centroids(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&crate::SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    let cache = SqliteEmbeddingCache::new(db);
    match_centroids_from_cache(db, query_embeddings, threshold, top_k, sql_filter, None, &cache)
}

/// Cache-backed wrapper for callers with an external embedding cache.
pub fn match_centroids_with_cache(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&crate::SqlStatementInternal>,
    embedder: &Embedder,
    cache: &dyn EmbeddingCache,
) -> Result<Vec<(f32, u32, u32)>> {
    match_centroids_from_cache(
        db,
        query_embeddings,
        threshold,
        top_k,
        sql_filter,
        Some(embedder),
        cache,
    )
}

pub(crate) fn match_centroids_from_cache(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&crate::SqlStatementInternal>,
    _embedder: Option<&Embedder>,
    _cache: &dyn EmbeddingCache,
) -> Result<Vec<(f32, u32, u32)>> {
    let index = index_for_db(db);
    if let Some(reason) = semantic_index_unavailable_reason(db)? {
        anyhow::bail!(
            "semantic index is not ready: {reason}; run warp-cli index or warp-cli reindex"
        );
    }
    let generation_files = index.generation_files()?;

    let scored_results = match_centroids_raw(
        &generation_files,
        query_embeddings,
        &[],
        threshold,
        top_k,
    )?;

    match sql_filter {
        Some(filter) => {
            let (filter_sql, filter_params) = build_filter_sql_and_params(Some(filter))?;
            db.execute("DROP TABLE IF EXISTS temp2")?;
            db.execute(
                "CREATE TEMPORARY TABLE temp2(rowid INTEGER, sub_idx INTEGER, score FLOAT, UNIQUE(rowid, sub_idx))",
            )?;
            let mut insert_temp_query = db.query("INSERT INTO temp2 VALUES(?1, ?2, ?3)")?;
            for &(score, rowid, sub_idx) in &scored_results {
                let _ = insert_temp_query.execute((rowid, sub_idx, score));
            }
            drop(insert_temp_query);

            let sql = format!(
                "SELECT score,document.rowid,sub_idx
                FROM document,temp2
                WHERE document.rowid = temp2.rowid
                AND {filter_sql}
                ORDER BY score DESC
                LIMIT ?",
            );
            let mut scored_documents_query = db.query(&sql)?;
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = filter_params;
            params.push(Box::new(top_k as i64));
            let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

            let filtered = scored_documents_query
                .query_map(param_refs.as_slice(), |row| {
                    Ok((
                        row.get::<_, f32>(0)?,
                        row.get::<_, u32>(1)?,
                        row.get::<_, u32>(2)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(scored_documents_query);
            db.execute("DROP TABLE temp2")?;
            Ok(filtered)
        }
        None => Ok(scored_results),
    }
}

fn index_for_db(db: &DB) -> FileBackedIndex {
    FileBackedIndex::new(db.path().clone())
}

pub fn semantic_index_unavailable_reason(db: &DB) -> Result<Option<String>> {
    let index = index_for_db(db);
    if let Some(reason) = index.unavailable_reason()? {
        return Ok(Some(reason));
    }
    if !index.buffered_rowid_records()?.is_empty() {
        return Ok(Some(
            "semantic index has buffered unindexed rowids".to_string(),
        ));
    }
    if !index.generation_files()?.is_empty() {
        return Ok(None);
    }
    let has_documents = db.query("SELECT 1 FROM document WHERE length(body) > 0 LIMIT 1")?
        .query_row((), |_| Ok(()))
        .optional()?
        .is_some();
    if has_documents {
        Ok(Some("semantic index is missing".to_string()))
    } else {
        Ok(None)
    }
}

struct DocumentRowidPlan {
    records: Vec<RowidRecord>,
    hashes: HashMap<u64, String>,
}

impl DocumentRowidPlan {
    fn new() -> Self {
        Self {
            records: vec![],
            hashes: HashMap::new(),
        }
    }
}

fn current_document_rowid_plan(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<DocumentRowidPlan> {
    let mut query = db.query(
        "SELECT rowid, hash, body, lens FROM document
         WHERE length(body) > 0
         ORDER BY rowid",
    )?;
    let rows = query.query_map((), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;

    let mut plan = DocumentRowidPlan::new();
    for row in rows {
        let (rowid, hash, body, lens) = row?;
        let rowid_u64: u64 = rowid.try_into()?;
        let hash = hash.unwrap_or_else(|| document_cache_hash(&body, &lens));
        let Some(embeddings) =
            cached_embeddings_for_document(cache, embedder, rowid_u64, &hash, &body, &lens)?
        else {
            continue;
        };
        let indexed_embeddings = cached_embeddings_for_index(embeddings)?;
        plan.hashes.insert(rowid_u64, hash);
        plan.records.push(RowidRecord {
            rowid: rowid_u64,
            rows: indexed_embeddings.embedding_count.try_into()?,
        });
    }
    Ok(plan)
}

fn pending_rowid_records(
    index: &FileBackedIndex,
    current_records: &[RowidRecord],
) -> Result<Vec<RowidRecord>> {
    let mut known = index.all_rowid_map()?;
    let mut pending = vec![];

    for record in current_records {
        match known.remove(&record.rowid) {
            Some(rows) if rows == record.rows => {}
            _ => pending.push(*record),
        }
    }

    for (rowid, rows) in known {
        if rows > 0 {
            pending.push(RowidRecord { rowid, rows: 0 });
        }
    }
    pending.sort_unstable_by_key(|record| record.rowid);
    Ok(pending)
}

fn queued_document_tombstones(db: &DB) -> Result<(Vec<RowidRecord>, Vec<u64>)> {
    let mut query = db.query(
        "SELECT document_index_tombstone.rowid
         FROM document_index_tombstone
         LEFT JOIN document ON document.rowid = document_index_tombstone.rowid
         WHERE document.rowid IS NULL
         ORDER BY document_index_tombstone.rowid",
    )?;
    let rows = query.query_map((), |row| row.get::<_, i64>(0))?;

    let mut records = vec![];
    let mut queued = vec![];
    for row in rows {
        let rowid: u64 = row?.try_into()?;
        queued.push(rowid);
        records.push(RowidRecord { rowid, rows: 0 });
    }
    Ok((records, queued))
}

fn clear_queued_document_tombstones(db: &DB, rowids: &[u64]) -> Result<()> {
    if rowids.is_empty() {
        return Ok(());
    }

    let mut statement = db.query("DELETE FROM document_index_tombstone WHERE rowid = ?1")?;
    for rowid in rowids {
        let rowid: i64 = (*rowid).try_into()?;
        statement.execute((rowid,))?;
    }
    Ok(())
}

struct DocumentEmbeddingSource<'a> {
    cache: &'a dyn EmbeddingCache,
    hashes: HashMap<u64, String>,
}

impl<'a> DocumentEmbeddingSource<'a> {
    fn new(cache: &'a dyn EmbeddingCache, hashes: HashMap<u64, String>) -> Self {
        Self { cache, hashes }
    }
}

impl EmbeddingCache for DocumentEmbeddingSource<'_> {
    fn get(&self, hash: &str) -> Result<Option<CachedEmbeddings>> {
        self.cache.get(hash)
    }

    fn get_for_document(&self, rowid: u64, _hash: &str) -> Result<Option<CachedEmbeddings>> {
        let Some(hash) = self.hashes.get(&rowid) else {
            return Ok(None);
        };
        self.cache.get(hash)
    }

    fn put(&self, hash: &str, embeddings: &CachedEmbeddings) -> Result<()> {
        self.cache.put(hash, embeddings)
    }
}

pub(crate) fn fts5_query(q: &str) -> Option<(String, String)> {
    let terms: Vec<&str> = q
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .collect();
    if terms.is_empty() {
        return None;
    }

    let last_is_space = q.chars().last().is_some_and(char::is_whitespace);
    let mut query = String::new();
    let mut normalized = String::new();
    for (idx, term) in terms.iter().enumerate() {
        if idx != 0 {
            query.push_str(" OR ");
            normalized.push(' ');
        }
        query.push('"');
        query.push_str(term);
        query.push('"');
        if idx + 1 == terms.len() && !last_is_space {
            query.push('*');
        }
        normalized.push_str(term);
    }
    Some((query, normalized))
}

pub fn fulltext_search(
    db: &DB,
    q: &str,
    top_k: usize,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    let mut fts_matches = vec![];

    let fts_query = fts5_query(q);

    let (filter_sql, mut filter_params) = build_filter_sql_and_params(sql_filter)?;
    let filter_clause = if !filter_sql.is_empty() {
        format!("AND {}", filter_sql)
    } else {
        String::new()
    };

    let sql = if fts_query.is_some() {
        format!(
            "SELECT document.rowid, document.body, document.lens,
            bm25(document_fts) AS score
            FROM document,document_fts
            WHERE document.rowid = document_fts.rowid
            AND document_fts MATCH ? {filter_clause}
            ORDER BY score,date DESC
            LIMIT ?",
        )
    } else {
        format!(
            "SELECT rowid,\"\",\"\",0.0
            FROM document
            WHERE 1=1 {filter_clause}
            ORDER BY date DESC
            LIMIT ?",
        )
    };
    let mut query = db.query(&sql)?;

    // Build complete params list: query param (if q.len() > 0), filter params, top_k
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
    if let Some((query, _)) = &fts_query {
        params.push(Box::new(query.clone()));
    }
    params.append(&mut filter_params);
    params.push(Box::new(top_k as i64));

    let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let results = query.query_map(param_refs.as_slice(), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, f32>(3)?,
        ))
    })?;
    for result in results {
        let (rowid, body, lens, score) = result?;
        let rank_score = -score;

        let lens: Vec<usize> = lens
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .collect();

        let mut i_max = 0;
        if !lens.is_empty() {
            let score_query = fts_query
                .as_ref()
                .map(|(_, normalized)| normalized.as_str())
                .unwrap_or("");
            let bodies = split_by_codepoints(&body, &lens);
            let mut max = -1.0f64;
            for (i, &b) in bodies.iter().enumerate() {
                let s = strsim::jaro_winkler(score_query, b);
                if s > max {
                    max = s;
                    i_max = i;
                }
            }
        }
        fts_matches.push((rank_score, rowid, i_max as u32));
    }
    Ok(fts_matches)
}

fn unmaterialized_embedding_count(
    index: &FileBackedIndex,
    current_records: &[RowidRecord],
) -> Result<usize> {
    let indexed = index.indexed_rowid_map()?;
    let mut count = 0usize;
    for record in current_records {
        if indexed.get(&record.rowid).copied() != Some(record.rows) {
            count += record.rows as usize;
        }
    }
    Ok(count)
}

pub fn embed_chunks_with_cache(
    db: &DB,
    embedder: &Embedder,
    cache: &dyn EmbeddingCache,
    limit: Option<usize>,
) -> Result<usize> {
    let _priority_mgr = crate::priority::PriorityManager::new();

    let mut progress = {
        let count_sql = format!(
            "SELECT COUNT(*) FROM (
                SELECT rowid FROM document
                WHERE length(document.body) > 0
                ORDER BY rowid
                {}
            )",
            match limit {
                Some(limit) => format!("LIMIT {limit}"),
                _ => String::new(),
            }
        );
        let mut count_query = db.query(&count_sql)?;
        let total: i64 = count_query.query_row((), |row| row.get(0))?;
        let total: usize = total.try_into()?;
        ProgressReporter::new("embed", total)
    };

    let sql = format!(
        "SELECT
        document.rowid,document.hash,document.body,document.lens
        FROM document
        WHERE length(document.body) > 0
        ORDER BY rowid
        {}",
        match limit {
            Some(limit) => format!("LIMIT {limit}"),
            _ => String::new(),
        }
    );
    let mut query = db.query(&sql)?;

    let mut documents = query.query_map((), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;

    let mut count = 0;
    for result in documents.by_ref() {
        let (rowid, hash, body, lens) = result?;
        let hash = hash.unwrap_or_else(|| document_cache_hash(&body, &lens));
        let (_cached, computed) =
            load_or_compute_cached_embeddings(cache, rowid.try_into()?, &hash, &body, &lens, embedder)?;
        if computed {
            count += 1;
        }
        progress.inc(1);
    }
    progress.finish();

    debug!("computed {count} embedding cache entries");
    if count > 0 {
        db.checkpoint();
    }
    Ok(count)
}

pub fn embed_chunks(db: &DB, embedder: &Embedder, limit: Option<usize>) -> Result<usize> {
    let cache = SqliteEmbeddingCache::new(db);
    embed_chunks_with_cache(db, embedder, &cache, limit)
}

pub fn count_unindexed_embeddings(db: &DB) -> Result<usize> {
    let cache = SqliteEmbeddingCache::new(db);
    let index = index_for_db(db);
    let current = current_document_rowid_plan(db, &cache, None)?;
    unmaterialized_embedding_count(&index, &current.records)
}

pub fn count_unindexed_embeddings_with_cache(
    db: &DB,
    embedder: &Embedder,
    cache: &dyn EmbeddingCache,
) -> Result<usize> {
    let index = index_for_db(db);
    let current = current_document_rowid_plan(db, cache, Some(embedder))?;
    unmaterialized_embedding_count(&index, &current.records)
}

pub fn count_unindexed_cached_embeddings(db: &DB, cache: &dyn EmbeddingCache) -> Result<usize> {
    let index = index_for_db(db);
    let current = current_document_rowid_plan(db, cache, None)?;
    unmaterialized_embedding_count(&index, &current.records)
}

fn cached_embeddings_for_document(
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    rowid: u64,
    hash: &str,
    body: &str,
    lens: &str,
) -> Result<Option<CachedEmbeddings>> {
    if let Some(embedder) = embedder {
        let (embeddings, _computed) =
            load_or_compute_cached_embeddings(cache, rowid, hash, body, lens, embedder)?;
        Ok(Some(embeddings))
    } else {
        load_cached_embeddings(cache, rowid, hash)
    }
}

pub fn index_chunks(
    db: &DB,
    embedder: Option<&Embedder>,
    reset: bool,
) -> Result<()> {
    index_chunks_with_options(db, embedder, reset, IndexOptions::default())
}

pub fn index_chunks_with_options(
    db: &DB,
    embedder: Option<&Embedder>,
    reset: bool,
    options: IndexOptions,
) -> Result<()> {
    let cache = SqliteEmbeddingCache::new(db);
    index_chunks_with_cache_and_options(db, &cache, embedder, reset, options)
}

pub fn index_chunks_with_cache(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    reset: bool,
) -> Result<()> {
    index_chunks_with_cache_and_options(db, cache, embedder, reset, IndexOptions::default())
}

pub fn index_chunks_with_cache_and_options(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    reset: bool,
    options: IndexOptions,
) -> Result<()> {
    let index = index_for_db(db);
    if reset {
        index.clear()?;
        clear_generations_cache();
        db.remove_all_bucket_data_sidecars();
    }

    let current = current_document_rowid_plan(db, cache, embedder)?;
    let unmaterialized = unmaterialized_embedding_count(&index, &current.records)?;
    let mut pending = pending_rowid_records(&index, &current.records)?;
    log::info!("get queued_tombstones");
    let (queued_tombstones, queued_tombstone_rowids) = queued_document_tombstones(db)?;
    pending.extend(queued_tombstones);
    let pending = sort_dedup_rowid_records(pending);
    if pending.is_empty() && unmaterialized == 0 {
        clear_queued_document_tombstones(db, &queued_tombstone_rowids)?;
        return Ok(());
    }

    log::info!("get pending");
    for record in &pending {
        index.append_rowid_record(record.rowid, record.rows)?;
    }
    clear_queued_document_tombstones(db, &queued_tombstone_rowids)?;

    let x = unmaterialized;
    let indexed = index.indexed_embedding_count()?;
    info!("database has {} unindexed embeddings ({} indexed)", x, indexed);

    let source = DocumentEmbeddingSource::new(cache, current.hashes);
    index_buffered_embeddings_with_options(&index, &source, options)?;
    db.checkpoint();
    Ok(())
}

pub fn search(
    db: &DB,
    embedder: Option<&Embedder>,
    cache: &mut EmbeddingsCache,
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, String, Vec<String>, u32, String)>> {
    let now = std::time::Instant::now();

    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");

    let fts_matches = if use_fulltext {
        fulltext_search(db, &q, top_k, sql_filter)?
    } else {
        vec![]
    };

    let sem_matches = if let Some(embedder) = embedder {
        if q.len() > 3 {
            let qe = match cache.get(&q) {
                Some(existing) => existing,
                None => {
                    let (qe, _) = embedder.embed(&q)?;
                    let qe = qe.get(0)?;
                    cache.put(&q, &qe);
                    qe
                }
            };
            let embedding_cache = SqliteEmbeddingCache::new(db);
            match_centroids_with_cache(
                db,
                &qe,
                threshold,
                top_k,
                sql_filter,
                embedder,
                &embedding_cache,
            )?
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let mut scores: HashMap<DocPtr, f32> = HashMap::new();
    let mut offsets: HashMap<DocPtr, u32> = HashMap::new();

    for (score, idx, offset) in &fts_matches {
        let key = (*idx, *offset);
        scores.insert(key, *score);
        offsets.insert(key, *offset);
    }
    for (score, idx, offset) in &sem_matches {
        let key = (*idx, *offset);
        scores.insert(key, *score);
        offsets.insert(key, *offset);
    }

    let sem_idxs: Vec<DocPtr> = sem_matches.iter().map(|&(_, idx, sub_idx)| (idx, sub_idx)).collect();
    info!("semantic search found {} matches", sem_idxs.len());

    let mut fused = if use_fulltext {
        let fts_idxs: Vec<DocPtr> = fts_matches.iter().map(|&(_, idx, sub_idx)| (idx, sub_idx)).collect();
        hybrid_reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
    } else {
        sem_idxs
    };
    fused.truncate(top_k);

    let mut results = vec![];
    // Stale bucket entries from before a re-chunking may have out-of-range sub_idx
    // values that clamp to the same position, producing duplicates.
    let mut seen: HashMap<u32, bool> = HashMap::new();
    let mut body_query = db.query("SELECT metadata,body,lens,date FROM document WHERE rowid = ?1")?;
    for (idx, sub_idx) in fused {
        let tuple : DocPtr = (idx, sub_idx);
        let score = match scores.get(&tuple) {
            Some(score) => *score,
            None => 0.0f32,
        };
        let row = body_query.query_row((idx,), |row| {
            let (metadata, body, lens, date) = (
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            );
            let lens: Vec<usize> = lens
                .split(',')
                .map(|x| x.parse::<usize>().unwrap())
                .collect();
            let bodies: Vec<String> = split_by_codepoints(&body, &lens)
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            Ok((metadata, bodies, date))
        }).optional()?;
        let Some((metadata, bodies, date)) = row else {
            continue;
        };

        let sub = (sub_idx as usize).min(bodies.len().saturating_sub(1)) as u32;
        if seen.insert(idx, true).is_some() {
            continue;
        }
        results.push((score, metadata, bodies, sub, date));
    }

    let mut max = -1.0f32;
    for (score, _, _, _, _) in results.iter_mut().rev() {
        max = max.max(*score);
        *score = max;
    }

    debug!(
        "witchcraft search took {} ms end-to-end.",
        now.elapsed().as_millis()
    );
    Ok(results)
}

pub fn search_rowids(
    db: &DB,
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<u64>> {
    let embedding_cache = SqliteEmbeddingCache::new(db);
    search_rowids_inner(
        db,
        embedder,
        cache,
        Some(&embedding_cache),
        true,
        q,
        threshold,
        top_k,
        use_fulltext,
        sql_filter,
    )
}

pub fn search_cached_rowids_with_cache(
    db: &DB,
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    embedding_cache: &dyn EmbeddingCache,
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<u64>> {
    search_rowids_inner(
        db,
        embedder,
        cache,
        Some(embedding_cache),
        false,
        q,
        threshold,
        top_k,
        use_fulltext,
        sql_filter,
    )
}

fn search_rowids_inner(
    db: &DB,
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    embedding_cache: Option<&dyn EmbeddingCache>,
    compute_missing_embeddings: bool,
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<u64>> {
    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");

    let fts_matches = if use_fulltext {
        fulltext_search(db, &q, top_k, sql_filter)?
    } else {
        vec![]
    };

    let sem_matches = if q.len() > 3 {
        let qe = match cache.get(&q) {
            Some(existing) => existing,
            None => {
                let (qe, _) = embedder.embed(&q)?;
                let qe = qe.get(0)?;
                cache.put(&q, &qe);
                qe
            }
        };
        match (embedding_cache, compute_missing_embeddings) {
            (Some(embedding_cache), true) => match_centroids_with_cache(
                db,
                &qe,
                threshold,
                top_k,
                sql_filter,
                embedder,
                embedding_cache,
            )?,
            (Some(embedding_cache), false) => match_centroids_from_cache(
                db,
                &qe,
                threshold,
                top_k,
                sql_filter,
                None,
                embedding_cache,
            )?,
            (None, _) => match_centroids(db, &qe, threshold, top_k, sql_filter)?,
        }
    } else {
        vec![]
    };

    let sem_idxs: Vec<DocPtr> = sem_matches.iter().map(|&(_, idx, sub_idx)| (idx, sub_idx)).collect();
    let mut fused = if use_fulltext {
        let fts_idxs: Vec<DocPtr> = fts_matches.iter().map(|&(_, idx, sub_idx)| (idx, sub_idx)).collect();
        hybrid_reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
    } else {
        sem_idxs
    };
    fused.truncate(top_k);

    let mut rowids = Vec::with_capacity(fused.len());
    let mut seen: HashMap<u32, bool> = HashMap::new();
    let mut exists_query = db.query("SELECT 1 FROM document WHERE rowid = ?1")?;
    for (rowid, _) in fused {
        if seen.insert(rowid, true).is_none()
            && exists_query
                .query_row((rowid,), |_| Ok(()))
                .optional()?
                .is_some()
        {
            rowids.push(rowid as u64);
        }
    }
    Ok(rowids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;

    #[test]
    fn fts5_query_splits_punctuation_and_uses_or_terms() {
        let (query, normalized) = fts5_query("what is the origin of COVID-19").unwrap();

        assert_eq!(
            query,
            "\"what\" OR \"is\" OR \"the\" OR \"origin\" OR \"of\" OR \"COVID\" OR \"19\"*"
        );
        assert_eq!(normalized, "what is the origin of COVID 19");
    }

    #[test]
    fn fulltext_search_matches_hyphenated_terms_without_requiring_every_word() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("warp.sqlite");
        let mut db = DB::new(path)?;

        let relevant = "The origin of COVID-19 was investigated in early pandemic research.";
        let distractor = "This paragraph says what is the origin of an unrelated weather report.";
        let relevant_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, relevant.as_bytes());
        let distractor_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, distractor.as_bytes());
        db.add_doc(Some(1), &relevant_uuid, None, "relevant", relevant, None)?;
        db.add_doc(
            Some(2),
            &distractor_uuid,
            None,
            "distractor",
            distractor,
            None,
        )?;

        let results = fulltext_search(&db, "what is the origin of COVID-19", 10, None)?;

        assert!(
            results.iter().any(|(_, rowid, _)| *rowid == 1),
            "expected relevant COVID-19 document in {results:?}"
        );
        Ok(())
    }
}
