use crate::file_index::{FileBackedIndex, RowidRecord, RowidRecordAppender};
use crate::packops::TensorPackOps;
use crate::progress_reporter::ProgressReporter;
use crate::sql_generator::build_filter_sql_and_params;
use crate::{
    cached_embeddings_for_index, clear_generations_cache, dim_from_model_id, document_cache_hash,
    embed_query_for_search, hybrid_reciprocal_rank_fusion, index_buffered_embeddings_with_options,
    load_cached_embeddings, load_or_compute_cached_embeddings, match_centroids_raw,
    model_id_for_dim, query_token_salience_enabled, split_by_codepoints, CachedEmbeddings, DocPtr,
    Embedder, EmbeddingCache, EmbeddingsCache, IndexOptions, QueryEmbeddings, SqlStatementInternal,
    DB,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use log::{debug, info};
use rusqlite::OptionalExtension;
use std::collections::{HashMap, HashSet};

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
            "SELECT model, counts, embedding_count, embeddings
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
            "SELECT chunk.model, chunk.counts, chunk.embedding_count, chunk.embeddings
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
                })
            })
            .optional()?;
        Ok(cached)
    }

    fn put(&self, hash: &str, embeddings: &CachedEmbeddings) -> Result<()> {
        let mut statement = self.db.query(
            "INSERT INTO chunk(hash, model, embeddings, counts, embedding_count)
             VALUES(?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(hash) DO UPDATE SET
                 model = excluded.model,
                 embeddings = excluded.embeddings,
                 counts = excluded.counts,
                 embedding_count = excluded.embedding_count",
        )?;
        statement.execute((
            hash,
            &embeddings.model,
            &embeddings.embeddings,
            &embeddings.counts,
            i64::try_from(embeddings.embedding_count)?,
        ))?;
        Ok(())
    }
}

/// DB-backed wrapper: loads generations and fetches buffered unindexed
/// documents that already have cached embeddings.
pub fn match_centroids(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&crate::SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    match_centroids_with_query_weights(db, query_embeddings, None, threshold, top_k, sql_filter)
}

pub fn match_centroids_with_query_weights(
    db: &DB,
    query_embeddings: &Tensor,
    query_weights: Option<&[f32]>,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&crate::SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    let cache = SqliteEmbeddingCache::new(db);
    match_centroids_from_cache(
        db,
        query_embeddings,
        query_weights,
        threshold,
        top_k,
        sql_filter,
        None,
        &cache,
    )
}

fn docptrs_for_cached_counts(rowid: u32, counts: &str) -> Vec<DocPtr> {
    let mut ptrs = Vec::new();
    for (sub_idx, count) in counts
        .split(',')
        .filter_map(|count| count.parse::<u32>().ok())
        .enumerate()
    {
        for _ in 0..count {
            ptrs.push((rowid, sub_idx as u32));
        }
    }
    ptrs
}

fn flush_exact_batch(
    query_embeddings: &[Tensor],
    ptrs: &mut Vec<DocPtr>,
    tensors: &mut Vec<Tensor>,
    rows: &mut usize,
    top_k: usize,
    out: &mut [Vec<(f32, u32, u32)>],
) -> Result<()> {
    if *rows == 0 {
        return Ok(());
    }
    let matrix = Tensor::cat(tensors, 0)?;
    let ptrs = std::mem::take(ptrs);
    for (query_idx, query) in query_embeddings.iter().enumerate() {
        let unindexed = vec![(ptrs.clone(), matrix.clone())];
        let mut batch =
            match_centroids_raw(&[], query, None, &unindexed, f32::NEG_INFINITY, top_k)?;
        out[query_idx].append(&mut batch);
        out[query_idx].sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        out[query_idx].truncate(top_k);
    }
    tensors.clear();
    *rows = 0;
    Ok(())
}

fn query_embeddings_for_search(
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    q: &str,
) -> Result<QueryEmbeddings> {
    if query_token_salience_enabled() {
        return embed_query_for_search(embedder, q);
    }

    let embeddings = match cache.get(&q.to_string()) {
        Some(existing) => existing,
        None => {
            let (embeddings, _) = embedder.embed(q)?;
            let embeddings = embeddings.get(0)?;
            cache.put(&q.to_string(), &embeddings);
            embeddings
        }
    };
    Ok(QueryEmbeddings {
        embeddings,
        weights: None,
    })
}

pub fn exact_match_centroids_bulk(
    db: &DB,
    query_embeddings: &[Tensor],
    top_k: usize,
) -> Result<Vec<Vec<(f32, u32, u32)>>> {
    const EXACT_BATCH_ROWS: usize = 200_000;

    let cache = SqliteEmbeddingCache::new(db);
    let mut query = db.query(
        "SELECT rowid, hash FROM document
         WHERE length(body) > 0
         ORDER BY rowid",
    )?;
    let documents = query.query_map((), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
    })?;

    let mut ptrs = Vec::new();
    let mut tensors = Vec::new();
    let mut rows = 0usize;
    let mut out = vec![Vec::new(); query_embeddings.len()];

    for document in documents {
        let (rowid, hash) = document?;
        let rowid_u64: u64 = rowid.try_into()?;
        let rowid_u32: u32 = rowid.try_into()?;
        let Some(cached) = cache.get_for_document(rowid_u64, hash.as_deref().unwrap_or(""))? else {
            continue;
        };
        let cached = cached_embeddings_for_index(cached)?;
        if cached.embedding_count == 0 {
            continue;
        }
        let docptrs = docptrs_for_cached_counts(rowid_u32, &cached.counts);
        anyhow::ensure!(
            docptrs.len() == cached.embedding_count,
            "rowid {rowid} has {} docptrs but {} embeddings",
            docptrs.len(),
            cached.embedding_count
        );
        let embeddings = Tensor::embeddings_from_packed(
            &cached.embeddings,
            dim_from_model_id(&cached.model),
            &Device::Cpu,
        )?;
        rows += cached.embedding_count;
        ptrs.extend(docptrs);
        tensors.push(embeddings);
        if rows >= EXACT_BATCH_ROWS {
            flush_exact_batch(
                query_embeddings,
                &mut ptrs,
                &mut tensors,
                &mut rows,
                top_k,
                &mut out,
            )?;
        }
    }
    flush_exact_batch(
        query_embeddings,
        &mut ptrs,
        &mut tensors,
        &mut rows,
        top_k,
        &mut out,
    )?;
    Ok(out)
}

/// DB-backed wrapper that can demand-populate missing buffered document
/// embeddings through the supplied cache.
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
        None,
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
    query_weights: Option<&[f32]>,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&crate::SqlStatementInternal>,
    embedder: Option<&Embedder>,
    cache: &dyn EmbeddingCache,
) -> Result<Vec<(f32, u32, u32)>> {
    let index = index_for_db(db);
    let generation_files = index.generation_files()?;
    let source = if embedder.is_some() {
        DocumentEmbeddingSource::new(db, cache)
    } else {
        DocumentEmbeddingSource::compatible(db, cache)
    };
    let buffered = index.buffered_rowid_records()?;
    let unindexed = if !buffered.is_empty() {
        unindexed_embeddings_for_buffered_rowids(db, &buffered, cache, embedder, &source)?
    } else if generation_files.is_empty() {
        unindexed_embeddings_for_current_documents(db, cache, embedder, &source)?
    } else {
        vec![]
    };

    let scored_results = match_centroids_raw(
        &generation_files,
        query_embeddings,
        query_weights,
        &unindexed,
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
    let has_documents = db
        .query("SELECT 1 FROM document WHERE length(body) > 0 LIMIT 1")?
        .query_row((), |_| Ok(()))
        .optional()?
        .is_some();
    if has_documents {
        Ok(Some("semantic index is missing".to_string()))
    } else {
        Ok(None)
    }
}

fn document_hash_for_rowid(db: &DB, rowid: u64) -> Result<Option<String>> {
    let rowid: i64 = rowid.try_into()?;
    let mut hash_query = db.query(
        "SELECT hash FROM document
         WHERE rowid = ?1 AND length(body) > 0 AND hash IS NOT NULL",
    )?;
    if let Some(hash) = hash_query
        .query_row((rowid,), |row| row.get::<_, String>(0))
        .optional()?
    {
        return Ok(Some(hash));
    }

    let mut body_query = db.query(
        "SELECT body, lens FROM document
         WHERE rowid = ?1 AND length(body) > 0",
    )?;
    let row = body_query
        .query_row((rowid,), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .optional()?;
    Ok(row.map(|(body, lens)| document_cache_hash(&body, &lens)))
}

fn for_each_current_document_rowid_record(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    mut f: impl FnMut(RowidRecord) -> Result<()>,
) -> Result<()> {
    if let Some(embedder) = embedder {
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

        for row in rows {
            let (rowid, hash, body, lens) = row?;
            let rowid: u64 = rowid.try_into()?;
            let hash = hash.unwrap_or_else(|| document_cache_hash(&body, &lens));
            let Some(embeddings) =
                cached_embeddings_for_document(cache, Some(embedder), rowid, &hash, &body, &lens)?
            else {
                continue;
            };
            f(RowidRecord {
                rowid,
                rows: embeddings.embedding_count.try_into()?,
            })?;
        }
        return Ok(());
    }

    let mut query = db.query(
        "SELECT rowid, hash FROM document
         WHERE length(body) > 0
         ORDER BY rowid",
    )?;
    let rows = query.query_map((), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
    })?;

    for row in rows {
        let (rowid, hash) = row?;
        let rowid: u64 = rowid.try_into()?;
        let hash = match hash {
            Some(hash) => hash,
            None => {
                let Some(hash) = document_hash_for_rowid(db, rowid)? else {
                    continue;
                };
                hash
            }
        };
        let Some(embeddings) = load_cached_embeddings(cache, rowid, &hash)? else {
            continue;
        };
        f(RowidRecord {
            rowid,
            rows: embeddings.embedding_count.try_into()?,
        })?;
    }
    Ok(())
}

fn for_each_buffered_document_rowid_record(
    db: &DB,
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    mut f: impl FnMut(RowidRecord) -> Result<()>,
) -> Result<()> {
    let mut query = db.query(
        "SELECT hash, body, lens FROM document
         WHERE rowid = ?1 AND length(body) > 0",
    )?;

    for record in records.iter().copied().filter(|record| record.rows > 0) {
        let rowid: i64 = record.rowid.try_into()?;
        let row = query
            .query_row((rowid,), |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .optional()?;
        let Some((hash, body, lens)) = row else {
            continue;
        };
        let hash = hash.unwrap_or_else(|| document_cache_hash(&body, &lens));
        let Some(embeddings) =
            cached_embeddings_for_document(cache, embedder, record.rowid, &hash, &body, &lens)?
        else {
            continue;
        };
        anyhow::ensure!(
            embeddings.embedding_count == record.rows as usize,
            "rowid {} catalog says {} vectors but embedding blob has {}",
            record.rowid,
            record.rows,
            embeddings.embedding_count
        );
        f(record)?;
    }
    Ok(())
}

fn model_without_pruning_fraction(model: &str) -> Option<String> {
    let (family, suffix) = model.rsplit_once("-p")?;
    let dim_suffix = suffix.find("-d").map(|idx| &suffix[idx..]).unwrap_or("");
    Some(format!("{family}{dim_suffix}"))
}

fn cached_model_compatible_with_current_encoder(model: &str) -> bool {
    let current = model_id_for_dim(dim_from_model_id(model));
    match (
        model_without_pruning_fraction(model),
        model_without_pruning_fraction(&current),
    ) {
        (Some(model), Some(current)) => model == current,
        _ => model == current,
    }
}

fn normalize_compatible_cached_embeddings(
    mut embeddings: CachedEmbeddings,
) -> Option<CachedEmbeddings> {
    if !cached_model_compatible_with_current_encoder(&embeddings.model) {
        return None;
    }
    embeddings.model = model_id_for_dim(dim_from_model_id(&embeddings.model));
    Some(embeddings)
}

fn current_model_sql_filter_values() -> (String, String) {
    let current_model = model_id_for_dim(dim_from_model_id(""));
    let family = current_model
        .rsplit_once("-p")
        .map(|(family, _)| family)
        .unwrap_or(current_model.as_str());
    let default_dim_models = format!("{family}-p[0-9][0-9][0-9]");
    let dim_models = format!("{family}-p[0-9][0-9][0-9]-d[0-9]*");
    (default_dim_models, dim_models)
}

fn for_each_current_sqlite_cached_document_rowid_record(
    db: &DB,
    mut f: impl FnMut(RowidRecord) -> Result<()>,
) -> Result<()> {
    let (default_dim_models, dim_models) = current_model_sql_filter_values();
    let mut query = db.query(
        "SELECT document.rowid, chunk.model, chunk.embedding_count
         FROM document INDEXED BY document_nonempty_hash_index
         JOIN chunk INDEXED BY chunk_hash_model_embedding_count_index
         ON document.hash = chunk.hash
         WHERE length(document.body) > 0
           AND (chunk.model GLOB ?1 OR chunk.model GLOB ?2)",
    )?;
    let rows = query.query_map((&default_dim_models, &dim_models), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    for row in rows {
        let (rowid, model, embedding_count) = row?;
        if !cached_model_compatible_with_current_encoder(&model) {
            continue;
        }
        f(RowidRecord {
            rowid: rowid.try_into()?,
            rows: embedding_count.try_into()?,
        })?;
    }
    Ok(())
}

fn current_unmaterialized_embedding_count_from(
    index: &FileBackedIndex,
    for_each_record: impl FnOnce(&mut dyn FnMut(RowidRecord) -> Result<()>) -> Result<()>,
) -> Result<usize> {
    let indexed = index.indexed_rowid_map()?;
    let mut count = 0usize;
    let mut visit = |record: RowidRecord| {
        if indexed.get(&record.rowid).copied() != Some(record.rows) {
            count += record.rows as usize;
        }
        Ok(())
    };
    for_each_record(&mut visit)?;
    Ok(count)
}

fn current_unmaterialized_embedding_count(
    index: &FileBackedIndex,
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<usize> {
    current_unmaterialized_embedding_count_from(index, |visit| {
        for_each_current_document_rowid_record(db, cache, embedder, visit)
    })
}

fn current_sqlite_cached_unmaterialized_embedding_count(
    index: &FileBackedIndex,
    db: &DB,
) -> Result<usize> {
    current_unmaterialized_embedding_count_from(index, |visit| {
        for_each_current_sqlite_cached_document_rowid_record(db, visit)
    })
}

fn current_document_row_count(db: &DB) -> Result<usize> {
    let mut query = db.query("SELECT COUNT(*) FROM document WHERE length(body) > 0")?;
    let total: i64 = query.query_row((), |row| row.get(0))?;
    Ok(total.try_into()?)
}

fn current_sqlite_cached_document_row_count(db: &DB) -> Result<usize> {
    let (default_dim_models, dim_models) = current_model_sql_filter_values();
    let mut query = db.query(
        "SELECT COUNT(*)
         FROM document INDEXED BY document_nonempty_hash_index
         JOIN chunk INDEXED BY chunk_hash_model_embedding_count_index
         ON document.hash = chunk.hash
         WHERE length(document.body) > 0
           AND (chunk.model GLOB ?1 OR chunk.model GLOB ?2)",
    )?;
    let total: i64 = query.query_row((&default_dim_models, &dim_models), |row| row.get(0))?;
    Ok(total.try_into()?)
}

fn append_unindexed_embedding(
    unindexed: &mut Vec<(Vec<DocPtr>, Tensor)>,
    record: RowidRecord,
    cache: &dyn EmbeddingCache,
) -> Result<()> {
    if record.rows == 0 {
        return Ok(());
    }

    let cached = load_cached_embeddings(cache, record.rowid, "")?
        .ok_or_else(|| anyhow::anyhow!("missing embeddings for rowid {}", record.rowid))?;
    anyhow::ensure!(
        cached.embedding_count == record.rows as usize,
        "rowid {} catalog says {} vectors but embedding blob has {}",
        record.rowid,
        record.rows,
        cached.embedding_count
    );
    let embeddings = Tensor::embeddings_from_packed(
        &cached.embeddings,
        dim_from_model_id(&cached.model),
        &Device::Cpu,
    )?;
    let rowid: u32 = record.rowid.try_into()?;
    let ptrs = docptrs_for_cached_counts(rowid, &cached.counts);
    anyhow::ensure!(
        ptrs.len() == cached.embedding_count,
        "rowid {} has {} embedding rows but {} document indices",
        record.rowid,
        cached.embedding_count,
        ptrs.len()
    );
    unindexed.push((ptrs, embeddings));
    Ok(())
}

fn unindexed_embeddings_for_buffered_rowids(
    db: &DB,
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    source: &dyn EmbeddingCache,
) -> Result<Vec<(Vec<DocPtr>, Tensor)>> {
    let mut unindexed = vec![];
    for_each_buffered_document_rowid_record(db, records, cache, embedder, |record| {
        append_unindexed_embedding(&mut unindexed, record, source)
    })?;
    Ok(unindexed)
}

fn unindexed_embeddings_for_current_documents(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    source: &dyn EmbeddingCache,
) -> Result<Vec<(Vec<DocPtr>, Tensor)>> {
    let mut unindexed = vec![];
    for_each_current_document_rowid_record(db, cache, embedder, |record| {
        append_unindexed_embedding(&mut unindexed, record, source)
    })?;
    Ok(unindexed)
}

#[derive(Default)]
struct CurrentRowidDelta {
    pending_records: usize,
    unmaterialized_embeddings: usize,
}

fn append_rowid_record(
    index: &FileBackedIndex,
    appender: &mut Option<RowidRecordAppender>,
    record: RowidRecord,
) -> Result<()> {
    if appender.is_none() {
        *appender = Some(index.rowid_record_appender()?);
    }
    appender
        .as_mut()
        .expect("rowid appender should be initialized")
        .append(record)
}

fn append_pending_rowid_records_from(
    index: &FileBackedIndex,
    db: &DB,
    total: usize,
    for_each_record: impl FnOnce(&mut dyn FnMut(RowidRecord) -> Result<()>) -> Result<()>,
) -> Result<(CurrentRowidDelta, Vec<u64>)> {
    let indexed = index.indexed_rowid_map()?;
    let mut known = index.all_rowid_map()?;
    let mut delta = CurrentRowidDelta::default();
    let mut progress = ProgressReporter::new("queue rowids", total);
    let mut appender = None;

    {
        let mut visit = |record: RowidRecord| {
            if indexed.get(&record.rowid).copied() != Some(record.rows) {
                delta.unmaterialized_embeddings += record.rows as usize;
            }
            match known.remove(&record.rowid) {
                Some(rows) if rows == record.rows => {}
                _ => {
                    append_rowid_record(index, &mut appender, record)?;
                    delta.pending_records += 1;
                }
            }
            progress.inc(1);
            Ok(())
        };
        for_each_record(&mut visit)?;
    }
    progress.finish();

    let mut tombstones = HashSet::new();
    for (rowid, rows) in known {
        if rows > 0 {
            tombstones.insert(rowid);
        }
    }

    let (queued_tombstones, queued_tombstone_rowids) = queued_document_tombstones(db)?;
    for tombstone in queued_tombstones {
        tombstones.insert(tombstone.rowid);
    }

    let mut tombstones: Vec<_> = tombstones.into_iter().collect();
    tombstones.sort_unstable();
    for rowid in tombstones {
        append_rowid_record(index, &mut appender, RowidRecord { rowid, rows: 0 })?;
        delta.pending_records += 1;
    }

    if let Some(appender) = appender {
        appender.finish()?;
    }

    Ok((delta, queued_tombstone_rowids))
}

fn append_pending_rowid_records(
    index: &FileBackedIndex,
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<(CurrentRowidDelta, Vec<u64>)> {
    let total = if embedder.is_some() {
        current_document_row_count(db)?
    } else {
        0
    };
    append_pending_rowid_records_from(index, db, total, |visit| {
        for_each_current_document_rowid_record(db, cache, embedder, visit)
    })
}

fn append_pending_sqlite_cached_rowid_records(
    index: &FileBackedIndex,
    db: &DB,
) -> Result<(CurrentRowidDelta, Vec<u64>)> {
    append_pending_rowid_records_from(
        index,
        db,
        current_sqlite_cached_document_row_count(db)?,
        |visit| for_each_current_sqlite_cached_document_rowid_record(db, visit),
    )
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
    db: &'a DB,
    cache: &'a dyn EmbeddingCache,
    accept_compatible_models: bool,
    resolve_hash_for_rowid: bool,
}

impl<'a> DocumentEmbeddingSource<'a> {
    fn new(db: &'a DB, cache: &'a dyn EmbeddingCache) -> Self {
        Self {
            db,
            cache,
            accept_compatible_models: false,
            resolve_hash_for_rowid: true,
        }
    }

    fn compatible(db: &'a DB, cache: &'a dyn EmbeddingCache) -> Self {
        Self {
            db,
            cache,
            accept_compatible_models: true,
            resolve_hash_for_rowid: true,
        }
    }

    fn compatible_with_rowid_lookup(db: &'a DB, cache: &'a dyn EmbeddingCache) -> Self {
        Self {
            db,
            cache,
            accept_compatible_models: true,
            resolve_hash_for_rowid: false,
        }
    }

    fn filter_cached_embeddings(
        &self,
        embeddings: Option<CachedEmbeddings>,
    ) -> Option<CachedEmbeddings> {
        if self.accept_compatible_models {
            embeddings.and_then(normalize_compatible_cached_embeddings)
        } else {
            embeddings
        }
    }
}

impl EmbeddingCache for DocumentEmbeddingSource<'_> {
    fn get(&self, hash: &str) -> Result<Option<CachedEmbeddings>> {
        Ok(self.filter_cached_embeddings(self.cache.get(hash)?))
    }

    fn get_for_document(&self, rowid: u64, hash: &str) -> Result<Option<CachedEmbeddings>> {
        if !hash.is_empty() || !self.resolve_hash_for_rowid {
            return Ok(self.filter_cached_embeddings(self.cache.get_for_document(rowid, hash)?));
        }

        let Some(hash) = document_hash_for_rowid(self.db, rowid)? else {
            return Ok(None);
        };
        Ok(self.filter_cached_embeddings(self.cache.get_for_document(rowid, &hash)?))
    }

    fn put(&self, hash: &str, embeddings: &CachedEmbeddings) -> Result<()> {
        self.cache.put(hash, embeddings)
    }
}

fn load_fts_stopwords(db: &DB) -> Result<HashSet<String>> {
    let mut exists_query = db.query(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'document_fts_stopword'",
    )?;
    let has_stopword_table = exists_query.query_row((), |_| Ok(())).optional()?.is_some();
    if !has_stopword_table {
        debug!("document_fts_stopword table missing; continuing without FTS stopwords");
        return Ok(HashSet::new());
    }

    let mut query = db.query("SELECT term FROM document_fts_stopword")?;
    let rows = query.query_map((), |row| row.get::<_, String>(0))?;
    let mut stopwords = HashSet::new();
    for row in rows {
        stopwords.insert(row?.to_lowercase());
    }
    Ok(stopwords)
}

pub(crate) fn fts5_query(q: &str, stopwords: &HashSet<String>) -> Option<(String, String)> {
    fts5_query_with_prefix_wildcard(q, false, stopwords)
}

fn fts5_query_with_prefix_wildcard(
    q: &str,
    prefix_last_term: bool,
    stopwords: &HashSet<String>,
) -> Option<(String, String)> {
    let raw_terms: Vec<&str> = q
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .collect();
    let last_raw_term_idx = raw_terms.len().checked_sub(1)?;
    let terms: Vec<(usize, &str)> = raw_terms
        .iter()
        .enumerate()
        .filter_map(|(idx, &term)| {
            if stopwords.contains(&term.to_lowercase()) {
                None
            } else {
                Some((idx, term))
            }
        })
        .collect();
    if terms.is_empty() {
        return None;
    }

    let last_is_space = q.chars().last().is_some_and(char::is_whitespace);
    let mut query = String::new();
    let mut normalized = String::new();
    for (idx, (raw_idx, term)) in terms.iter().enumerate() {
        if idx != 0 {
            query.push_str(" OR ");
            normalized.push(' ');
        }
        query.push('"');
        query.push_str(term);
        query.push('"');
        if prefix_last_term && *raw_idx == last_raw_term_idx && !last_is_space {
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
    fulltext_search_with_prefix_wildcard(db, q, top_k, sql_filter, false)
}

pub fn fulltext_search_with_prefix_wildcard(
    db: &DB,
    q: &str,
    top_k: usize,
    sql_filter: Option<&SqlStatementInternal>,
    prefix_last_term: bool,
) -> Result<Vec<(f32, u32, u32)>> {
    let mut fts_matches = vec![];

    let stopwords = load_fts_stopwords(db)?;
    let fts_query = if prefix_last_term {
        fts5_query_with_prefix_wildcard(q, true, &stopwords)
    } else {
        fts5_query(q, &stopwords)
    };

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
        let (_cached, computed) = load_or_compute_cached_embeddings(
            cache,
            rowid.try_into()?,
            &hash,
            &body,
            &lens,
            embedder,
        )?;
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
    let index = index_for_db(db);
    current_sqlite_cached_unmaterialized_embedding_count(&index, db)
}

pub fn count_unindexed_embeddings_with_cache(
    db: &DB,
    embedder: &Embedder,
    cache: &dyn EmbeddingCache,
) -> Result<usize> {
    let index = index_for_db(db);
    current_unmaterialized_embedding_count(&index, db, cache, Some(embedder))
}

pub fn count_unindexed_cached_embeddings(db: &DB, cache: &dyn EmbeddingCache) -> Result<usize> {
    let index = index_for_db(db);
    current_unmaterialized_embedding_count(&index, db, cache, None)
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

pub fn index_chunks(db: &DB, embedder: Option<&Embedder>, reset: bool) -> Result<()> {
    index_chunks_with_options(db, embedder, reset, IndexOptions::default())
}

pub fn index_chunks_with_options(
    db: &DB,
    embedder: Option<&Embedder>,
    reset: bool,
    options: IndexOptions,
) -> Result<()> {
    if let Some(embedder) = embedder {
        let cache = SqliteEmbeddingCache::new(db);
        return index_chunks_with_cache_and_options(db, &cache, Some(embedder), reset, options);
    }

    let index = index_for_db(db);
    if reset {
        index.clear()?;
        clear_generations_cache();
        db.remove_all_bucket_data_sidecars();
    }

    let (delta, queued_tombstone_rowids) = append_pending_sqlite_cached_rowid_records(&index, db)?;
    if delta.pending_records == 0 && delta.unmaterialized_embeddings == 0 {
        clear_queued_document_tombstones(db, &queued_tombstone_rowids)?;
        return Ok(());
    }

    clear_queued_document_tombstones(db, &queued_tombstone_rowids)?;

    let indexed = index.indexed_embedding_count()?;
    info!(
        "database has {} unindexed embeddings ({} indexed)",
        delta.unmaterialized_embeddings, indexed
    );

    let cache = SqliteEmbeddingCache::new(db);
    let source = DocumentEmbeddingSource::compatible_with_rowid_lookup(db, &cache);
    index_buffered_embeddings_with_options(&index, &source, options)?;
    db.checkpoint();
    Ok(())
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

    let (delta, queued_tombstone_rowids) =
        append_pending_rowid_records(&index, db, cache, embedder)?;
    if delta.pending_records == 0 && delta.unmaterialized_embeddings == 0 {
        clear_queued_document_tombstones(db, &queued_tombstone_rowids)?;
        return Ok(());
    }

    clear_queued_document_tombstones(db, &queued_tombstone_rowids)?;

    let indexed = index.indexed_embedding_count()?;
    info!(
        "database has {} unindexed embeddings ({} indexed)",
        delta.unmaterialized_embeddings, indexed
    );

    let source = if embedder.is_some() {
        DocumentEmbeddingSource::new(db, cache)
    } else {
        DocumentEmbeddingSource::compatible(db, cache)
    };
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
    search_inner(
        db,
        embedder,
        cache,
        q,
        threshold,
        top_k,
        use_fulltext,
        sql_filter,
        false,
    )
}

pub fn search_with_fulltext_prefix_wildcard(
    db: &DB,
    embedder: Option<&Embedder>,
    cache: &mut EmbeddingsCache,
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, String, Vec<String>, u32, String)>> {
    search_inner(
        db,
        embedder,
        cache,
        q,
        threshold,
        top_k,
        use_fulltext,
        sql_filter,
        true,
    )
}

fn search_inner(
    db: &DB,
    embedder: Option<&Embedder>,
    cache: &mut EmbeddingsCache,
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
    fulltext_prefix_wildcard: bool,
) -> Result<Vec<(f32, String, Vec<String>, u32, String)>> {
    let now = std::time::Instant::now();

    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");

    let fts_matches = if use_fulltext {
        fulltext_search_with_prefix_wildcard(db, &q, top_k, sql_filter, fulltext_prefix_wildcard)?
    } else {
        vec![]
    };

    let sem_matches = if let Some(embedder) = embedder {
        if q.len() > 3 {
            let qe = query_embeddings_for_search(embedder, cache, &q)?;
            let embedding_cache = SqliteEmbeddingCache::new(db);
            match_centroids_from_cache(
                db,
                &qe.embeddings,
                qe.weights.as_deref(),
                threshold,
                top_k,
                sql_filter,
                Some(embedder),
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

    let sem_idxs: Vec<DocPtr> = sem_matches
        .iter()
        .map(|&(_, idx, sub_idx)| (idx, sub_idx))
        .collect();
    info!("semantic search found {} matches", sem_idxs.len());

    let mut fused = if use_fulltext {
        let fts_idxs: Vec<DocPtr> = fts_matches
            .iter()
            .map(|&(_, idx, sub_idx)| (idx, sub_idx))
            .collect();
        hybrid_reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
    } else {
        sem_idxs
    };
    fused.truncate(top_k);

    let mut results = vec![];
    // Stale bucket entries from before a re-chunking may have out-of-range sub_idx
    // values that clamp to the same position, producing duplicates.
    let mut seen: HashMap<u32, bool> = HashMap::new();
    let mut body_query =
        db.query("SELECT metadata,body,lens,date FROM document WHERE rowid = ?1")?;
    for (idx, sub_idx) in fused {
        let tuple: DocPtr = (idx, sub_idx);
        let score = match scores.get(&tuple) {
            Some(score) => *score,
            None => 0.0f32,
        };
        let row = body_query
            .query_row((idx,), |row| {
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
            })
            .optional()?;
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
        let qe = query_embeddings_for_search(embedder, cache, &q)?;
        match (embedding_cache, compute_missing_embeddings) {
            (Some(embedding_cache), true) => match_centroids_from_cache(
                db,
                &qe.embeddings,
                qe.weights.as_deref(),
                threshold,
                top_k,
                sql_filter,
                Some(embedder),
                embedding_cache,
            )?,
            (Some(embedding_cache), false) => match_centroids_from_cache(
                db,
                &qe.embeddings,
                qe.weights.as_deref(),
                threshold,
                top_k,
                sql_filter,
                None,
                embedding_cache,
            )?,
            (None, _) => match_centroids_with_query_weights(
                db,
                &qe.embeddings,
                qe.weights.as_deref(),
                threshold,
                top_k,
                sql_filter,
            )?,
        }
    } else {
        vec![]
    };

    let sem_idxs: Vec<DocPtr> = sem_matches
        .iter()
        .map(|&(_, idx, sub_idx)| (idx, sub_idx))
        .collect();
    let mut fused = if use_fulltext {
        let fts_idxs: Vec<DocPtr> = fts_matches
            .iter()
            .map(|&(_, idx, sub_idx)| (idx, sub_idx))
            .collect();
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

    fn test_stopwords(words: &[&str]) -> HashSet<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    #[test]
    fn cached_model_compatibility_ignores_document_pruning_fraction() {
        let current = model_id_for_dim(96);
        let (family, suffix) = current.rsplit_once("-p").unwrap();
        let dim_suffix = suffix.find("-d").map(|idx| &suffix[idx..]).unwrap_or("");
        let alternate_pruning_fraction = format!("{family}-p025{dim_suffix}");

        assert!(cached_model_compatible_with_current_encoder(&current));
        assert!(cached_model_compatible_with_current_encoder(
            &alternate_pruning_fraction
        ));
        let normalized = normalize_compatible_cached_embeddings(CachedEmbeddings {
            model: alternate_pruning_fraction,
            counts: "1".to_string(),
            embedding_count: 1,
            embeddings: vec![],
        })
        .unwrap();
        assert_eq!(normalized.model, current);
        assert!(!cached_model_compatible_with_current_encoder(
            "other-encoder-p025-d96"
        ));
    }

    #[test]
    fn fts5_query_splits_punctuation_and_uses_or_terms() {
        let stopwords = HashSet::new();
        let (query, normalized) = fts5_query("what is the origin of COVID-19", &stopwords).unwrap();

        assert_eq!(
            query,
            "\"what\" OR \"is\" OR \"the\" OR \"origin\" OR \"of\" OR \"COVID\" OR \"19\""
        );
        assert_eq!(normalized, "what is the origin of COVID 19");
    }

    #[test]
    fn fts5_query_can_enable_final_prefix_wildcard() {
        let stopwords = HashSet::new();
        let (query, normalized) =
            fts5_query_with_prefix_wildcard("what is the origin of COVID-19", true, &stopwords)
                .unwrap();

        assert_eq!(
            query,
            "\"what\" OR \"is\" OR \"the\" OR \"origin\" OR \"of\" OR \"COVID\" OR \"19\"*"
        );
        assert_eq!(normalized, "what is the origin of COVID 19");

        let (query, _) = fts5_query_with_prefix_wildcard("COVID ", true, &stopwords).unwrap();
        assert_eq!(query, "\"COVID\"");
    }

    #[test]
    fn fts5_query_filters_configured_stopwords() {
        let stopwords = test_stopwords(&["is", "of", "the"]);
        let (query, normalized) = fts5_query("what is the origin of COVID-19", &stopwords).unwrap();

        assert_eq!(query, "\"what\" OR \"origin\" OR \"COVID\" OR \"19\"");
        assert_eq!(normalized, "what origin COVID 19");
        assert!(fts5_query("is the of", &stopwords).is_none());
    }

    #[test]
    fn fts5_query_does_not_move_prefix_wildcard_before_final_stopword() {
        let stopwords = test_stopwords(&["the"]);
        let (query, normalized) =
            fts5_query_with_prefix_wildcard("COVID the", true, &stopwords).unwrap();

        assert_eq!(query, "\"COVID\"");
        assert_eq!(normalized, "COVID");
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
