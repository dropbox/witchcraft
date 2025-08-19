#![allow(dead_code)]

use csv;
use indicatif::ProgressBar;
use rusqlite::{Connection, OpenFlags, Result as SQLResult, Statement};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use once_cell::sync::Lazy;
use std::sync::RwLock;
mod quantized_t5;
pub mod assets;

mod packops;
use packops::TensorPackOps;

mod merger;
//pub mod qtensor;
//pub mod varbuilder;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use tokenizers::Tokenizer;

const EMBEDDING_DIM: usize = 128;

pub fn make_device() -> Device {
    if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap()
    } else {
        Device::Cpu
    }
}

fn kmeans(data: &Tensor, k: usize, max_iter: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let (m, n) = data.dims2()?;
    println!("kmeans k={} m={} n={}...", k, m, n);

    let total: u64 = (max_iter * k).try_into().unwrap();
    let bar = ProgressBar::new(total);

    let mut rng = rand::rng();
    let centroid_idx = rand::seq::index::sample(&mut rng, m, k).into_vec();
    let centroid_idx: Vec<u32> = centroid_idx.iter().map(|&i| i as u32).collect();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    let centroid_idx_tensor = centroid_idx_tensor.to_device(data.device())?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((m,), DType::U32, device)?;

    for _ in 0..max_iter {
        let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
        cluster_assignments = sim.argmax(D::Minus1)?;
        let mut centers_vec = vec![];
        for i in 0..k {
            let mut indices = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == i as u32 {
                        indices.push(j as u32);
                    }
                });
            if indices.len() > 0 {
                let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
                let indices = indices.to_device(data.device())?;
                let cluster_data = data.index_select(&indices, 0)?;
                let sum = cluster_data.sum(0)?;
                let normalized = sum.broadcast_div(&sum.sqr()?.sum_keepdim(0)?.sqrt()?);
                centers_vec.push(normalized?);
            } else {
                let idx = rand::seq::index::sample(&mut rng, m, 1).into_vec()[0];
                let center = data.get(idx)?;
                let normalized = center.broadcast_div(&center.sqr()?.sum_keepdim(0)?.sqrt()?);
                centers_vec.push(normalized?);
            }
            bar.inc(1);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    bar.finish();
    Ok((centers, cluster_assignments))
}

fn write_buckets(db: &DB, centers: &Tensor, device: &Device) -> Result<()> {
    let (k, _) = centers.dims2()?;
    println!("k={}", k);
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let embeddings_count = db
        .query("SELECT sum(length(embeddings)/?1) FROM chunk")
        .query_row((EMBEDDING_DIM,), |row| Ok(row.get::<_, u32>(0)?))
        .unwrap();
    assert!(embeddings_count > 0);
    let bar = ProgressBar::new(embeddings_count as u64);

    //let mut document_indices = vec![];
    let mut document_indices = Vec::<u32>::new();
    let mut all_hashes = vec![];
    let mut all_embeddings = vec![];

    let mut query = db.query(
        "SELECT document.rowid,chunk.hash,chunk.embeddings FROM document,chunk
        WHERE document.hash = chunk.hash
        ORDER BY document.rowid",
    );

    let mut results = query.query_map((), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Vec<u8>>(2)?,
        ))
    })?;

    let mut done = false;
    let mut batch = 0;
    let mut tmpfiles = vec![];
    let centers_cpu = centers.to_device(&Device::Cpu)?;
    while !done {
        match results.next() {
            Some(result) => {
                let (id, hash, embeddings) = result?;
                let t = Tensor::from_q8_bytes(&embeddings, EMBEDDING_DIM, &Device::Cpu)?
                    .dequantize(8)?
                    .l2_normalize()?;
                let split = split_tensor(&t);
                let m = split.len();
                for _ in 0..m {
                    document_indices.push(id as u32);
                }
                all_hashes.push(hash);
                all_embeddings.extend(split);
                batch += m;
            }
            None => {
                done = true;
            }
        }

        let batch_size = 0x10000;

        if batch >= batch_size || done {
            let now = std::time::Instant::now();

            let take = batch.min(batch_size);
            let left = batch - take;

            let embeddings = all_embeddings.split_off(left);
            let indices = document_indices.split_off(left);
            let data = Tensor::cat(&embeddings, 0)?.to_device(&device).unwrap();

            let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
            let cluster_assignments = sim.argmax(D::Minus1)?.to_device(&Device::Cpu)?;
            mmuls_total += now.elapsed().as_millis();

            let now = std::time::Instant::now();
            let mut writer = merger::Writer::new().unwrap();

            let mut pairs: Vec<(usize, u32)> = cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .map(|(i, &bucket)| (i, bucket))
                .collect();
            pairs.sort_by_key(|&(_, bucket)| bucket);

            let mut indices_bytes: Vec<u8> = Vec::with_capacity(take * 4);
            let mut residuals_bytes: Vec<u8> = Vec::with_capacity(take * 64);
            let (_, mut prev_bucket) = pairs[0];
            let mut count = 0;
            let mut bucket_done = false;

            for i in 0.. {
                let (sample, bucket) = if i < take {
                    pairs[i]
                } else {
                    bucket_done = true;
                    (0, std::u32::MAX)
                };

                if (bucket != prev_bucket || bucket_done) && count > 0 {
                    assert!(prev_bucket < bucket);
                    writer.write_record(prev_bucket, count, &indices_bytes, &residuals_bytes)?;

                    indices_bytes.clear();
                    residuals_bytes.clear();
                    prev_bucket = bucket;
                    count = 0;
                }

                if bucket_done {
                    break;
                }

                let doc_idx = indices[sample];
                indices_bytes.extend_from_slice(&doc_idx.to_ne_bytes());
                let center = centers_cpu.get(bucket as usize)?;
                let residual = (embeddings[sample].get(0) - &center)?;
                let residual_quantized = residual.compand()?.quantize(4)?.to_q4_bytes()?;
                residuals_bytes.extend(&residual_quantized);
                count += 1;
            }
            tmpfiles.push(writer.finish()?);
            writes_total += now.elapsed().as_millis();
            bar.inc(take as u64);

            batch = left;
        }
    }
    bar.finish();

    println!("mmuls took {} ms.", mmuls_total);
    println!("writes took {} ms.", writes_total);
    println!("merge all");

    db.begin_transaction().unwrap();

    let max_generation = db
        .query("SELECT max(generation) FROM indexed_chunk")
        .query_row((), |row| Ok(row.get::<_, u32>(0)?))
        .unwrap_or(0);
    let next_generation = max_generation + 1;

    let mut merger = merger::Merger::from_tempfiles(tmpfiles)?;
    for result in &mut merger {
        let entry = result?;
        let center = centers_cpu.get(entry.value as usize)?;
        let center_bytes = center.to_f32_bytes()?;
        db.add_bucket(
            entry.value,
            next_generation,
            &center_bytes,
            &entry.tags,
            &entry.data,
        )
        .unwrap();
    }
    println!("write {} hashes to indexed_chunk", all_hashes.len());
    for hash in all_hashes {
        db.add_indexed_chunk(&hash, next_generation).unwrap();
    }

    db.query("DELETE FROM bucket WHERE generation = ?1")
        .execute((max_generation,))
        .unwrap();
    db.query("DELETE FROM indexed_chunk WHERE generation = ?1")
        .execute((max_generation,))
        .unwrap();
    db.commit_transaction().unwrap();

    Ok(())
}

fn fulltext_search(db: &DB, q: &String, sql_filter: Option<&str>) -> Result<Vec<u32>> {
    let mut fts_idxs = vec![];

    let sql = format!(
        "SELECT document.rowid,bm25(document_fts) AS score
        FROM document,document_fts
        WHERE document.rowid == document_fts.rowid
        AND document_fts MATCH ?1
        {}
        ORDER BY score",
        match sql_filter {
            Some(filter) => format!("AND {filter}"),
            _ => String::new(),
        }
    );
    let mut query = db.query(&sql);

    let q: String = q
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    let results = query.query_map((&q,), |row| Ok((row.get::<_, u32>(0)?,)))?;
    for result in results {
        let (rowid,) = result?;
        fts_idxs.push(rowid);
    }
    println!("full text found {} matches", fts_idxs.len());
    Ok(fts_idxs)
}

fn reciprocal_rank_fusion(list1: &[u32], list2: &[u32], k: f64) -> Vec<u32> {
    let mut scores: HashMap<u32, f64> = HashMap::new();

    for (rank, &doc_id) in list1.iter().enumerate() {
        let score = 1.0 / (k + rank as f64);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }

    for (rank, &doc_id) in list2.iter().enumerate() {
        let score = 1.0 / (k + rank as f64);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }

    let mut results: Vec<(u32, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort descending by score
    let results: Vec<u32> = results.iter().map(|&(idx, _)| idx).collect();
    results
}

struct CachedTensor {
    version: u64,
    cluster_ids: Vec<u32>,
    sizes: Vec<usize>,
    centers: Vec<Tensor>,
    tensor: Tensor,
}

static CACHED: Lazy<RwLock<Option<CachedTensor>>> = Lazy::new(|| RwLock::new(None));

fn get_centers(db: &DB, device: &Device, version: u64) -> Result<(Vec<u32>, Vec<usize>, Vec<Tensor>, Tensor)> {
    let now = std::time::Instant::now();
    {
        let cache = CACHED.read().unwrap();
        if let Some(cached) = &*cache {
            if cached.version == version {
                return Ok((cached.cluster_ids.clone(), cached.sizes.clone(), cached.centers.clone(), cached.tensor.clone()));
            }
        }
    }

    let mut center_query =
        db.query("SELECT id,length(indices)/4,center FROM bucket ORDER BY id");
    let mut cluster_ids = vec![];
    let mut sizes = vec![];
    let mut centers = vec![];
    for result in center_query.query_map((), |row| {
        let id = row.get(0)?;
        let size = row.get(1)?;
        let blob: Vec<u8> = row.get(2)?;
        Ok((id, size, blob))
    })? {
        let (id, size, center) = result?;
        cluster_ids.push(id);
        sizes.push(size);
        let t = Tensor::from_f32_bytes(&center, EMBEDDING_DIM, &Device::Cpu)?.flatten_all()?;
        centers.push(t);
    }
    let tensor = if centers.len() > 0 {
        Tensor::stack(&centers, 0)?.to_device(&device)?
    } else {
        Tensor::zeros(&[0, EMBEDDING_DIM], DType::F32, &device)?
    };
    println!(
        "reading and stacking centers took {} ms (caching result for future queries.)",
        now.elapsed().as_millis()
    );

    let mut cache = CACHED.write().unwrap();
    *cache = Some(CachedTensor {
        version: version,
        cluster_ids: cluster_ids.clone(),
        sizes: sizes.clone(),
        centers: centers.clone(),
        tensor: tensor.clone(),
    });

    Ok((cluster_ids, sizes, centers, tensor))
}

fn match_centroids(
    db: &DB,
    query_embeddings: &Tensor,
    cutoff: f32,
    top_k: usize,
    sql_filter: Option<&str>,
) -> Result<Vec<(f32, u32)>> {
    let max_generation = db
        .query("SELECT MAX(generation) FROM indexed_chunk")
        .query_row((), |row| Ok(row.get::<_, u32>(0)?))
        .unwrap_or(0);

    let k = 32;
    let t_prime = 40000;
    let device = query_embeddings.device();
    let (m, n) = query_embeddings.dims2()?;


    let mut all_document_embeddings = vec![];
    let mut all = vec![];
    let mut count = 0;
    let mut missing = vec![];

    let (cluster_ids, sizes, centers, centers_matrix) = get_centers(&db, &device, max_generation as u64)?;

    if centers.len() > 0 {
        let now = std::time::Instant::now();

        let query_centroid_similarity = query_embeddings
            .matmul(&centers_matrix.transpose(D::Minus1, D::Minus2)?)
            .unwrap();
        let query_centroid_similarity = query_centroid_similarity.to_device(&Device::Cpu)?;

        let sorted_indices = query_centroid_similarity.arg_sort_last_dim(false)?;
        let (m, n) = sorted_indices.dims2()?;

        let mut topk_clusters = vec![];
        for i in 0..m {
            let row = sorted_indices.get(i)?;
            let row_scores_sorted = query_centroid_similarity.get(i)?.gather(&row, D::Minus1)?;
            let mut cumsum = 0;
            let mut score = 0.0f32;
            for j in 0..n {
                let idx = row.get(j)?.to_scalar::<u32>()?;
                if j < k {
                    topk_clusters.push(idx);
                }
                let size = sizes[idx as usize];
                if cumsum < t_prime {
                    score = row_scores_sorted.get(j)?.to_scalar::<f32>()?;
                }
                cumsum += size;

                if j >= k || cumsum >= t_prime {
                    break;
                }
            }
            missing.push(score);
        }
        topk_clusters.sort();
        topk_clusters.dedup();
        println!(
            "finding top-{} out of {} clusters took {} ms.",
            topk_clusters.len(),
            n,
            now.elapsed().as_millis()
        );

        let now = std::time::Instant::now();

        db.execute("CREATE TEMPORARY TABLE temp(id INTEGER)")
            .unwrap();
        let mut bucket_query = db.query(
            "SELECT bucket.id,indices,residuals FROM bucket
                JOIN temp ON bucket.id = temp.id
                WHERE generation = ?1",
        );
        let mut insert_temp_query = db.query("INSERT INTO TEMP VALUES(?1)");

        for i in topk_clusters {
            insert_temp_query
                .execute((cluster_ids[i as usize],))
                .unwrap();
        }

        let results = bucket_query.query_map((max_generation,), |row| {
            Ok((
                row.get::<_, u32>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?;

        for result in results {
            let (cluster_id, document_indices, document_embeddings) = result?;

            let center = &centers[cluster_id as usize];
            let document_indices = u8_to_vec_u32(&document_indices);

            //let residuals = Tensor::from_q4_bytes(&document_embeddings, EMBEDDING_DIM, &device)?.dequantize(4)?.inv_compand()?;
            let residuals =
                Tensor::from_companded_q4_bytes(&document_embeddings, EMBEDDING_DIM, &Device::Cpu)?;
            let embeddings = residuals.broadcast_add(&center)?;
            all_document_embeddings.push(embeddings);

            let (m, _) = residuals.dims2()?;
            for j in 0..m {
                all.push((document_indices[j], count));
                count += 1;
            }
        }
        db.execute("DROP TABLE temp").unwrap();

        println!(
            "reading in {} indexed embeddings took {} ms.",
            count,
            now.elapsed().as_millis()
        );
    } else {
        for _ in 0..m {
            missing.push(0.0);
        }
    }

    let missing_similarities = Tensor::from_vec(missing, m, &Device::Cpu)?;

    let now = std::time::Instant::now();
    let mut num_unindexed = 0;
    let mut unindexed_chunks_query = db.query(
        "
        SELECT d.rowid, c.hash, c.embeddings
        FROM chunk c
        JOIN document d ON c.hash = d.hash
        LEFT JOIN indexed_chunk ic ON ic.generation = ?1 AND ic.hash = c.hash
        WHERE ic.hash IS NULL",
    );

    let results = unindexed_chunks_query.query_map((max_generation,), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Vec<u8>>(2)?,
        ))
    })?;
    for result in results {
        let (id, _hash, embeddings) = result?;
        let embeddings = Tensor::from_q8_bytes(&embeddings, EMBEDDING_DIM, &Device::Cpu)?
            .dequantize(8)?
            .l2_normalize()?;
        let (m, _) = embeddings.dims2()?;
        all_document_embeddings.push(embeddings);
        for _ in 0..m {
            all.push((id, count));
            count += 1;
        }
        num_unindexed += m;
    }
    println!(
        "reading {} unindexed embeddings took {} ms.",
        num_unindexed,
        now.elapsed().as_millis()
    );

    if all_document_embeddings.len() == 0 {
        return Ok([].to_vec());
    }

    let all_document_embeddings = Tensor::cat(&all_document_embeddings, 0).unwrap();
    let all_document_embeddings = all_document_embeddings.to_device(query_embeddings.device())?;

    let now = std::time::Instant::now();
    let sim = query_embeddings
        .matmul(&all_document_embeddings.t()?)?
        .transpose(0, 1)
        .unwrap();
    let sim = sim.to_device(&Device::Cpu)?;

    println!("sim mmul took {} ms.", now.elapsed().as_millis());
    let now = std::time::Instant::now();

    all.sort();

    let mut current = Tensor::zeros((n,), DType::F32, &Device::Cpu)?;

    let mut unique_docs = 0;

    let mut prev_idx = 0;

    db.execute("CREATE TEMPORARY TABLE temp2(rowid INTEGER PRIMARY KEY, score FLOAT)").unwrap();
    let mut insert_temp_query = db.query("INSERT INTO temp2 VALUES(?1, ?2)");

    for i in 0.. {
        let is_last = i == all.len() - 1;
        let (idx, pos) = all[i];
        if i > 0 && (prev_idx != idx || is_last) {
            unique_docs += 1;
            let score = current.mean(0)?.to_scalar::<f32>()?;
            if score >= cutoff {
                insert_temp_query.execute((prev_idx, score)).unwrap();
            }
        }

        if is_last {
            break;
        }

        let row = sim.get(pos)?;
        current = if idx == prev_idx {
            row.maximum(&current)?
        } else {
            row.maximum(&missing_similarities)?
        };

        prev_idx = idx;
    }
    println!(
        "scoring {} documents took {} ms.",
        unique_docs,
        now.elapsed().as_millis()
    );

    let sql = format!(
        "SELECT score,document.rowid
        FROM document,temp2
        WHERE document.rowid = temp2.rowid
        {}
        ORDER BY score DESC
        LIMIT ?1",
        match sql_filter {
            Some(filter) => format!("AND {filter}"),
            _ => String::new(),
        }
    );

    let mut scored_documents_query = db.query(&sql);
    let results = scored_documents_query
        .query_map((top_k,), |row| {
            Ok((row.get::<_, f32>(0)?, row.get::<_, u32>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    db.execute("DROP TABLE temp2").unwrap();
    Ok(results)
}

fn split_tensor(tensor: &Tensor) -> Vec<Tensor> {
    let dims = tensor.dims();
    let num_rows = dims[0];

    // Collect each row as a separate Tensor of shape [EMBEDDING_DIM]
    (0..num_rows)
        .map(|i| {
            let row_tensor = tensor.i(i).unwrap();
            row_tensor.unsqueeze(0).unwrap()
        })
        .collect()
}

pub fn scan_documents_dir(db: &DB, dirname: &str) -> Result<()> {
    println!("register documents...");
    let mut entries: Vec<_> = fs::read_dir(dirname)?
        .filter_map(Result::ok) // Filter out errors
        .filter(|e| e.path().is_file()) // Only files
        .collect();

    // Sort entries by file name
    entries.sort_by_key(|e| e.file_name());
    let bar = ProgressBar::new(entries.len() as u64);

    for entry in entries {
        let path = entry.path();
        let filename = path.to_str().unwrap();
        let body = fs::read_to_string(filename)?;

        let mut hasher = Sha256::new();
        hasher.update(&body);
        let hash = format!("{:x}", hasher.finalize());

        db.add_doc(&filename, &hash, &body).unwrap();
        bar.inc(1);
    }
    bar.finish();

    Ok(())
}

#[derive(Debug, Deserialize)]
struct Record {
    name: String,
    body: String,
}

pub fn read_csv(db: &DB, csvname: &str) -> Result<()> {
    println!("register documents from CSV {}...", csvname);

    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    for result in rdr.deserialize() {
        let record: Record = result?;
        let metadata = json!({ "key": record.name }).to_string();
        let body = record.body;

        let mut hasher = Sha256::new();
        hasher.update(&body);
        let hash = format!("{:x}", hasher.finalize());
        db.add_doc(&metadata, &hash, &body).unwrap();
    }

    Ok(())
}

pub fn add_doc_from_file(db: &DB, filename: &str) -> Result<()> {
    let body = fs::read_to_string(filename)?;
    let mut hasher = Sha256::new();
    hasher.update(&body);
    let hash = format!("{:x}", hasher.finalize());
    let metadata = json!({ "filename": filename }).to_string();
    db.add_doc(&metadata, &hash, &body).unwrap();
    Ok(())
}

pub fn add_doc_from_string(db: &DB, metadata: &str, body: &str) -> Result<()> {
    let mut hasher = Sha256::new();
    hasher.update(&body);
    let hash = format!("{:x}", hasher.finalize());
    db.add_doc(metadata, &hash, &body).unwrap();
    Ok(())
}

pub struct Embedder {
    tokenizer: Tokenizer,
    //model: t5::T5EncoderModel,
    model: quantized_t5::T5EncoderModel,
}

impl Embedder {
    pub fn new(device: &Device) -> Self {
        let (builder, tokenizer) = quantized_t5::T5ModelBuilder::load().unwrap();
        let model = builder.build_encoder(&device).unwrap();
        Self { tokenizer, model }
    }
    fn embed(self: &Self, text: &str) -> Result<Tensor> {
        //let now = std::time::Instant::now();
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        //println!("embedder took {} ms.", now.elapsed().as_millis());
        normalize_l2(&embeddings)
    }
}

pub struct Gatherer<'a> {
    documents: Box<dyn Iterator<Item = (String, String)> + 'a>,
    embedder: &'a Embedder,
}

impl<'a> Gatherer<'a> {
    fn new(stmt: &'a mut Statement, embedder: &'a Embedder) -> Self {
        let documents = Box::new(
            stmt
                .query_map((), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .unwrap()
                .map(Result::unwrap),
        );

        Self {
            documents,
            embedder,
        }
    }
}

impl<'a> Iterator for Gatherer<'a> {
    type Item = (String, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let mut doc_embedding = vec![];
        match self.documents.next() {
            Some((hash, body)) => {

                let now = std::time::Instant::now();
                let embeddings = self.embedder.embed(&body).unwrap().to_device(&Device::Cpu).unwrap();

                let (_b, m, _n) = embeddings.dims3().unwrap();
                let dt = now.elapsed().as_secs_f64();
                println!("embedder took {} ms ({} rows/s).", now.elapsed().as_millis(), ((m as f64) / dt).round());

                let split = split_tensor(&embeddings.get(0).ok()?);
                doc_embedding.extend(split);
                Some((hash, Tensor::cat(&doc_embedding, 0).unwrap()))
            }
            None => None,
        }
    }
}

pub struct DB {
    connection: Connection,
}

impl DB {
    pub fn new_reader(db_fn: &str) -> Self {
        let connection =
            Connection::open_with_flags(db_fn, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        Self { connection }
    }

    pub fn new(db_fn: &str) -> Self {
        let connection = Connection::open(db_fn).unwrap();

        //connection
            //.pragma_update(None, "journal_mode", &"WAL")
            //.unwrap();
        connection.busy_timeout(std::time::Duration::from_secs(5)).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS document(metadata JSON,
            hash TEXT NOT NULL, body TEXT, UNIQUE(metadata, hash))";
        connection.execute(query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS document_index ON document(hash)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(body, content='document', content_rowid='rowid')";
        connection.execute(query, ()).unwrap();

        let query = "INSERT INTO document_fts(document_fts) VALUES('rebuild')";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY NOT NULL, embeddings BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS chunk_index ON chunk(hash)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS bucket(id INTEGER PRIMARY KEY,
            generation INTEGER NOT NULL,
            center BLOB NOT NULL, indices BLOB NOT NULL, residuals BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS bucket_index ON bucket(generation, id)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS indexed_chunk(hash TEXT PRIMARY KEY NOT NULL, generation INTEGER NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query =
            "CREATE INDEX IF NOT EXISTS indexed_chunk_index ON indexed_chunk(generation, hash)";
        connection.execute(query, ()).unwrap();

        Self { connection }
    }

    fn execute(self: &Self, sql: &str) -> SQLResult<()> {
        self.connection.execute(sql, ()).unwrap();
        Ok(())
    }

    fn query(self: &Self, sql: &str) -> Statement<'_> {
        self.connection.prepare(&sql).unwrap()
    }

    fn begin_transaction(&self) -> SQLResult<()> {
        self.connection.execute("BEGIN", ()).unwrap();
        Ok(())
    }

    fn commit_transaction(&self) -> SQLResult<()> {
        self.connection.execute("COMMIT", ()).unwrap();
        Ok(())
    }

    fn add_doc(self: &Self, metadata: &str, hash: &str, body: &str) -> SQLResult<()> {
        self.connection.execute(
            "INSERT OR IGNORE INTO document VALUES(?1, ?2, ?3)",
            (&metadata, &hash, &body),
        )?;
        Ok(())
    }

    fn add_chunk(self: &Self, hash: &str, embeddings: &Vec<u8>) -> SQLResult<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO chunk VALUES(?1, ?2)",
                (&hash, embeddings),
            )
            .unwrap();
        Ok(())
    }

    fn add_bucket(
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

    fn add_indexed_chunk(self: &Self, hash: &str, generation: u32) -> SQLResult<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO indexed_chunk VALUES(?1, ?2)",
                (hash, generation),
            )
            .unwrap();
        Ok(())
    }
}

fn u8_to_vec_u32(bytes: &[u8]) -> Vec<u32> {
    let u32_size = size_of::<u32>();

    assert!(bytes.len() % u32_size == 0);
    let total_u32s = bytes.len() / u32_size;

    let mut u32s = Vec::with_capacity(total_u32s);
    for chunk in bytes.chunks_exact(u32_size) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        u32s.push(u32::from_ne_bytes(arr));
    }
    u32s
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}

pub fn embed_chunks(db: &DB, device: &Device) -> Result<()> {
    let embedder = Embedder::new(&device);

    let mut query = db
        .query(
            "SELECT
        document.hash,document.body
        FROM document
        LEFT JOIN chunk ON document.hash = chunk.hash
        WHERE chunk.hash IS NULL
        ORDER BY document.hash",
        );

    let embedding_iter = Gatherer::new(&mut query, &embedder);
    for (hash, embeddings) in embedding_iter {
        println!(
            "got embedding for chunk with hash {} {:?}",
            hash,
            embeddings.dims2().unwrap()
        );

        //let now = std::time::Instant::now();
        let bytes = embeddings.stretch_rows()?.quantize(8)?.to_q8_bytes()?;
        //println!("quantization took {} ms.", now.elapsed().as_millis());

        //let now = std::time::Instant::now();
        db.add_chunk(&hash, &bytes).unwrap();
        //println!("database insert took {} ms.", now.elapsed().as_millis());
    }
    Ok(())
}

pub fn count_unindexed_chunks(db: &DB) -> Result<usize> {
    let mut unindexed_chunks_query = db.query(
        "
        SELECT IFNULL(SUM(LENGTH(c.embeddings)/128), 0)
        FROM chunk c
        JOIN document d ON c.hash = d.hash
        LEFT JOIN indexed_chunk ic ON ic.generation =
            (SELECT MAX(generation) FROM indexed_chunk) AND ic.hash = c.hash
        WHERE ic.hash IS NULL",
    );

    let count = unindexed_chunks_query.query_row((), |row| {
        Ok( row.get::<_, usize>(0)? )
    })?;
    Ok(count)
}

pub fn index_chunks(db: &DB, device: &Device) -> Result<()> {
    let mut kmeans_query = db.query("SELECT chunk.embeddings FROM chunk");
    let mut total_embeddings = 0;
    let mut rng = rand::rng();
    let mut all_embeddings = vec![];
    println!("read embeddings...");
    for embeddings in kmeans_query.query_map((), |row| Ok(row.get::<_, Vec<u8>>(0)?))? {
        let t = Tensor::from_q8_bytes(&embeddings?, EMBEDDING_DIM, &Device::Cpu)?;
        let (m, _) = t.dims2()?;
        let k = ((m as f32).sqrt().ceil()) as usize;
        let subset_idx = rand::seq::index::sample(&mut rng, m, k).into_vec();
        for i in subset_idx {
            let row = t.get(i as usize)?;
            all_embeddings.push(row);
        }
        total_embeddings += m;
    }
    let matrix = Tensor::stack(&all_embeddings, 0)?
        .to_device(&device)?
        .dequantize(8)?
        .l2_normalize()?;

    let now = std::time::Instant::now();
    let log2_k = (16.0 * (total_embeddings as f64).sqrt()).log(2.0).floor() as u32;
    let mut k = 1 << log2_k;
    println!("total_embeddings={} k={}", total_embeddings, k);
    let (m, _) = matrix.dims2()?;
    if m < k {
        k = m / 4;
    }
    let (centers, idxs) = kmeans(&matrix, k as usize, 5, &device)?;
    println!("kmeans took {} ms.", now.elapsed().as_millis());
    println!("idxs {}", idxs);

    println!("write buckets...");
    let now = std::time::Instant::now();
    write_buckets(&db, &centers, &device).unwrap();
    println!("write buckets took {} ms.", now.elapsed().as_millis());
    Ok(())
}

use lru::LruCache;
use std::num::NonZeroUsize;

pub struct EmbeddingsCache {
    cache: LruCache<String, Tensor>,
}

impl EmbeddingsCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self { cache: LruCache::new(cap) }
    }

    pub fn get(&mut self, key: &String) -> Option<Tensor> {
        self.cache.get(key).cloned()
    }

    pub fn put(&mut self, key: &String, value: &Tensor) {
        self.cache.put(key.into(), value.clone());
    }
}

pub fn search(
    db: &DB,
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    q: &String,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&str>,
) -> Result<Vec<(String, String)>> {
    let now = std::time::Instant::now();

    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");

    let fts_idxs = if use_fulltext {
        println!("Doing full text search for: `{}'", q);
        fulltext_search(&db, &q, sql_filter)?
    } else {
        [].to_vec()
    };

    println!("Doing semantic search for: `{}'", q);

    let qe = match cache.get(&q) {
        Some(existing) => {
            println!("found cached query");
            existing
        }
        None => {
            let qe = embedder.embed(&q)?.get(0)?;
            cache.put(&q, &qe);
            qe
        }
    };

    let sem_matches = match_centroids(&db, &qe, threshold, top_k, sql_filter).unwrap();
    let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx)| idx).collect();
    println!("semantic search found {} matches", sem_idxs.len());

    let fused = if use_fulltext {
        reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
    } else {
        sem_idxs
    };

    let mut results = vec![];
    let mut body_query = db.query("SELECT metadata,body FROM document WHERE rowid = ?1");
    for idx in fused {
        let (metadata, body) = body_query.query_row((idx,), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        results.push((metadata, body));
    }
    println!("search took {} ms end-to-end.", now.elapsed().as_millis());
    Ok(results)
}

pub fn score_query_sentences(
    embedder: &Embedder,
    cache: &mut EmbeddingsCache,
    q: &String,
    sentences: &[String],
) -> Result<Vec<f32>> {

    let now = std::time::Instant::now();
    let qe = match cache.get(&q) {
        Some(existing) => {
            existing
        }
        None => {
            let qe = embedder.embed(&q)?.get(0)?;
            cache.put(&q, &qe);
            qe
        }
    };
    let mut sizes = vec![];
    let mut ses = vec![];
    for s in sentences.iter() {
        let se = embedder.embed(&s)?.get(0)?;
        let split = split_tensor(&se);
        sizes.push(split.len());
        ses.extend(split);
    }
    let ses = Tensor::cat(&ses, 0)?;
    let sim = ses.matmul(&qe.transpose(D::Minus1, D::Minus2)?)?;
    let sim = sim.to_device(&Device::Cpu)?;

    let mut scores = vec![];
    let mut i = 0;
    for sz in sizes.iter() {
        let sz = *sz;
        let mut max = sim.get(i)?;
        for j in 1usize..sz {
            let row = sim.get(i + j)?;
            max = max.maximum(&row)?;
        }
        scores.push(max.mean(0)?.to_scalar::<f32>()?);
        i += sz;
    }
    for i in 0usize..sentences.len() {
        println!("warp sentence score {} {}", sentences[i], scores[i]);
    }
    println!("scoring {} sentences took {} ms.", sentences.len(), now.elapsed().as_millis());
    Ok(scores)
}

pub fn bulk_search(
    db: &DB,
    embedder: &Embedder,
    csvname: &String,
    outputname: &String,
    use_fulltext: bool,
    use_semantic: bool,
) -> Result<()> {
    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let file = File::create(outputname).unwrap();
    let mut writer = BufWriter::new(file);

    let mut filename_query = db.query("SELECT metadata FROM document WHERE rowid = ?1");

    for result in rdr.deserialize() {
        let record: (String, String) = result?;
        let key = record.0;
        let question = record.1;

        println!("\nSearching for: {}", question);
        let fts_idxs = if use_fulltext {
            fulltext_search(&db, &question, None)?
        } else {
            [].to_vec()
        };

        let sem_matches = if use_semantic {
            let qe = embedder.embed(&question)?.get(0)?;
            match_centroids(&db, &qe, 0.0, 100, None).unwrap()
        } else {
            [].to_vec()
        };
        let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx)| idx).collect();
        if use_semantic {
            println!("semantic search found {} matches", sem_idxs.len());
        }

        let fused = if use_fulltext && use_semantic {
            reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
        } else if use_fulltext {
            fts_idxs
        } else {
            sem_idxs
        };

        let mut filenames = vec![];
        for idx in fused {
            let filename = filename_query.query_row((idx,), |row| Ok(row.get::<_, String>(0)?))?;
            filenames.push(filename);
        }

        write!(writer, "{}\t", key).unwrap();
        for filename in &filenames {
            write!(writer, "{},", filename).unwrap();
        }
        write!(writer, "\n").unwrap();
        writer.flush().unwrap();
    }
    Ok(())
}
