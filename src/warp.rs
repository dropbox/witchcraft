use log::{debug, info, warn};
use once_cell::sync::Lazy;
use rusqlite::Statement;
use std::collections::HashMap;
use std::sync::RwLock;
//mod t5;
//use t5 as t5_encoder;

mod quantized_t5;
use quantized_t5 as t5_encoder;

mod db;
pub use db::DB;

mod embedder;
pub use embedder::Embedder;

pub mod assets;

mod packops;
use packops::TensorPackOps;

mod merger;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};

const EMBEDDING_DIM: usize = 128;

pub fn make_device() -> Device {
    if cfg!(target_os = "macos") {
        match Device::new_metal(0) {
            Ok(device) => device,
            Err(v) => {
                warn!("unable to create metal device: {v}");
                Device::Cpu
            }
        }
    } else {
        Device::Cpu
    }
}

#[cfg(feature = "progress")]
pub mod progress {
    use indicatif::ProgressBar;
    pub struct Bar(pub ProgressBar);
    pub fn new(len: u64) -> Bar {
        let pb = ProgressBar::new(len);
        Bar(pb)
    }
    impl Bar {
        pub fn inc(&self, n: u64) {
            self.0.inc(n);
        }
        pub fn finish(&self) {
            self.0.finish();
        }
    }
}

#[cfg(not(feature = "progress"))]
pub mod progress {
    #[derive(Clone, Copy)]
    pub struct Bar;
    pub fn new(_len: u64) -> Bar {
        Bar
    }
    impl Bar {
        pub fn inc(&self, _n: u64) {}
        pub fn finish(&self) {}
    }
}

fn kmeans(data: &Tensor, k: usize, max_iter: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let (m, n) = data.dims2()?;
    debug!("kmeans k={} m={} n={}...", k, m, n);

    let total: u64 = (max_iter * k).try_into()?;
    let bar = progress::new(total);

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

fn compress_keys(keys: &[(u32, u32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(keys.len() * 8);
    let mut iter = keys.iter();

    // Store first key as-is
    if let Some(&(major, minor)) = iter.next() {
        bytes.extend_from_slice(&major.to_ne_bytes());
        bytes.extend_from_slice(&minor.to_ne_bytes());

        let mut base = (major, minor);

        for &(major, minor) in iter {
            let delta_major = major - base.0;
            let delta_minor = if delta_major == 0 {
                minor - base.1
            } else {
                minor
            };

            bytes.extend_from_slice(&delta_major.to_ne_bytes());
            bytes.extend_from_slice(&delta_minor.to_ne_bytes());

            base = (major, minor);
        }
    }

    lz4_flex::block::compress_prepend_size(&bytes)
}

fn decompress_keys(bytes: &[u8]) -> Result<Vec<(u32, u32)>> {
    let decompressed = lz4_flex::block::decompress_size_prepended(bytes)?;
    let mut keys = Vec::with_capacity(decompressed.len() / 8);
    let mut chunks = decompressed.chunks_exact(8);

    // decode first key as absolute
    if let Some(chunk) = chunks.next() {
        let major = u32::from_ne_bytes(chunk[0..4].try_into()?);
        let minor = u32::from_ne_bytes(chunk[4..8].try_into()?);

        let mut base = (major, minor);
        keys.push(base);

        for chunk in chunks {
            let delta_major = u32::from_ne_bytes(chunk[0..4].try_into()?);
            let delta_minor = u32::from_ne_bytes(chunk[4..8].try_into()?);

            let major = base.0.wrapping_add(delta_major);
            let minor = if delta_major == 0 {
                base.1.wrapping_add(delta_minor)
            } else {
                delta_minor
            };

            base = (major, minor);
            keys.push(base);
        }
    }

    Ok(keys)
}

fn write_buckets(db: &DB, centers: &Tensor, device: &Device) -> Result<()> {
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let embeddings_count = db
        .query("SELECT sum(length(embeddings)/?1) FROM chunk")?
        .query_row((EMBEDDING_DIM,), |row| Ok(row.get::<_, u32>(0)?))?;
    assert!(embeddings_count > 0);
    let bar = progress::new(embeddings_count as u64);

    let mut document_indices = Vec::<(u32, u32)>::new();
    let mut all_chunkids = vec![];
    let mut all_embeddings = vec![];

    let mut query = db.query(
        "SELECT document.rowid,chunk.rowid,chunk.embeddings,chunk.counts FROM document,chunk
        WHERE document.hash = chunk.hash
        ORDER BY document.rowid",
    )?;

    let mut results = query.query_map((), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, u32>(1)?,
            row.get::<_, Vec<u8>>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;

    let mut done = false;
    let mut batch = 0;
    let mut tmpfiles = vec![];
    let centers_cpu = centers.to_device(&Device::Cpu)?;
    while !done {
        match results.next() {
            Some(result) => {
                let (id, chunkid, embeddings, counts) = result?;

                let t = Tensor::from_q8_bytes(&embeddings, EMBEDDING_DIM, &Device::Cpu)?
                    .dequantize(8)?
                    .l2_normalize()?;
                let split = split_tensor(&t);
                let m = split.len();

                for (i, count) in counts
                    .split(',')
                    .filter_map(|s| s.parse::<u32>().ok()).enumerate() {
                        for _ in 0..count {
                            document_indices.push((id, i as u32));
                        }
                    }
                all_chunkids.push(chunkid);
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
            let data = Tensor::cat(&embeddings, 0)?.to_device(&device)?;

            let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
            let cluster_assignments = sim.argmax(D::Minus1)?.to_device(&Device::Cpu)?;
            mmuls_total += now.elapsed().as_millis();

            let now = std::time::Instant::now();
            let mut writer = merger::Writer::new()?;

            let mut pairs: Vec<(usize, u32)> = cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .map(|(i, &bucket)| (i, bucket))
                .collect();
            pairs.sort_by_key(|&(_, bucket)| bucket);

            let mut keys: Vec<(u32, u32)> = Vec::with_capacity(take);
            let mut residuals_bytes: Vec<u8> = Vec::with_capacity(take * 64);
            let (_, mut prev_bucket) = pairs[0];
            let mut bucket_done = false;

            for i in 0.. {
                let (sample, bucket) = if i < take {
                    pairs[i]
                } else {
                    bucket_done = true;
                    (0, std::u32::MAX)
                };

                if (bucket != prev_bucket || bucket_done) && keys.len() > 0 {
                    assert!(prev_bucket < bucket);
                    writer.write_record(prev_bucket, &keys, &residuals_bytes)?;

                    keys.clear();
                    residuals_bytes.clear();
                    prev_bucket = bucket;
                }

                if bucket_done {
                    break;
                }

                keys.push(indices[sample]);

                let center = centers_cpu.get(bucket as usize)?;
                let residual = (embeddings[sample].get(0) - &center)?;
                let residual_quantized = residual.compand()?.quantize(4)?.to_q4_bytes()?;
                residuals_bytes.extend(&residual_quantized);
            }
            tmpfiles.push(writer.finish()?);
            writes_total += now.elapsed().as_millis();
            bar.inc(take as u64);

            batch = left;
        }
    }
    bar.finish();

    debug!("mmuls took {} ms.", mmuls_total);
    debug!("writes took {} ms.", writes_total);

    let txstatus = match db.begin_transaction() {
        Ok(()) => {
            let max_generation = db
                .query("SELECT max(generation) FROM indexed_chunk")?
                .query_row((), |row| Ok(row.get::<_, u32>(0)?))
                .unwrap_or(0);
            let next_generation = max_generation + 1;

            let mut merger = merger::Merger::from_tempfiles(tmpfiles)?;
            for result in &mut merger {
                let entry = result?;
                let center = centers_cpu.get(entry.value as usize)?;
                let center_bytes = center.to_f32_bytes()?;
                let compressed_keys = compress_keys(&entry.keys);
                db.add_bucket(
                    entry.value,
                    next_generation,
                    &center_bytes,
                    &compressed_keys,
                    &entry.data,
                )?;
            }
            info!("write {} chunk ids to indexed_chunk", all_chunkids.len());
            for chunkid in all_chunkids {
                let _ = db.add_indexed_chunk(chunkid, next_generation)?;
            }

            db.query("DELETE FROM bucket WHERE generation <= ?1")?
                .execute((max_generation,))?;
            db.query("DELETE FROM indexed_chunk WHERE generation <= ?1")?
                .execute((max_generation,))?;
            Ok(())
        }
        Err(v) => {
            warn!("unable to begin transaction, index not created/updated! {v}");
            Err(v)
        }
    };
    match txstatus {
        Ok(()) => Ok(db.commit_transaction()?),
        Err(v) => {
            warn!("failure during indexing transaction {v}");
            let _ = db.rollback_transaction();
            Err(v.into())
        }
    }
}

fn bm25_to_cosine_approx(bm25: f32) -> f32 {
    1.0 - (1.0 / (1.0 + (-bm25 / 2.0).exp())) // logistic-shaped
}

pub fn fulltext_search(
    db: &DB,
    q: &String,
    top_k: usize,
    sql_filter: Option<&str>,
) -> Result<Vec<(f32, u32, String)>> {
    let mut fts_matches = vec![];

    let mut last_is_space = false;
    for c in q.chars() {
        last_is_space = c == ' ';
    }
    let q: String = q
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    let filter = match sql_filter {
        Some(filter) => format!("AND {filter}"),
        _ => String::new(),
    };

    let sql = if q.len() > 0 {
        format!(
            "SELECT document.rowid, snippet(document_fts,-1, '<em>','</em>','...',15),
            bm25(document_fts) AS score
            FROM document,document_fts
            WHERE document.rowid = document_fts.rowid
            AND document_fts MATCH ?1 {filter}
            ORDER BY score,date DESC
            LIMIT ?2",
        )
    } else {
        format!(
            "SELECT rowid,\"\",0.0
            FROM document
            WHERE ?1 = ?1 {filter}
            ORDER BY date DESC
            LIMIT ?2",
        )
    };

    let q = if last_is_space {
        q.trim()
    } else {
        &format!("{q}*").to_string()
    };

    let mut query = db.query(&sql)?;
    let results = query.query_map((&q, top_k), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f32>(2)?
        ))
    })?;
    for result in results {
        let (rowid, snippet, score) = result?;
        let score2 = bm25_to_cosine_approx(score);
        fts_matches.push((score2, rowid, snippet));
    }
    info!("full text found {} matches", fts_matches.len());
    Ok(fts_matches)
}

pub fn reciprocal_rank_fusion(list1: &[u32], list2: &[u32], k: f64) -> Vec<u32> {
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

fn get_centers(
    db: &DB,
    device: &Device,
    version: u64,
) -> Result<(Vec<u32>, Vec<usize>, Vec<Tensor>, Tensor)> {
    let now = std::time::Instant::now();
    {
        let cache = CACHED.read().unwrap();
        if let Some(cached) = &*cache {
            if cached.version == version {
                return Ok((
                    cached.cluster_ids.clone(),
                    cached.sizes.clone(),
                    cached.centers.clone(),
                    cached.tensor.clone(),
                ));
            }
        }
    }

    let mut center_query =
        db.query("SELECT id,length(indices)/4,center FROM bucket ORDER BY id")?;
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
    debug!(
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

#[inline(always)]
fn vmax_inplace(current: &mut [f32], row: &[f32]) {
    debug_assert_eq!(current.len(), row.len());
    // Process 8 at a time (helps LLVM emit SIMD), then handle the tail.
    let (c8, c_tail) = current.as_chunks_mut::<8>();
    let (r8, r_tail) = row.as_chunks::<8>();

    for (c, r) in c8.iter_mut().zip(r8.iter()) {
        // Unrolled 8-lane max; safe, no bounds checks in the loop body.
        c[0] = c[0].max(r[0]);
        c[1] = c[1].max(r[1]);
        c[2] = c[2].max(r[2]);
        c[3] = c[3].max(r[3]);
        c[4] = c[4].max(r[4]);
        c[5] = c[5].max(r[5]);
        c[6] = c[6].max(r[6]);
        c[7] = c[7].max(r[7]);
    }

    for (c, &r) in c_tail.iter_mut().zip(r_tail.iter()) {
        *c = c.max(r);
    }
}

pub fn match_centroids(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&str>,
) -> Result<Vec<(f32, u32, u32)>> {
    let max_generation = db
        .query("SELECT MAX(generation) FROM indexed_chunk")?
        .query_row((), |row| Ok(row.get::<_, u32>(0)?))
        .unwrap_or(0);
    let mut bucket_query =
        db.query("SELECT indices,residuals FROM bucket WHERE generation = ?1 and id = ?2")?;

    let k = 32;
    let t_prime = 40000;
    let device = query_embeddings.device();
    let (m, _n) = query_embeddings.dims2()?;

    let mut all_document_embeddings = vec![];
    let mut all = vec![];
    let mut count = 0;
    let mut missing = vec![];

    let (cluster_ids, sizes, centers, centers_matrix) =
        get_centers(&db, &device, max_generation as u64)?;

    if centers.len() > 0 {
        let now = std::time::Instant::now();

        let query_centroid_similarity =
            query_embeddings.matmul(&centers_matrix.transpose(D::Minus1, D::Minus2)?)?;
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
        debug!(
            "finding top-{} out of {} clusters took {} ms.",
            topk_clusters.len(),
            n,
            now.elapsed().as_millis()
        );

        let now = std::time::Instant::now();

        for i in topk_clusters {
            let (keys_compressed, document_embeddings) = bucket_query
                .query_row((max_generation, cluster_ids[i as usize]), |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
                })?;

            match centers.get(i as usize) {
                Some(center) => {
                    let document_indices = decompress_keys(&keys_compressed)?;

                    //let residuals = Tensor::from_q4_bytes(&document_embeddings, EMBEDDING_DIM, &device)?.dequantize(4)?.inv_compand()?;
                    let residuals = Tensor::from_companded_q4_bytes(
                        &document_embeddings,
                        EMBEDDING_DIM,
                        &Device::Cpu,
                    )?;
                    let embeddings = residuals.broadcast_add(&center)?;
                    all_document_embeddings.push(embeddings);

                    let (m, _) = residuals.dims2()?;
                    for j in 0..m {
                        all.push((document_indices[j], count));
                        count += 1;
                    }
                }
                None => {
                    warn!("OOB array access i={i}, skipping entry!");
                }
            }
        }
        debug!(
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

    let missing_score = missing_similarities.mean(0)?.to_scalar::<f32>()?;
    let cutoff = if missing_score > threshold {
        missing_score
    } else {
        threshold
    };

    let now = std::time::Instant::now();
    let mut num_unindexed = 0;
    let mut unindexed_chunks_query = db.query(
        "
        SELECT d.rowid, c.embeddings
        FROM document AS d
        JOIN chunk AS c ON c.hash = d.hash
        WHERE NOT EXISTS (
          SELECT 1
          FROM indexed_chunk AS i
          WHERE i.chunkid = c.rowid
            AND i.generation = ?1
        )
        ORDER BY d.rowid",
    )?;

    let results = unindexed_chunks_query.query_map((max_generation,), |row| {
        Ok((row.get::<_, u32>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    for result in results {
        let (id, embeddings) = result?;
        let embeddings = Tensor::from_q8_bytes(&embeddings, EMBEDDING_DIM, &Device::Cpu)?
            .dequantize(8)?
            .l2_normalize()?;
        let (m, _) = embeddings.dims2()?;
        all_document_embeddings.push(embeddings);
        for _ in 0..m {
            let key = (id, 0);
            all.push((key, count));
            count += 1;
        }
        num_unindexed += m;
    }
    debug!(
        "reading {} unindexed embeddings took {} ms.",
        num_unindexed,
        now.elapsed().as_millis()
    );

    if all_document_embeddings.len() == 0 {
        return Ok([].to_vec());
    }

    let all_document_embeddings = Tensor::cat(&all_document_embeddings, 0)?;
    let all_document_embeddings = all_document_embeddings.to_device(query_embeddings.device())?;

    let now = std::time::Instant::now();
    let sim = query_embeddings
        .matmul(&all_document_embeddings.t()?)?
        .transpose(0, 1)?;
    let sim = sim.to_device(&Device::Cpu)?;

    let sim = sim.to_dtype(DType::F32)?.contiguous()?;
    let (_, n) = sim.dims2()?;
    let sim: Vec<f32> = sim.flatten_all()?.to_vec1::<f32>()?;
    let row_at = |pos: usize| -> &[f32] {
        let start = pos * n;
        &sim[start..start + n]
    };

    let missing_similarities = missing_similarities.contiguous()?.to_vec1::<f32>()?;

    debug!("sim mmul took {} ms.", now.elapsed().as_millis());

    let now = std::time::Instant::now();
    all.sort_unstable();
    debug!(
        "sorting {} rows took {}ms",
        all.len(),
        now.elapsed().as_millis()
    );

    let now = std::time::Instant::now();
    let mut scores = vec![0.0f32; n];
    let mut sub_scores = vec![0.0f32; n];
    scores.copy_from_slice(&missing_similarities);
    sub_scores.copy_from_slice(&missing_similarities);

    let mut unique_docs = 0;
    let mut prev_idx = u32::MAX;
    let mut all_scored = vec![];

    let mut prev_sub_idx = u32::MAX;
    let mut max_sub_score = -1.0f32;
    let mut i_max_sub_score = 0;

    let scaler = 1.0f32 / n as f32;
    for i in 0.. {

        let ((idx, sub_idx), pos) = all[i];

        let is_last = i == all.len() - 1;

        let idx_change = prev_idx != idx;
        let sub_idx_change = idx_change || prev_sub_idx != sub_idx;

        if i > 0 {

            if sub_idx_change || is_last {
                let sub_score = scaler * (sub_scores.iter().copied().sum::<f32>());
                if sub_score > max_sub_score {
                    max_sub_score = sub_score;
                    i_max_sub_score = prev_sub_idx;
                }
                sub_scores.copy_from_slice(&missing_similarities);
            }

            if idx_change || is_last {
                unique_docs += 1;
                let score = scaler * (scores.iter().copied().sum::<f32>());
                if score > cutoff {
                    all_scored.push((prev_idx, score, i_max_sub_score));
                }
                max_sub_score = -1.0f32;
                i_max_sub_score = 0;
                scores.copy_from_slice(&missing_similarities);
            }

        }

        if is_last {
            break;
        }

        let row = row_at(pos);
        vmax_inplace(&mut scores, row);
        vmax_inplace(&mut sub_scores, row);

        assert!(i == 0 || prev_idx <= idx);
        assert!(i == 0 || (prev_idx != idx || prev_sub_idx <= sub_idx));
        prev_idx = idx;
        prev_sub_idx = sub_idx;
    }
    debug!(
        "scoring {} documents into {} candidates took {} ms.",
        unique_docs,
        all_scored.len(),
        now.elapsed().as_millis()
    );

    let now = std::time::Instant::now();
    db.execute("CREATE TEMPORARY TABLE temp2(rowid INTEGER PRIMARY KEY, score FLOAT, sub_idx INTEGER)")?;
    let mut insert_temp_query = db.query("INSERT INTO temp2 VALUES(?1, ?2, ?3)")?;

    for (idx, score, sub_idx) in all_scored.iter() {
        let _ = insert_temp_query.execute((idx, score, sub_idx));
    }

    let sql = format!(
        "SELECT score,document.rowid,sub_idx
        FROM document,temp2
        WHERE document.rowid = temp2.rowid
        {}
        ORDER BY score DESC
        LIMIT ?1",
        match sql_filter {
            Some(filter) => format!("WHERE {filter}"),
            _ => String::new(),
        }
    );

    let mut scored_documents_query = db.query(&sql)?;
    let results = scored_documents_query
        .query_map((top_k,), |row| {
            Ok((
                row.get::<_, f32>(0)?,
                row.get::<_, u32>(1)?,
                row.get::<_, u32>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    db.execute("DROP TABLE temp2")?;

    debug!(
        "reading scored and filtered document ids from DB took {} ms",
        now.elapsed().as_millis()
    );
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

pub struct Gatherer<'a> {
    documents: Box<dyn Iterator<Item = (String, String, String)> + 'a>,
    embedder: &'a Embedder,
}

impl<'a> Gatherer<'a> {
    fn new(stmt: &'a mut Statement, embedder: &'a Embedder) -> Self {
        let documents = Box::new(
            stmt.query_map((), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?
                ))
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
    type Item = (String, Tensor, Vec<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.documents.next() {
            Some((hash, body, lens)) => {
                let now = std::time::Instant::now();
                let (embeddings, offsets) = self.embedder.embed(&body).unwrap();
                let embeddings = embeddings
                    .squeeze(0)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap();
                let (m, _n) = embeddings.dims2().unwrap();
                let dt = now.elapsed().as_secs_f64();
                debug!(
                    "embedder took {} ms ({} rows/s).",
                    now.elapsed().as_millis(),
                    ((m as f64) / dt).round()
                );

                let mut lengths: Vec<usize> = lens
                    .split(',')
                    .filter_map(|s| s.parse::<usize>().ok())
                    .collect();
                for i in 1..lengths.len() {
                    lengths[i] += lengths[i - 1];
                }

                let mut i = 0;
                let mut j = 0;

                let i_end = offsets.len();
                let j_end = lengths.len();
                let mut count: u32 = 0;
                let mut done = false;
                let mut flush = false;
                let mut counts = vec!();

                while !done {

                    let o = if i < offsets.len() {
                        offsets[i].1
                    } else {
                        std::usize::MAX
                    };

                    let l = if j < lengths.len() {
                        lengths[j]
                    } else {
                        std::usize::MAX
                    };

                    if o <= l {
                        i += 1;
                        count += 1;
                    } else {
                        j += 1;
                        flush = true;
                    }

                    done = i == i_end && j == j_end;

                    if flush || done {
                        counts.push(count);
                        count = 0;
                        flush = false;
                    }

                }
                assert!(count == 0);
                assert!(counts.iter().sum::<u32>() == offsets.len() as u32);
                Some((hash, embeddings, counts))
            }
            None => None,
        }
    }
}

pub fn embed_chunks(db: &DB, embedder: &Embedder, limit: Option<usize>) -> Result<usize> {
    let sql = format!(
        "SELECT
        document.hash,document.body,document.lens
        FROM document
        LEFT JOIN chunk ON document.hash = chunk.hash
        WHERE chunk.hash IS NULL
        ORDER BY document.hash
        {}",
        match limit {
            Some(limit) => format!("LIMIT {limit}"),
            _ => String::new(),
        }
    );
    let mut query = db.query(&sql)?;

    let embedding_iter = Gatherer::new(&mut query, embedder);
    let mut count = 0;
    for (hash, embeddings, counts) in embedding_iter {
        info!(
            "got embedding for chunk with hash {} {:?} {:?}",
            hash,
            embeddings.dims2()?,
            counts,
        );

        let bytes = embeddings.stretch_rows()?.quantize(8)?.to_q8_bytes()?;
        let counts = counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",");

        match db.add_chunk(&hash, "xtr-base-en", &bytes, &counts) {
            Ok(()) => {
                count += 1;
            }
            Err(v) => {
                info!("add_chunk failed {}", v);
                break;
            }
        };
    }
    debug!("embedded {count} chunks");
    Ok(count)
}

pub fn count_unindexed_embeddings(db: &DB) -> Result<usize> {
    let mut unindexed_chunks_query = db.query(&format!(
        "SELECT IFNULL(SUM(length(c.embeddings)), 0)/{EMBEDDING_DIM} AS total
        FROM chunk AS c
        LEFT JOIN indexed_chunk AS i
        ON i.chunkid = c.rowid
        AND i.generation = (SELECT MAX(generation) FROM indexed_chunk)
        WHERE i.chunkid IS NULL"
    ))?;

    let count = unindexed_chunks_query.query_row((), |row| Ok(row.get::<_, usize>(0)?))?;
    Ok(count)
}

pub fn index_chunks(db: &DB, device: &Device) -> Result<()> {
    let unindexed = count_unindexed_embeddings(&db)?;
    info!(
        "database has {} unindexed embeddings, reindexing...",
        unindexed
    );

    let mut kmeans_query = db.query("SELECT chunk.embeddings FROM chunk")?;
    let mut total_embeddings = 0;
    let mut rng = rand::rng();
    let mut all_embeddings = vec![];
    debug!("read embeddings...");
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
    debug!("total_embeddings={} k={}", total_embeddings, k);
    let (m, _) = matrix.dims2()?;
    if m < k {
        k = m / 4;
    }
    let (centers, _idxs) = kmeans(&matrix, k as usize, 5, &device)?;
    debug!("kmeans took {} ms.", now.elapsed().as_millis());

    debug!("write buckets...");
    let now = std::time::Instant::now();
    match write_buckets(&db, &centers, &device) {
        Ok(()) => {}
        Err(v) => {
            info!("write buckets failed {}", v);
        }
    }
    info!("write buckets took {} ms.", now.elapsed().as_millis());
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
        Self {
            cache: LruCache::new(cap),
        }
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
) -> Result<Vec<(f32, String, String, u32)>> {
    let now = std::time::Instant::now();

    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");

    let fts_matches = if use_fulltext {
        fulltext_search(&db, &q, top_k, sql_filter)?
    } else {
        [].to_vec()
    };

    let mut fts_snippets: HashMap<u32, String> = HashMap::new();
    for (_score, idx, snippet) in &fts_matches {
        fts_snippets.insert(*idx, snippet.clone());
    }

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
        match match_centroids(&db, &qe, threshold, top_k, sql_filter) {
            Ok(result) => result,
            Err(v) => {
                warn!("match_centroids failed {v}");
                [].to_vec()
            }
        }
    } else {
        [].to_vec()
    };

    let mut scores: HashMap<u32, f32> = HashMap::new();
    let mut offsets: HashMap<u32, u32> = HashMap::new();
    for (score, idx, offset) in &sem_matches {
        scores.insert(*idx, *score);
        offsets.insert(*idx, *offset);
    }
    for (score, idx, _snippet) in &fts_matches {
        scores.insert(*idx, *score);
    }

    let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx, _)| idx).collect();
    info!("semantic search found {} matches", sem_idxs.len());

    let mut fused = if use_fulltext {
        let fts_idxs: Vec<u32> = fts_matches.iter().map(|&(_, idx, _)| idx).collect();
        reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
    } else {
        sem_idxs
    };
    fused.truncate(top_k);

    let mut results = vec![];
    let mut body_query = db.query("SELECT metadata,body FROM document WHERE rowid = ?1")?;
    for idx in fused {
        let score = match scores.get(&idx) {
            Some(score) => *score,
            None => 0.0f32,
        };
        let (metadata, body) = body_query.query_row((idx,), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
            ))
        })?;

        let body_idx = match offsets.get(&idx) {
            Some(offset) => *offset,
            None => 0u32
        };
        results.push((score, metadata, body, body_idx));
    }

    let mut max = -1.0f32;
    for (score, _, _, _) in results.iter_mut().rev() {
        max = max.max(*score);
        *score = max;
    }

    debug!(
        "warp search took {} ms end-to-end.",
        now.elapsed().as_millis()
    );
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
            let (qe, _offsets) = embedder.embed(&q)?;
            let qe = qe.get(0)?;
            qe
        }
    };
    let mut sizes = vec![];
    let mut ses = vec![];
    for s in sentences.iter() {
        let (se, _offsets) = embedder.embed(&s)?;
        let se = se.get(0)?;
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
    debug!(
        "scoring {} sentences took {} ms.",
        sentences.len(),
        now.elapsed().as_millis()
    );
    Ok(scores)
}

/*
pub fn split_by_codepoints<'a>(s: &'a str, lengths: &[usize]) -> Result<Vec<&'a str>, String> {
    // Precompute byte indices of every char boundary: [0, b1, b2, ..., s.len()]
    let mut boundaries: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    boundaries.push(s.len());
    let char_len = boundaries.len() - 1;

    let sum_chars: usize = lengths.iter().copied().sum();
    if sum_chars != char_len {
        return Err(format!("lengths sum ({}) != char count ({})", sum_chars, char_len));
    }

    let mut parts = Vec::with_capacity(lengths.len());
    let mut pos = 0usize; // index into `boundaries` (in chars)

    for &chunk_chars in lengths {
        let start_byte = boundaries[pos];
        let end_pos = pos + chunk_chars;
        let end_byte = boundaries[end_pos];
        // Slicing on these byte indices is always valid by construction.
        parts.push(&s[start_byte..end_byte]);
        pos = end_pos;
    }

    Ok(parts)
}
*/
#[test]
fn test_compress_decompress_keys_roundtrip() {
    let test_cases = vec![
        vec![],
        vec![(1, 1)],
        vec![(1, 1), (1, 3), (2, 0), (2, 1)],
        vec![(10, 0), (10, 1), (10, 2), (11, 0)],
        vec![(1, 2), (1, 2), (1, 2), (2, 2)],
        vec![(0, 0)],
        vec![],
    ];

    for original in test_cases {
        let compressed = compress_keys(&original);
        let decompressed = decompress_keys(&compressed).unwrap();
        assert_eq!(original, decompressed);
    }
}

#[cfg(test)]
mod tests;
