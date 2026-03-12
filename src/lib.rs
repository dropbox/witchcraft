use log::{debug, info, warn};
use once_cell::sync::Lazy;
use rusqlite::Statement;
use std::collections::HashMap;
use std::sync::RwLock;
// Conditionally compile T5 encoder based on features
#[cfg(feature = "t5-quantized")]
pub mod quantized_t5;
#[cfg(feature = "t5-quantized")]
use quantized_t5 as t5_encoder;
pub mod fast_ops;
#[cfg(feature = "hybrid-dequant")]
pub mod fused_matmul;

#[cfg(feature = "t5-openvino")]
mod openvino_t5;
#[cfg(feature = "t5-openvino")]
use openvino_t5 as t5_encoder;

// Compile-time checks for mutual exclusivity
#[cfg(not(any(feature = "t5-quantized", feature = "t5-openvino")))]
compile_error!("Must enable exactly one T5 backend: t5-quantized or t5-openvino");

#[cfg(all(feature = "t5-quantized", feature = "t5-openvino"))]
compile_error!("Cannot enable multiple T5 backends simultaneously");

// hybrid-dequant is a CPU-only optimization and cannot be used with Metal
#[cfg(all(feature = "hybrid-dequant", feature = "metal"))]
compile_error!("hybrid-dequant is incompatible with metal (use accelerate only for CPU, or metal without hybrid-dequant for GPU)");

mod db;
pub use db::DB;

mod embedder;
pub use embedder::Embedder;

pub mod assets;

mod packops;
use packops::TensorPackOps;

mod haarops;

pub mod rans64;

mod merger;

mod priority;
use priority::PriorityManager;

mod progress_reporter;
use progress_reporter::ProgressReporter;

pub mod types;
pub use types::SqlStatementInternal;

mod sql_generator;
use sql_generator::build_filter_sql_and_params;

#[cfg(feature = "napi")]
#[allow(dead_code)]
mod napi;

#[cfg(feature = "napi")]
pub use types::{SqlCondition, SqlLogic, SqlOperator, SqlStatement, SqlStatementType};

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};

const EMBEDDING_DIM: usize = 128;

pub fn make_device() -> Device {
    // Metal only works on Apple Silicon (ARM), not Intel x86_64
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
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

#[cfg(all(feature = "progress", not(feature = "napi")))]
pub mod progress {
    use indicatif::{ProgressBar, ProgressStyle};
    pub struct Bar {
        pb: ProgressBar,
    }
    pub fn new(len: u64) -> Bar {
        new_with_label(len, "")
    }
    pub fn new_with_label(len: u64, label: &str) -> Bar {
        let pb = ProgressBar::new(len);
        if !label.is_empty() {
            let style = ProgressStyle::default_bar()
                .template(&format!("{{msg}} [{{bar:40}}] {{pos}}/{{len}}"))
                .unwrap();
            pb.set_style(style);
            pb.set_message(label.to_string());
        }
        Bar { pb }
    }
    impl Bar {
        pub fn inc(&self, n: u64) {
            self.pb.inc(n);
        }
        pub fn finish(&self) {
            self.pb.finish();
        }
    }
}

#[cfg(feature = "napi")]
pub mod progress {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub struct Bar {
        total: u64,
        current: AtomicU64,
        label: String,
    }
    pub fn new(len: u64) -> Bar {
        new_with_label(len, "")
    }
    pub fn new_with_label(len: u64, label: &str) -> Bar {
        Bar {
            total: len,
            current: AtomicU64::new(0),
            label: label.to_string(),
        }
    }
    impl Bar {
        pub fn inc(&self, n: u64) {
            let current = self.current.fetch_add(n, Ordering::Relaxed) + n;
            if self.total > 0 {
                let progress = (current as f64) / (self.total as f64);
                #[cfg(feature = "napi")]
                crate::napi::progress_update(progress.min(1.0), &self.label);
            }
        }
        pub fn finish(&self) {
            #[cfg(feature = "napi")]
            crate::napi::progress_update(1.0, &self.label);
        }
    }
}

#[cfg(not(any(feature = "progress", feature = "napi")))]
pub mod progress {
    #[derive(Clone, Copy)]
    pub struct Bar;
    #[allow(dead_code)]
    pub fn new(_len: u64) -> Bar {
        Bar
    }
    pub fn new_with_label(_len: u64, _label: &str) -> Bar {
        Bar
    }
    impl Bar {
        pub fn inc(&self, _n: u64) {}
        pub fn finish(&self) {}
    }
}

fn matmul_argmax_batched(
    t: &Tensor,
    centers: &fast_ops::PackedRight,
    batch_size: usize,
) -> Result<Tensor> {
    let (m, _n) = t.dims2()?;
    let device = t.device();

    let mut assignments = Vec::with_capacity(m);

    for start in (0..m).step_by(batch_size) {
        let end = (start + batch_size).min(m);
        let batch_len = end - start;
        let batch = t.narrow(0, start, batch_len)?;
        let sim = centers.matmul(&batch)?;
        let batch_assignments = sim.argmax(D::Minus1)?;
        let batch_assignments = batch_assignments.to_vec1::<u32>()?;
        assignments.extend(batch_assignments);
    }

    Ok(Tensor::from_vec(assignments, m, device)?)
}

fn kmeans(data: &Tensor, k: usize, max_iter: usize) -> Result<Tensor> {
    let (m, n) = data.dims2()?;
    debug!("kmeans k={} m={} n={}...", k, m, n);

    let _priority_mgr = PriorityManager::new();
    let total: u64 = (max_iter * k).try_into()?;
    let bar = progress::new_with_label(total, "kmeans");
    let device = data.device();

    let mut rng = rand::rng();
    let centroid_idx = rand::seq::index::sample(&mut rng, m, k).into_vec();
    let centroid_idx: Vec<u32> = centroid_idx.iter().map(|&i| i as u32).collect();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    //let centroid_idx_tensor = centroid_idx_tensor.to_device(device)?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;

    // Pull data out once; kmeans always runs on CPU.
    let data_flat = data.flatten_all()?.to_vec1::<f32>()?;

    for _ in 0..max_iter {
        let packed_centers = fast_ops::PackedRight::new(&centers)?;
        let cluster_assignments = matmul_argmax_batched(data, &packed_centers, 1024)?;
        let assignments = cluster_assignments.to_vec1::<u32>()?;

        // Single O(m × n) pass: accumulate per-cluster sums directly into a
        // flat Vec<f32>, avoiding O(k × m) scans and k separate tensor ops.
        let mut sums = vec![0f32; k * n];
        let mut counts = vec![0u32; k];
        for (j, &c) in assignments.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1;
            let src = &data_flat[j * n..(j + 1) * n];
            let dst = &mut sums[c * n..(c + 1) * n];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        }

        // Normalize each cluster sum; reinit empty clusters from a random point.
        let mut centers_flat = vec![0f32; k * n];
        for i in 0..k {
            let src = &sums[i * n..(i + 1) * n];
            let dst = &mut centers_flat[i * n..(i + 1) * n];
            let (emb, owned);
            if counts[i] > 0 {
                emb = src;
            } else {
                let idx = rand::seq::index::sample(&mut rng, m, 1).into_vec()[0];
                owned = data_flat[idx * n..(idx + 1) * n].to_vec();
                emb = &owned;
            }
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for (d, e) in dst.iter_mut().zip(emb) {
                    *d = e / norm;
                }
            } else {
                dst.copy_from_slice(emb);
            }
        }

        centers = Tensor::from_vec(centers_flat, (k, n), device)?;
        bar.inc(k as u64);
    }
    bar.finish();
    Ok(centers)
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

fn write_buckets(
    db: &DB,
    centers: &Tensor,
    device: &Device,
    embeddings_sql: &str,
    expected_count: u64,
) -> Result<(Vec<tempfile::NamedTempFile>, Vec<u32>, Tensor)> {
    let _priority_mgr = PriorityManager::new();
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let bar = progress::new_with_label(expected_count, "indexing");

    let mut document_indices = Vec::<(u32, u32)>::new();
    let mut all_chunkids = vec![];
    let mut all_embeddings = vec![];

    let mut query = db.query(embeddings_sql)?;

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
    let packed_centers = fast_ops::PackedRight::new(centers)?;
    while !done {
        match results.next() {
            Some(result) => {
                let (id, chunkid, embeddings, counts) = result?;

                let t = Tensor::embeddings_from_packed(&embeddings, EMBEDDING_DIM, &Device::Cpu)?;
                let split = split_tensor(&t);
                let m = split.len();

                for (i, count) in counts
                    .split(',')
                    .filter_map(|s| s.parse::<u32>().ok())
                    .enumerate()
                {
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
            let data = Tensor::cat(&embeddings, 0)?.to_device(device)?;

            // Use batched matmul+argmax to avoid materializing huge intermediate matrix
            let cluster_assignments =
                matmul_argmax_batched(&data, &packed_centers, 1024)?.to_device(&Device::Cpu)?;
            debug!("simmax took {} ms", now.elapsed().as_millis());
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

            for (sample, bucket) in pairs.iter().copied().chain(std::iter::once((0, u32::MAX))) {
                let bucket_done = bucket == u32::MAX;

                if (bucket != prev_bucket || bucket_done) && !keys.is_empty() {
                    assert!(prev_bucket < bucket);
                    writer.write_record(prev_bucket, &keys, &residuals_bytes)?;

                    keys.clear();
                    residuals_bytes.clear();
                    prev_bucket = bucket;
                }

                if bucket_done {
                    break;
                }

                match indices.get(sample) {
                    Some(pair) => {
                        keys.push(*pair);
                    }
                    None => {
                        warn!("unable to get key pair from indices @{sample}");
                        keys.push((0, 0));
                    }
                }

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

    Ok((tmpfiles, all_chunkids, centers_cpu))
}

fn merge_and_write_buckets(
    db: &DB,
    tmpfiles: Vec<tempfile::NamedTempFile>,
    all_chunkids: &[u32],
    centers_cpu: &Tensor,
    id_offset: u32,
) -> Result<()> {
    let mut merger = merger::Merger::from_tempfiles(tmpfiles)?;
    for result in &mut merger {
        let entry = result?;
        let center = centers_cpu.get(entry.value as usize)?;
        let center_bytes = center.to_f32_bytes()?;
        let compressed_keys = compress_keys(&entry.keys);
        db.add_bucket(
            id_offset + entry.value,
            &center_bytes,
            &compressed_keys,
            &entry.data,
        )?;
    }
    info!("write {} chunk ids to indexed_chunk", all_chunkids.len());
    for &chunkid in all_chunkids {
        db.add_indexed_chunk(chunkid)?;
    }
    Ok(())
}

pub fn fulltext_search(
    db: &DB,
    q: &str,
    top_k: usize,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    let mut fts_matches = vec![];

    let mut last_is_space = false;
    for c in q.chars() {
        last_is_space = c == ' ';
    }
    let q: String = q
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    let (filter_sql, mut filter_params) = build_filter_sql_and_params(sql_filter)?;
    let filter_clause = if !filter_sql.is_empty() {
        format!("AND {}", filter_sql)
    } else {
        String::new()
    };

    let q_param = if last_is_space {
        q.trim().to_string()
    } else {
        format!("{q}*")
    };

    let sql = if !q.is_empty() {
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
        // For empty query, we don't need the query param in the WHERE clause
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
    if !q.is_empty() {
        params.push(Box::new(q_param.clone()));
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
        let (rowid, body, lens, _score) = result?;
        let score2 = strsim::jaro_winkler(&q_param, &body) as f32;

        let lens: Vec<usize> = lens
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .collect();

        let mut max = -1.0f64;
        let mut i_max = 0;
        if !lens.is_empty() {
            let bodies = split_by_codepoints(&body, &lens);
            for (i, &b) in bodies.iter().enumerate() {
                let score = strsim::jaro_winkler(&q_param, b);
                if score > max {
                    max = score;
                    i_max = i;
                }
            }
        }
        fts_matches.push((score2, rowid, i_max as u32));
    }
    info!("full text found {} matches", fts_matches.len());
    Ok(fts_matches)
}

pub fn reciprocal_rank_fusion(list1: &[u32], list2: &[u32], k: f64) -> Vec<u32> {
    let mut scores: HashMap<u32, f64> = HashMap::new();

    for (rank, &doc_id) in list1.iter().enumerate() {
        let score = 1.0 / (3.0 + k + rank as f64);
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

type CentersCache = (Vec<u32>, Vec<usize>, Vec<Tensor>, Tensor);

static CACHED: Lazy<RwLock<Option<CentersCache>>> = Lazy::new(|| RwLock::new(None));

fn invalidate_center_cache() {
    *CACHED.write().unwrap() = None;
}

#[allow(clippy::type_complexity)]
fn get_centers(db: &DB, device: &Device) -> Result<(Vec<u32>, Vec<usize>, Vec<Tensor>, Tensor)> {
    let now = std::time::Instant::now();
    {
        let cache_lock_start = std::time::Instant::now();
        let cache = CACHED.read().unwrap();
        let lock_time = cache_lock_start.elapsed().as_micros();
        if let Some((cluster_ids, sizes, centers, tensor)) = &*cache {
            let clone_start = std::time::Instant::now();
            let result = Ok((
                cluster_ids.clone(),
                sizes.clone(),
                centers.clone(),
                tensor.clone(),
            ));
            debug!(
                "get_centers cache hit: lock {}μs, clone {}μs",
                lock_time,
                clone_start.elapsed().as_micros()
            );
            return result;
        }
    }

    let mut center_query =
        db.query("SELECT id,length(residuals)/64,center FROM bucket ORDER BY id")?;
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
    let tensor = if !centers.is_empty() {
        Tensor::stack(&centers, 0)?.to_device(device)?
    } else {
        Tensor::zeros(&[0, EMBEDDING_DIM], DType::F32, device)?
    };
    debug!(
        "reading and stacking centers took {} ms (caching result for future queries.)",
        now.elapsed().as_millis()
    );

    let mut cache = CACHED.write().unwrap();
    *cache = Some((
        cluster_ids.clone(),
        sizes.clone(),
        centers.clone(),
        tensor.clone(),
    ));

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
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    let total_start = std::time::Instant::now();
    let setup_start = std::time::Instant::now();

    let query_prep_start = std::time::Instant::now();
    let mut bucket_query = db.query("SELECT indices,residuals FROM bucket WHERE id = ?1")?;
    let query_prep_time = query_prep_start.elapsed().as_millis();

    let k = 32;
    let t_prime = 40000;
    let device = query_embeddings.device();
    let (m, _n) = query_embeddings.dims2()?;

    let mut all_residuals = vec![]; // Store residuals, not full embeddings
    let mut document_clusters = vec![]; // Track which cluster each document belongs to
    let mut all = vec![];
    let mut count = 0;
    let mut missing = vec![];
    let mut query_centroid_scores: Vec<Vec<f32>> = vec![]; // Centroid scores for reuse

    let centers_start = std::time::Instant::now();
    let (cluster_ids, sizes, centers, centers_matrix) = get_centers(db, device)?;
    let centers_time = centers_start.elapsed().as_millis();

    debug!(
        "setup: bucket query prep {}ms, get_centers {}ms, total {}ms",
        query_prep_time,
        centers_time,
        setup_start.elapsed().as_millis()
    );

    if !centers.is_empty() {
        let now = std::time::Instant::now();

        let query_centroid_similarity = fast_ops::matmul_t(query_embeddings, &centers_matrix)?;
        let query_centroid_similarity = query_centroid_similarity.to_device(&Device::Cpu)?;

        // Extract centroid scores for later use (m queries × n_clusters)
        query_centroid_scores = query_centroid_similarity.to_vec2::<f32>()?;

        let sorted_indices = query_centroid_similarity.arg_sort_last_dim(false)?;
        let (m, n) = sorted_indices.dims2()?;

        let mut topk_clusters = Vec::with_capacity(k);
        for i in 0..m {
            let row = sorted_indices.get(i)?;
            let row_scores_sorted = query_centroid_similarity.get(i)?.gather(&row, D::Minus1)?;
            let row_scores_sorted = row_scores_sorted.to_vec1::<f32>()?;
            let row = row.to_vec1::<u32>()?;
            let mut cumsum = 0;
            let mut score = 0.0f32;
            for j in 0..n.min(k) {
                let idx = row[j];
                topk_clusters.push(idx);
                let size = sizes[idx as usize];
                if cumsum < t_prime {
                    score = row_scores_sorted[j];
                }
                cumsum += size;

                if cumsum >= t_prime {
                    break;
                }
            }
            missing.push(score);
        }
        topk_clusters.sort_unstable();
        topk_clusters.dedup();
        debug!(
            "finding top-{} out of {} clusters took {} ms.",
            topk_clusters.len(),
            n,
            now.elapsed().as_millis()
        );

        let now = std::time::Instant::now();
        let table: [f32; 16] = packops::make_q4_dequant_table()?;

        let mut db_read_time = 0u128;
        let mut decompress_time = 0u128;
        let mut dequant_time = 0u128;

        for i in topk_clusters {
            let db_start = std::time::Instant::now();
            let (keys_compressed, document_embeddings) = bucket_query
                .query_row((cluster_ids[i as usize],), |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
                })?;
            db_read_time += db_start.elapsed().as_millis();

            match centers.get(i as usize) {
                Some(_center) => {
                    let decompress_start = std::time::Instant::now();
                    let document_indices = decompress_keys(&keys_compressed)?;
                    decompress_time += decompress_start.elapsed().as_millis();

                    let dequant_start = std::time::Instant::now();
                    let residuals = Tensor::from_companded_q4_bytes(
                        &document_embeddings,
                        EMBEDDING_DIM,
                        &table,
                        &Device::Cpu,
                    )?;
                    dequant_time += dequant_start.elapsed().as_millis();

                    // Store residuals (not embeddings) and track cluster membership
                    let (num_docs, _) = residuals.dims2()?;
                    all_residuals.push(residuals);
                    for idx in &document_indices[..num_docs] {
                        document_clusters.push(i as usize); // Track which cluster this doc belongs to
                        all.push((*idx, count));
                        count += 1;
                    }
                }
                None => {
                    warn!("OOB array access i={i}, skipping entry!");
                }
            }
        }
        debug!(
            "reading {} indexed embeddings took {} ms (DB: {}ms, decompress keys: {}ms, dequant: {}ms).",
            count,
            now.elapsed().as_millis(),
            db_read_time,
            decompress_time,
            dequant_time
        );
    } else {
        missing.resize(missing.len() + m, 0.0);
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
    let mut unindexed_embeddings = vec![];

    // Check if there are any unindexed chunks before running expensive query
    let unindexed_count: u32 = db
        .query("SELECT COUNT(*) FROM chunk AS c WHERE NOT EXISTS (SELECT 1 FROM indexed_chunk AS i WHERE i.chunkid = c.rowid)")?
        .query_row((), |row| row.get(0))?;

    if unindexed_count > 0 {
        let mut unindexed_chunks_query = db.query(
            "
            SELECT d.rowid, c.embeddings
            FROM document AS d
            JOIN chunk AS c ON c.hash = d.hash
            WHERE NOT EXISTS (
              SELECT 1
              FROM indexed_chunk AS i
              WHERE i.chunkid = c.rowid
            )
            ORDER BY d.rowid",
        )?;

        let results = unindexed_chunks_query.query_map((), |row| {
            Ok((row.get::<_, u32>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for result in results {
            let (id, embeddings) = result?;
            let embeddings =
                Tensor::embeddings_from_packed(&embeddings, EMBEDDING_DIM, &Device::Cpu)?;
            let (num_docs, _) = embeddings.dims2()?;
            unindexed_embeddings.push(embeddings);
            for _ in 0..num_docs {
                let key = (id, 0);
                all.push((key, count));
                count += 1;
            }
            num_unindexed += num_docs;
        }
    }
    debug!(
        "reading {} unindexed embeddings took {} ms.",
        num_unindexed,
        now.elapsed().as_millis()
    );

    if count == 0 {
        return Ok(vec![]);
    }

    let now = std::time::Instant::now();
    let n = m; // Number of queries

    let mut sim: Vec<f32> = Vec::with_capacity(count * n);

    // Process indexed embeddings: compute query·residuals and add centroid scores
    if !all_residuals.is_empty() {
        let all_residuals = Tensor::cat(&all_residuals, 0)?;
        let all_residuals = all_residuals.to_device(query_embeddings.device())?;

        let residual_sims =
            fast_ops::matmul_t(query_embeddings, &all_residuals)?.transpose(0, 1)?;
        let residual_sims = residual_sims.to_device(&Device::Cpu)?;
        let residual_sims = residual_sims.to_dtype(DType::F32)?.contiguous()?;
        let (num_indexed, _) = residual_sims.dims2()?;

        // Add centroid scores to get final similarities for indexed documents
        let mut residual_sims_flat = residual_sims.flatten_all()?.to_vec1::<f32>()?;
        for (doc_idx, cluster_idx) in document_clusters.iter().enumerate().take(num_indexed) {
            for (query_idx, centroid_scores) in query_centroid_scores.iter().enumerate().take(n) {
                let offset = doc_idx * n + query_idx;
                residual_sims_flat[offset] += centroid_scores[*cluster_idx];
            }
        }
        sim.extend_from_slice(&residual_sims_flat);
    }

    // Process unindexed embeddings: compute full similarities
    if !unindexed_embeddings.is_empty() {
        let all_unindexed = Tensor::cat(&unindexed_embeddings, 0)?;
        let all_unindexed = all_unindexed.to_device(query_embeddings.device())?;

        let unindexed_sims =
            fast_ops::matmul_t(query_embeddings, &all_unindexed)?.transpose(0, 1)?;
        let unindexed_sims = unindexed_sims.to_device(&Device::Cpu)?;
        let unindexed_sims = unindexed_sims.to_dtype(DType::F32)?.contiguous()?;
        let unindexed_sims_flat = unindexed_sims.flatten_all()?.to_vec1::<f32>()?;
        sim.extend_from_slice(&unindexed_sims_flat);
    }

    debug!(
        "computing similarities for {} total embeddings took {} ms.",
        count,
        now.elapsed().as_millis()
    );

    let row_at = |pos: usize| -> &[f32] {
        let start = pos * n;
        &sim[start..start + n]
    };

    let missing_similarities = missing_similarities.contiguous()?.to_vec1::<f32>()?;

    let now = std::time::Instant::now();
    all.sort_unstable();
    debug!(
        "sorting {} rows took {}ms",
        all.len(),
        now.elapsed().as_millis()
    );

    let now = std::time::Instant::now();
    let mut sub_scores = vec![0.0f32; n];
    sub_scores.copy_from_slice(&missing_similarities);

    let mut all_scored = vec![];

    let mut prev_idx = u32::MAX;
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
                if max_sub_score > cutoff {
                    all_scored.push((prev_idx, max_sub_score, i_max_sub_score));
                }
                max_sub_score = -1.0f32;
                i_max_sub_score = 0;
            }
        }

        if is_last {
            break;
        }

        let row = row_at(pos);
        vmax_inplace(&mut sub_scores, row);

        assert!(i == 0 || prev_idx <= idx);
        assert!(i == 0 || (prev_idx != idx || prev_sub_idx <= sub_idx));
        prev_idx = idx;
        prev_sub_idx = sub_idx;
    }
    debug!(
        "scoring into {} candidates took {} ms.",
        all_scored.len(),
        now.elapsed().as_millis()
    );

    db.execute(
        "CREATE TEMPORARY TABLE temp2(rowid INTEGER PRIMARY KEY, score FLOAT, sub_idx INTEGER)",
    )?;
    let mut insert_temp_query = db.query("INSERT INTO temp2 VALUES(?1, ?2, ?3)")?;

    for (idx, score, sub_idx) in all_scored.iter() {
        let _ = insert_temp_query.execute((idx, score, sub_idx));
    }

    let (filter_sql, filter_params) = build_filter_sql_and_params(sql_filter)?;
    let filter_clause = if !filter_sql.is_empty() {
        format!("AND {}", filter_sql)
    } else {
        String::new()
    };

    let sql = format!(
        "SELECT score,document.rowid,sub_idx
        FROM document,temp2
        WHERE document.rowid = temp2.rowid
        {filter_clause}
        ORDER BY score DESC
        LIMIT ?",
    );

    let query_start = std::time::Instant::now();
    let results_status: Result<Vec<(f32, u32, u32)>> = {
        let mut scored_documents_query = db.query(&sql)?;

        // Build complete params list: filter params, top_k
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = filter_params;
        params.push(Box::new(top_k as i64));

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let results = scored_documents_query
            .query_map(param_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, f32>(0)?,
                    row.get::<_, u32>(1)?,
                    row.get::<_, u32>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    };
    debug!(
        "querying temp table took {} ms",
        query_start.elapsed().as_millis()
    );

    db.execute("DROP TABLE temp2")?;

    let results = match results_status {
        Ok(results) => results,
        Err(v) => {
            warn!("scoring query failed {}", v);
            [].into()
        }
    };

    debug!("DB operations took {} ms total", now.elapsed().as_millis());
    debug!(
        "match_centroids internal total: {} ms.",
        total_start.elapsed().as_millis()
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
                    row.get::<_, String>(2)?,
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
                let mut counts = vec![];

                while !done {
                    let o = if i < offsets.len() {
                        offsets[i].1
                    } else {
                        usize::MAX
                    };

                    let l = if j < lengths.len() {
                        lengths[j]
                    } else {
                        usize::MAX
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

#[cfg(debug_assertions)]
fn rowwise_cosine_min(a: &Tensor, b: &Tensor) -> Result<f32> {
    let (rows, cols) = a.dims2()?;
    assert_eq!(b.dims2()?, (rows, cols));

    let dot = (a * b)?.sum(1)?;
    let norm_a = a.sqr()?.sum(1)?.sqrt()?;
    let norm_b = b.sqr()?.sum(1)?.sqrt()?;
    let denom = (&norm_a * &norm_b)?;
    let cos = (&dot / &denom)?;
    Ok(cos.min_all()?.to_scalar::<f32>()?)
}

#[cfg(debug_assertions)]
fn stretch_rows(a: &Tensor) -> Result<Tensor> {
    let device = a.device();
    let (m, n) = a.dims2()?;

    let mut scaled_rows = Vec::with_capacity(m);

    for i in 0..m {
        let row = a.get(i)?;
        let v = row.to_vec1::<f32>()?;

        let mut max = f32::MIN;
        for x in &v {
            let a = (*x).abs();
            max = if a > max { a } else { max };
        }
        let range = max + 1e-6;
        let scale = 1.0 / range;

        let v2: Vec<f32> = v.iter().map(|x| scale * x).collect();

        scaled_rows.push(Tensor::from_vec(v2, n, device)?);
    }

    Ok(Tensor::stack(&scaled_rows, 0)?)
}

pub fn embed_chunks(db: &DB, embedder: &Embedder, limit: Option<usize>) -> Result<usize> {
    let _priority_mgr = PriorityManager::new();

    // Count total documents to embed for progress reporting
    let mut progress = {
        let count_sql = format!(
            "SELECT COUNT(*) FROM document
            LEFT JOIN chunk ON document.hash = chunk.hash
            WHERE chunk.hash IS NULL
            {}",
            match limit {
                Some(limit) => format!("LIMIT {limit}"),
                _ => String::new(),
            }
        );
        let mut count_query = db.query(&count_sql)?;
        let total: usize = count_query.query_row((), |row| row.get::<_, usize>(0))?;
        ProgressReporter::new("embed", total)
    };

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
        debug!(
            "got embedding for chunk with hash {} {:?} {:?}",
            hash,
            embeddings.dims2()?,
            counts,
        );

        let now = std::time::Instant::now();
        let bytes = embeddings.embeddings_to_packed()?;
        let (rows, cols) = embeddings.dims2()?;
        let pct = 100.0 * (bytes.len() as f32) / ((rows * cols) as f32);
        let bpe = 8.0 * (bytes.len() as f32) / ((rows * cols) as f32);
        debug!(
            "compressing to {pct:.2}% {bpe:.2}bpe took {} ms.",
            now.elapsed().as_millis()
        );

        #[cfg(debug_assertions)]
        {
            let t = Tensor::embeddings_from_packed(&bytes, EMBEDDING_DIM, &Device::Cpu)?;
            let min_acc = rowwise_cosine_min(&embeddings, &t)?;

            let n = bpe.ceil() as u32;
            let qn = stretch_rows(&embeddings)?.quantize(n)?.dequantize(n)?;
            let min_qn_acc = rowwise_cosine_min(&embeddings, &qn)?;
            println!("haar reconstruction accuracy={min_acc} compare at q{n}_acc={min_qn_acc}");
        }

        let counts = counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",");

        match db.add_chunk(&hash, "xtr-base-en", &bytes, &counts) {
            Ok(()) => {
                count += 1;
                progress.inc(1);
            }
            Err(v) => {
                warn!("add_chunk failed {}", v);
                break;
            }
        };
    }
    progress.finish();

    debug!("embedded {count} chunks");
    if count > 0 {
        db.checkpoint();
    }
    Ok(count)
}

pub fn count_unindexed_embeddings(db: &DB) -> Result<usize> {
    let mut unindexed_chunks_query = db.query(&format!(
        "SELECT IFNULL(SUM(length(c.embeddings)), 0)/{EMBEDDING_DIM} AS total
        FROM chunk AS c
        LEFT JOIN indexed_chunk AS i
        ON i.chunkid = c.rowid
        WHERE i.chunkid IS NULL"
    ))?;

    let count = unindexed_chunks_query.query_row((), |row| row.get::<_, usize>(0))?;
    Ok(count)
}

fn count_indexed_embeddings(db: &DB) -> Result<usize> {
    let count: usize = db
        .query("SELECT IFNULL(SUM(length(residuals)/64), 0) FROM bucket")?
        .query_row((), |row| row.get::<_, usize>(0))?;
    Ok(count)
}

fn count_buckets(db: &DB) -> Result<usize> {
    let count: usize = db
        .query("SELECT COUNT(*) FROM bucket")?
        .query_row((), |row| row.get::<_, usize>(0))?;
    Ok(count)
}

fn sample_embeddings_for_kmeans(db: &DB, sql: &str, device: &Device) -> Result<(Tensor, usize)> {
    let mut kmeans_query = db.query(sql)?;
    let mut total_embeddings = 0;
    let mut rng = rand::rng();
    let mut all_embeddings = vec![];
    for embeddings in kmeans_query.query_map((), |row| row.get::<_, Vec<u8>>(0))? {
        let t = Tensor::embeddings_from_packed(&embeddings?, EMBEDDING_DIM, &Device::Cpu)?;
        let (m, _) = t.dims2()?;
        let k = ((m as f32).sqrt().ceil()) as usize;
        let subset_idx = rand::seq::index::sample(&mut rng, m, k).into_vec();
        for i in subset_idx {
            let row = t.get(i)?;
            all_embeddings.push(row);
        }
        total_embeddings += m;
    }
    let matrix = Tensor::stack(&all_embeddings, 0)?.to_device(device)?;
    Ok((matrix, total_embeddings))
}

fn run_kmeans_for_index(matrix: &Tensor, total_embeddings: usize) -> Result<Tensor> {
    let now = std::time::Instant::now();
    let mut k = (16.0 * (total_embeddings as f64).sqrt()).round() as usize;
    k = k.max(1);
    debug!("total_embeddings={} k={}", total_embeddings, k);
    let (m, _) = matrix.dims2()?;
    if m < k {
        k = m / 4;
    }
    let centers = kmeans(matrix, k, 5)?;
    debug!("kmeans took {} ms.", now.elapsed().as_millis());
    Ok(centers)
}

pub fn full_index(db: &DB, device: &Device) -> Result<()> {
    debug!("read all embeddings for full index...");
    let (matrix, total_embeddings) =
        sample_embeddings_for_kmeans(db, "SELECT chunk.embeddings FROM chunk", device)?;

    let centers = run_kmeans_for_index(&matrix, total_embeddings)?;

    debug!("write buckets (full)...");
    let now = std::time::Instant::now();
    let embeddings_sql =
        "SELECT document.rowid,chunk.rowid,chunk.embeddings,chunk.counts FROM document,chunk
        WHERE document.hash = chunk.hash
        ORDER BY document.rowid";
    let (tmpfiles, all_chunkids, centers_cpu) = write_buckets(
        db,
        &centers,
        device,
        embeddings_sql,
        total_embeddings as u64,
    )?;
    info!(
        "write buckets (full) took {} ms.",
        now.elapsed().as_millis()
    );

    let txstatus = match db.begin_transaction() {
        Ok(()) => {
            db.execute("DELETE FROM bucket")?;
            db.execute("DELETE FROM indexed_chunk")?;
            merge_and_write_buckets(db, tmpfiles, &all_chunkids, &centers_cpu, 0)?;
            Ok(())
        }
        Err(v) => {
            warn!("unable to begin transaction, index not created/updated! {v}");
            Err(v)
        }
    };
    match txstatus {
        Ok(()) => {
            db.commit_transaction()?;
            invalidate_center_cache();
            Ok(())
        }
        Err(v) => {
            warn!("failure during indexing transaction {v}");
            let _ = db.rollback_transaction();
            Err(v.into())
        }
    }
}

fn incremental_index(db: &DB, device: &Device) -> Result<()> {
    debug!("read unindexed embeddings for incremental index...");
    let kmeans_sql = "SELECT c.embeddings FROM chunk AS c
        WHERE NOT EXISTS (SELECT 1 FROM indexed_chunk AS i WHERE i.chunkid = c.rowid)";
    let (matrix, total_embeddings) = sample_embeddings_for_kmeans(db, kmeans_sql, device)?;

    let centers = run_kmeans_for_index(&matrix, total_embeddings)?;

    debug!("write buckets (incremental)...");
    let now = std::time::Instant::now();
    let embeddings_sql =
        "SELECT document.rowid,chunk.rowid,chunk.embeddings,chunk.counts FROM document,chunk
        WHERE document.hash = chunk.hash
        AND NOT EXISTS (SELECT 1 FROM indexed_chunk WHERE chunkid = chunk.rowid)
        ORDER BY document.rowid";
    let (tmpfiles, all_chunkids, centers_cpu) = write_buckets(
        db,
        &centers,
        device,
        embeddings_sql,
        total_embeddings as u64,
    )?;
    info!(
        "write buckets (incremental) took {} ms.",
        now.elapsed().as_millis()
    );

    let txstatus = match db.begin_transaction() {
        Ok(()) => {
            let max_bucket_id: i64 = db
                .query("SELECT IFNULL(MAX(id), -1) FROM bucket")?
                .query_row((), |row| row.get(0))?;
            let id_offset = (max_bucket_id + 1) as u32;

            merge_and_write_buckets(db, tmpfiles, &all_chunkids, &centers_cpu, id_offset)?;
            Ok(())
        }
        Err(v) => {
            warn!("unable to begin transaction, index not created/updated! {v}");
            Err(v)
        }
    };
    match txstatus {
        Ok(()) => {
            db.commit_transaction()?;
            invalidate_center_cache();
            Ok(())
        }
        Err(v) => {
            warn!("failure during indexing transaction {v}");
            let _ = db.rollback_transaction();
            Err(v.into())
        }
    }
}

pub fn index_chunks(db: &DB, device: &Device) -> Result<()> {
    let unindexed = count_unindexed_embeddings(db)?;
    if unindexed == 0 {
        return Ok(());
    }

    let indexed = count_indexed_embeddings(db)?;
    let total_buckets = count_buckets(db)?;
    let expected_k = (16.0 * ((indexed + unindexed) as f64).sqrt()).round() as usize;

    info!(
        "database has {} unindexed embeddings ({} indexed, {} buckets, ~{} expected)",
        unindexed, indexed, total_buckets, expected_k
    );

    let result = if indexed == 0 || unindexed > indexed / 2 || total_buckets > expected_k * 10 {
        full_index(db, device)
    } else {
        incremental_index(db, device)
    };
    if result.is_ok() {
        db.checkpoint();
    }
    result
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
    q: &str,
    threshold: f32,
    top_k: usize,
    use_fulltext: bool,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, String, String, u32)>> {
    let now = std::time::Instant::now();

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
        match match_centroids(db, &qe, threshold, top_k, sql_filter) {
            Ok(result) => result,
            Err(v) => {
                warn!("match_centroids failed {v}");
                vec![]
            }
        }
    } else {
        vec![]
    };

    let mut scores: HashMap<u32, f32> = HashMap::new();
    let mut offsets: HashMap<u32, u32> = HashMap::new();

    for (score, idx, offset) in &fts_matches {
        scores.insert(*idx, *score);
        offsets.insert(*idx, *offset);
    }
    for (score, idx, offset) in &sem_matches {
        scores.insert(*idx, *score);
        offsets.insert(*idx, *offset);
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
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let body_idx = match offsets.get(&idx) {
            Some(offset) => *offset,
            None => 0u32,
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
    let qe = match cache.get(q) {
        Some(existing) => existing,
        None => {
            let (qe, _offsets) = embedder.embed(q)?;

            qe.get(0)?
        }
    };
    let mut sizes = vec![];
    let mut ses = vec![];
    for s in sentences.iter() {
        let (se, _offsets) = embedder.embed(s)?;
        let se = se.get(0)?;
        let split = split_tensor(&se);
        sizes.push(split.len());
        ses.extend(split);
    }
    let ses = Tensor::cat(&ses, 0)?;
    let sim = fast_ops::matmul_t(&ses, &qe)?;
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

pub fn split_by_codepoints<'a>(s: &'a str, lengths: &[usize]) -> Vec<&'a str> {
    // Precompute byte indices of every char boundary: [0, b1, b2, ..., s.len()]
    let mut boundaries: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    boundaries.push(s.len());

    let char_len = boundaries.len() - 1;
    let sum_chars: usize = lengths.iter().copied().sum();
    if sum_chars != char_len {
        warn!("sum of lengths does not match utf8-length of string!");
        return vec![];
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
    parts
}

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
