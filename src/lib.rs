use bitvec::prelude::*;
use log::{debug, info, warn};
use once_cell::sync::Lazy;
#[cfg(feature = "deterministic")]
use rand::SeedableRng;
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
mod max_heap;

mod progress_reporter;
use progress_reporter::ProgressReporter;

pub mod types;
pub use types::SqlStatementInternal;

mod sql_generator;
use sql_generator::build_filter_sql_and_params;

#[cfg(feature = "napi")]
#[allow(dead_code)]
mod napi;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};

const EMBEDDING_DIM: usize = 128;
const RESIDUAL_BYTES: usize = EMBEDDING_DIM / 2;

#[cfg(not(test))]
const L0_CAPACITY: usize = 1024;
#[cfg(test)]
const L0_CAPACITY: usize = 4;

#[cfg(not(test))]
const LSM_FANOUT: usize = 16;
#[cfg(test)]
const LSM_FANOUT: usize = 2;

/// A document pointer combining document ID and sub-chunk index
/// Allows precise location of results within subdivided documents
pub type DocPtr = (u32, u32);

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
                crate::napi::progress_update(progress.min(1.0), &self.label);
            }
        }

        pub fn finish(&self) {
            crate::napi::progress_update(1.0, &self.label);
        }
    }
}

#[cfg(not(any(feature = "progress", feature = "napi")))]
pub mod progress {
    #[derive(Clone, Copy)]
    pub struct Bar;

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

    #[cfg(feature = "deterministic")]
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    #[cfg(not(feature = "deterministic"))]
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
        // Replace underpopulated centroids with perturbed copies of the largest.
        let median_count = {
            let mut sorted_counts = counts.clone();
            sorted_counts.sort_unstable();
            sorted_counts[k / 2]
        };
        let threshold = median_count / 4;
        let mut by_count: Vec<usize> = (0..k).collect();
        by_count.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]));
        let mut donor = 0;
        for &small in by_count.iter().rev() {
            if counts[small] > threshold {
                break;
            }
            let big = by_count[donor];
            if counts[big] <= threshold {
                break;
            }
            let mean: Vec<f32> = (0..n)
                .map(|d| sums[big * n + d] / counts[big] as f32)
                .collect();
            let half = counts[big] / 2;
            let jitter_idx = rand::seq::index::sample(&mut rng, m, 1).into_vec()[0];
            let jitter_src = &data_flat[jitter_idx * n..(jitter_idx + 1) * n];
            for d in 0..n {
                let jitter = (jitter_src[d] - mean[d]) * 0.1;
                sums[small * n + d] = (mean[d] + jitter) * half as f32;
                sums[big * n + d] = (mean[d] - jitter) * half as f32;
            }
            counts[small] = half;
            counts[big] = half;
            donor += 1;
        }

        let mut centers_flat = vec![0f32; k * n];
        for i in 0..k {
            let src = &sums[i * n..(i + 1) * n];
            let dst = &mut centers_flat[i * n..(i + 1) * n];
            if counts[i] > 0 {
                let norm: f32 = src.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d = s / norm;
                    }
                }
            } else {
                let idx = rand::seq::index::sample(&mut rng, m, 1).into_vec()[0];
                let random = &data_flat[idx * n..(idx + 1) * n];
                let norm: f32 = random.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for (d, s) in dst.iter_mut().zip(random) {
                        *d = s / norm;
                    }
                }
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


fn encode_bucket_id(sub_indices: &[usize], k_sub: usize) -> u32 {
    let mut id = 0u32;
    let mut stride = 1u32;
    for &si in sub_indices {
        id += si as u32 * stride;
        stride *= k_sub as u32;
    }
    id
}

fn decode_bucket_id(id: u32, k_sub: usize, m: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(m);
    let mut rem = id as usize;
    for _ in 0..m {
        indices.push(rem % k_sub);
        rem /= k_sub;
    }
    indices
}

fn merge_and_write_buckets(
    db: &DB,
    tmpfiles: Vec<tempfile::NamedTempFile>,
    sub_centers_cpu: &[Tensor],
    k_sub: usize,
    generation_id: i64,
) -> Result<()> {
    let m = sub_centers_cpu.len();
    let mut merger = merger::Merger::from_tempfiles(tmpfiles, RESIDUAL_BYTES)?;
    for result in &mut merger {
        let entry = result?;
        let sub_indices = decode_bucket_id(entry.value, k_sub, m);
        let parts: Vec<Tensor> = sub_indices.iter().enumerate()
            .map(|(s, &si)| sub_centers_cpu[s].get(si))
            .collect::<candle_core::Result<_>>()?;
        let center = Tensor::cat(&parts, 0)?;
        let center_bytes = center.to_f32_bytes()?;
        let compressed_keys = compress_keys(&entry.keys);
        db.add_bucket(
            entry.value,
            generation_id,
            &center_bytes,
            &compressed_keys,
            &entry.data,
        )?;
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
    Ok(fts_matches)
}

pub fn reciprocal_rank_fusion(list1: &[DocPtr], list2: &[DocPtr], k: f64) -> Vec<DocPtr> {
    let mut scores: HashMap<DocPtr, f64> = HashMap::new();

    for (rank, &doc_id) in list1.iter().enumerate() {
        let score = 1.0 / (3.0 + k + rank as f64);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }

    for (rank, &doc_id) in list2.iter().enumerate() {
        let score = 1.0 / (k + rank as f64);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }

    let mut results: Vec<(DocPtr, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort descending by score
    let results: Vec<DocPtr> = results.iter().map(|&(idx, _)| idx).collect();
    results
}

/// Per-generation centroid data loaded from the database.
struct GenerationCentroids {
    generation_id: i64,
    bucket_ids: Vec<u32>,
    sizes: Vec<usize>,
    centers_matrix: Tensor,
    pq: Option<PqCentroids>,
}

struct PqCentroids {
    pq_groups: usize,
    k_sub: usize,
    sub_dim: usize,
    sub_centers: Vec<Tensor>,
    id_to_index: HashMap<u32, usize>,
}

type CentersCache = Vec<GenerationCentroids>;

static CACHED: Lazy<RwLock<Option<CentersCache>>> = Lazy::new(|| RwLock::new(None));

fn invalidate_center_cache() {
    *CACHED.write().unwrap() = None;
}

fn detect_pq_structure(
    bucket_ids: &[u32],
    centers_matrix: &Tensor,
    sizes: &[usize],
    pq_groups_log2: u32,
    device: &Device,
) -> Option<PqCentroids> {
    let max_id = *bucket_ids.iter().max()? as usize;
    let pq_groups = 1usize << pq_groups_log2;
    let k_sub = ((max_id + 1) as f64).powf(1.0 / pq_groups as f64).ceil() as usize;
    if k_sub < 2 {
        return None;
    }

    let sub_dim = EMBEDDING_DIM / pq_groups;
    let id_to_index: HashMap<u32, usize> = bucket_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let centers_cpu = centers_matrix.to_device(&Device::Cpu).ok()?;
    let mut accum = vec![vec![vec![0.0f64; sub_dim]; k_sub]; pq_groups];
    let mut counts = vec![vec![0usize; k_sub]; pq_groups];

    for (idx, &id) in bucket_ids.iter().enumerate() {
        let sub_indices = decode_bucket_id(id, k_sub, pq_groups);
        let w = sizes[idx];
        if w == 0 {
            continue;
        }
        let center = centers_cpu.get(idx).ok()?.to_vec1::<f32>().ok()?;
        for s in 0..pq_groups {
            let si = sub_indices[s];
            for d in 0..sub_dim {
                accum[s][si][d] += center[s * sub_dim + d] as f64 * w as f64;
            }
            counts[s][si] += w;
        }
    }

    let mut sub_centers = Vec::with_capacity(pq_groups);
    for s in 0..pq_groups {
        let mut flat = vec![0.0f32; k_sub * sub_dim];
        for i in 0..k_sub {
            if counts[s][i] > 0 {
                for d in 0..sub_dim {
                    flat[i * sub_dim + d] = (accum[s][i][d] / counts[s][i] as f64) as f32;
                }
            }
        }
        sub_centers.push(Tensor::from_vec(flat, (k_sub, sub_dim), device).ok()?);
    }

    debug!("detected PQ structure: m={} k_sub={}", pq_groups, k_sub);
    Some(PqCentroids {
        pq_groups,
        k_sub,
        sub_dim,
        sub_centers,
        id_to_index,
    })
}

fn get_all_generation_centers(db: &DB, device: &Device) -> Result<Vec<GenerationCentroids>> {
    let now = std::time::Instant::now();
    {
        let cache = CACHED.read().unwrap();
        if let Some(cached) = &*cache {
            debug!(
                "get_all_generation_centers cache hit ({} generations)",
                cached.len()
            );
            return Ok(cached.clone());
        }
    }

    let mut gen_query = db.query("SELECT id, level, pq_groups_log2 FROM generation ORDER BY level, id")?;
    let gens: Vec<(i64, u32, u32)> = gen_query
        .query_map((), |row| Ok((row.get::<_, i64>(0)?, row.get::<_, u32>(1)?, row.get::<_, u32>(2)?)))?
        .collect::<Result<Vec<_>, _>>()?;

    let mut all = Vec::with_capacity(gens.len());

    for (gen_id, _level, pq_groups_log2) in gens {
        let mut center_query = db.query(
            "SELECT id, length(residuals) / ?1, center FROM bucket
             WHERE generation_id = ?2 ORDER BY id",
        )?;
        let mut bucket_ids = vec![];
        let mut sizes = vec![];
        let mut centers = vec![];
        for result in center_query.query_map((RESIDUAL_BYTES as i64, gen_id), |row| {
            let id = row.get(0)?;
            let size = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            Ok((id, size, blob))
        })? {
            let (id, size, center) = result?;
            bucket_ids.push(id);
            sizes.push(size);
            let t = Tensor::from_f32_bytes(&center, EMBEDDING_DIM, &Device::Cpu)?.flatten_all()?;
            centers.push(t);
        }
        let centers_matrix = if !centers.is_empty() {
            Tensor::stack(&centers, 0)?.to_device(device)?
        } else {
            Tensor::zeros(&[0, EMBEDDING_DIM], DType::F32, device)?
        };

        let pq = if !bucket_ids.is_empty() {
            detect_pq_structure(&bucket_ids, &centers_matrix, &sizes, pq_groups_log2, device)
        } else {
            None
        };

        all.push(GenerationCentroids {
            generation_id: gen_id,
            bucket_ids,
            sizes,
            centers_matrix,
            pq,
        });
    }

    debug!(
        "reading centers for {} generations took {} ms",
        all.len(),
        now.elapsed().as_millis()
    );

    let mut cache = CACHED.write().unwrap();
    *cache = Some(all.clone());
    Ok(all)
}

impl Clone for PqCentroids {
    fn clone(&self) -> Self {
        Self {
            pq_groups: self.pq_groups,
            k_sub: self.k_sub,
            sub_dim: self.sub_dim,
            sub_centers: self.sub_centers.clone(),
            id_to_index: self.id_to_index.clone(),
        }
    }
}

impl Clone for GenerationCentroids {
    fn clone(&self) -> Self {
        Self {
            generation_id: self.generation_id,
            bucket_ids: self.bucket_ids.clone(),
            sizes: self.sizes.clone(),
            centers_matrix: self.centers_matrix.clone(),
            pq: self.pq.clone(),
        }
    }
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

    let t_prime_base: usize = std::env::var("WARP_T_PRIME").ok().and_then(|v| v.parse().ok()).unwrap_or(10000);
    let device = query_embeddings.device();
    let (m, _n) = query_embeddings.dims2()?;

    let mut all_q4_bytes: Vec<u8> = vec![];
    let mut q4_count = 0usize;
    let mut document_clusters: Vec<(usize, usize)> = vec![]; // (gen_idx, cluster_idx)
    let mut gen_centroid_scores_all: Vec<Vec<Vec<f32>>> = vec![];
    let mut all = vec![];
    let mut count = 0;
    let mut missing = vec![0.0f32; m];

    let generations = get_all_generation_centers(db, device)?;

    let table: [f32; 16] = packops::make_q4_dequant_table()?;

    for gen in &generations {
        if gen.bucket_ids.is_empty() {
            continue;
        }

        let gen_idx = gen_centroid_scores_all.len();
        let n_centroids = gen.bucket_ids.len();

        if let Some(pq) = &gen.pq {
            let k_sub = pq.k_sub;
            let pg = pq.pq_groups;
            let sub_dim = pq.sub_dim;
            let t_prime = 4 * t_prime_base * pg;

            // Score query tokens against each subspace's centroids.
            let sub_scores: Vec<Vec<Vec<f32>>> = (0..pg).map(|s| {
                let q_sub = query_embeddings.narrow(1, s * sub_dim, sub_dim)
                    .and_then(|t| t.contiguous())
                    .unwrap();
                fast_ops::matmul_t(&q_sub, &pq.sub_centers[s])
                    .and_then(|t| t.to_device(&Device::Cpu))
                    .and_then(|t| t.to_vec2::<f32>())
                    .unwrap()
            }).collect();

            // For each (token, subspace), sort centroid indices by descending score.
            // Use GPU arg_sort when there's only one PQ group (avoids CPU sort on large k_sub).
            let sub_sorted: Vec<Vec<Vec<u16>>> = if pg == 1 {
                let sim = fast_ops::matmul_t(query_embeddings, &pq.sub_centers[0])?;
                let sorted = sim.arg_sort_last_dim(false)?.to_device(&Device::Cpu)?;
                let sorted_vec = sorted.to_vec2::<u32>()?;
                vec![sorted_vec.into_iter().map(|row|
                    row.into_iter().map(|v| v as u16).collect()
                ).collect()]
            } else {
                (0..pg).map(|s| {
                    (0..m).map(|qi| {
                        let mut indices: Vec<u16> = (0..k_sub as u16).collect();
                        indices.sort_unstable_by(|&a, &b|
                            sub_scores[s][qi][b as usize].partial_cmp(&sub_scores[s][qi][a as usize]).unwrap());
                        indices
                    }).collect()
                }).collect()
            };

            let grid: usize = k_sub.checked_pow(pg as u32).unwrap();
            let mut token_visited: Vec<BitVec> = (0..m).map(|_| bitvec![0; grid]).collect();
            let mut global_selected = vec![false; n_centroids];
            let mut global_cumsum = 0usize;
            let mut topk_clusters = Vec::new();

            // Compute score for a rank tuple: sum of sub_scores[s][qi][sorted[s][qi][rank[s]]]
            let score_for = |qi: usize, ranks: &[u16]| -> f32 {
                (0..pg).map(|s| {
                    let ci = sub_sorted[s][qi][ranks[s] as usize] as usize;
                    sub_scores[s][qi][ci]
                }).sum()
            };

            // Compute flat index for a rank tuple in the k_sub^M grid.
            let flat_index = |ranks: &[u16]| -> usize {
                let mut idx = 0usize;
                let mut stride = 1usize;
                for &r in ranks {
                    idx += r as usize * stride;
                    stride *= k_sub;
                }
                idx
            };

            let mut heaps: Vec<max_heap::MaxHeap<Vec<u16>>> = Vec::with_capacity(m);
            let mut token_cumsum = vec![0usize; m];
            let mut token_done = vec![false; m];
            let origin = vec![0u16; pg];
            for qi in 0..m {
                let mut heap = max_heap::MaxHeap::new();
                heap.push(score_for(qi, &origin), origin.clone());
                token_visited[qi].set(0, true);
                heaps.push(heap);
            }

            loop {
                let mut any_progress = false;
                for qi in 0..m {
                    if token_done[qi] { continue; }
                    if let Some((_score, ranks)) = heaps[qi].pop() {
                        any_progress = true;
                        let sub_indices: Vec<usize> = (0..pg).map(|s|
                            sub_sorted[s][qi][ranks[s] as usize] as usize
                        ).collect();
                        let bucket_id = encode_bucket_id(&sub_indices, k_sub);

                        if let Some(&idx) = pq.id_to_index.get(&bucket_id) {
                            let sz = gen.sizes[idx];
                            token_cumsum[qi] += sz;
                            if !global_selected[idx] {
                                global_selected[idx] = true;
                                topk_clusters.push(idx as u32);
                                global_cumsum += sz;
                            }
                        }

                        if token_cumsum[qi] >= t_prime {
                            token_done[qi] = true;
                        } else {
                            for s in 0..pg {
                                let mut neighbor = ranks.clone();
                                neighbor[s] += 1;
                                if (neighbor[s] as usize) < k_sub {
                                    let fi = flat_index(&neighbor);
                                    if !token_visited[qi][fi] {
                                        token_visited[qi].set(fi, true);
                                        heaps[qi].push(score_for(qi, &neighbor), neighbor);
                                    }
                                }
                            }
                        }
                    } else {
                        token_done[qi] = true;
                        let worst: Vec<u16> = vec![k_sub as u16 - 1; pg];
                        missing[qi] = missing[qi].max(score_for(qi, &worst));
                    }
                }
                if !any_progress || token_done.iter().all(|&d| d) || global_cumsum >= t_prime {
                    let worst: Vec<u16> = vec![k_sub as u16 - 1; pg];
                    for qi in 0..m {
                        if !token_done[qi] {
                            missing[qi] = missing[qi].max(score_for(qi, &worst));
                        }
                    }
                    break;
                }
            }
            topk_clusters.sort_unstable();

            debug!("pq search: pg={} k_sub={} loading {} / {} buckets",
                   pg, k_sub, topk_clusters.len(), n_centroids);

            let mut gen_centroid_scores = vec![vec![0.0f32; n_centroids]; m];
            for &idx in &topk_clusters {
                let id = gen.bucket_ids[idx as usize];
                let sub_indices = decode_bucket_id(id, k_sub, pg);
                for qi in 0..m {
                    let score: f32 = (0..pg).map(|s| sub_scores[s][qi][sub_indices[s]]).sum();
                    gen_centroid_scores[qi][idx as usize] = score;
                }
            }
            gen_centroid_scores_all.push(gen_centroid_scores);

            {
                let placeholders: String = topk_clusters.iter().map(|i| gen.bucket_ids[*i as usize].to_string()).collect::<Vec<_>>().join(",");
                let sql = format!("SELECT id, indices, residuals FROM bucket WHERE id IN ({placeholders}) ORDER BY id");
                let mut batch_query = db.query(&sql)?;
                let rows = batch_query.query_map((), |row| {
                    Ok((row.get::<_, u32>(0)?, row.get::<_, Vec<u8>>(1)?, row.get::<_, Vec<u8>>(2)?))
                })?;
                let mut bucket_data: HashMap<u32, (Vec<u8>, Vec<u8>)> = HashMap::new();
                for r in rows {
                    let (id, keys, residuals) = r?;
                    bucket_data.insert(id, (keys, residuals));
                }
                drop(batch_query);
                for &i in &topk_clusters {
                    let bucket_id = gen.bucket_ids[i as usize];
                    let (keys_compressed, residual_bytes) = bucket_data.remove(&bucket_id).unwrap();
                    let document_indices = decompress_keys(&keys_compressed)?;
                    let num_docs = residual_bytes.len() / RESIDUAL_BYTES;
                    all_q4_bytes.extend_from_slice(&residual_bytes);
                    q4_count += num_docs;
                    for idx in &document_indices[..num_docs] {
                        document_clusters.push((gen_idx, i as usize));
                        all.push((*idx, count));
                        count += 1;
                    }
                }
            }
        } else {
            warn!("no PQ structure detected for generation {}, skipping", gen.generation_id);
        }
    }

    // Also load any unindexed chunks (documents not yet in any generation)
    let max_indexed_rowid: i64 = db
        .query("SELECT IFNULL(MAX(max_chunk_rowid), 0) FROM generation")?
        .query_row((), |row| row.get(0))?;

    let mut unindexed_embeddings = vec![];
    {
        let mut unindexed_query = db.query(
            "SELECT d.rowid, c.embeddings
             FROM document AS d
             JOIN chunk AS c ON c.hash = d.hash
             WHERE c.rowid > ?1
             ORDER BY d.rowid",
        )?;
        let results = unindexed_query.query_map((max_indexed_rowid,), |row| {
            Ok((row.get::<_, u32>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for result in results {
            let (id, embeddings) = result?;
            let embeddings =
                Tensor::embeddings_from_packed(&embeddings, EMBEDDING_DIM, &Device::Cpu)?;
            let (num_docs, _) = embeddings.dims2()?;
            unindexed_embeddings.push(embeddings);
            for _ in 0..num_docs {
                all.push(((id, 0), count));
                count += 1;
            }
        }
    }

    if count == 0 {
        return Ok(vec![]);
    }

    let n = m;
    let mut sim: Vec<f32> = Vec::with_capacity(count * n);

    // Process indexed embeddings: query·residuals + centroid scores
    if q4_count > 0 {
        let mut pair_table = [0u32; 256];
        for b in 0..256u16 {
            let hi = half::f16::from_f32(table[(b >> 4) as usize]).to_bits() as u32;
            let lo = half::f16::from_f32(table[(b & 0x0f) as usize]).to_bits() as u32;
            pair_table[b as usize] = hi | (lo << 16);
        }
        let total_elems = q4_count * EMBEDDING_DIM;
        let total_pairs = total_elems / 2;
        let mut dequant_u32: Vec<u32> = Vec::with_capacity(total_pairs);
        unsafe { dequant_u32.set_len(total_pairs); }
        let dst: *mut u32 = dequant_u32.as_mut_ptr();
        for (i, &byte) in all_q4_bytes.iter().enumerate() {
            unsafe { *dst.add(i) = pair_table[byte as usize]; }
        }
        let dequant_flat = unsafe {
            let ptr = dequant_u32.as_mut_ptr() as *mut half::f16;
            let len = total_elems;
            let cap = total_elems;
            std::mem::forget(dequant_u32);
            Vec::from_raw_parts(ptr, len, cap)
        };
        let all_residuals = Tensor::from_vec(dequant_flat, &[q4_count, EMBEDDING_DIM], device)?;

        let mut bias_flat = vec![0.0f32; q4_count * n];
        for (doc_idx, &(gen_idx, cluster_idx)) in
            document_clusters.iter().enumerate().take(q4_count)
        {
            let centroid_scores = &gen_centroid_scores_all[gen_idx];
            for (query_idx, scores) in centroid_scores.iter().enumerate().take(n) {
                bias_flat[doc_idx * n + query_idx] = scores[cluster_idx];
            }
        }
        let bias = Tensor::from_vec(bias_flat, &[q4_count, n], device)?;

        let query_f16 = query_embeddings.to_dtype(DType::F16)?;
        let residual_sims =
            fast_ops::matmul_t(&query_f16, &all_residuals)?.transpose(0, 1)?;
        let combined = (residual_sims.to_dtype(DType::F32)? + bias)?;
        let combined = combined.to_device(&Device::Cpu)?.contiguous()?;
        sim.extend_from_slice(&combined.flatten_all()?.to_vec1::<f32>()?);
    }

    // Process unindexed embeddings: full similarities
    if !unindexed_embeddings.is_empty() {
        let all_unindexed = Tensor::cat(&unindexed_embeddings, 0)?;
        let all_unindexed = all_unindexed.to_device(device)?;

        let unindexed_sims =
            fast_ops::matmul_t(query_embeddings, &all_unindexed)?.transpose(0, 1)?;
        let unindexed_sims = unindexed_sims.to_device(&Device::Cpu)?;
        let unindexed_sims = unindexed_sims.to_dtype(DType::F32)?.contiguous()?;
        let unindexed_sims_flat = unindexed_sims.flatten_all()?.to_vec1::<f32>()?;
        sim.extend_from_slice(&unindexed_sims_flat);
    }

    let missing_similarities = missing;

    let missing_score: f32 = missing_similarities.iter().sum::<f32>() / m as f32;
    let cutoff = if missing_score > threshold {
        missing_score
    } else {
        threshold
    };

    let row_at = |pos: usize| -> &[f32] {
        let start = pos * n;
        &sim[start..start + n]
    };

    all.sort_unstable();

    let mut sub_scores = vec![0.0f32; n];
    sub_scores.copy_from_slice(&missing_similarities);
    let mut doc_scores = vec![0.0f32; n];
    doc_scores.copy_from_slice(&missing_similarities);

    let mut scored_results: Vec<(f32, u32, u32)> = Vec::new();
    let mut prev_idx = 0u32;
    let mut prev_sub_idx = 0u32;

    let scaler = 1.0f32 / n as f32;
    for i in 0.. {

        let is_beyond_end = i == all.len();
        let ((idx, sub_idx), pos) = if is_beyond_end {
            ((u32::MAX, u32::MAX), 0)
        } else {
            all[i]
        };

        if i > 0 {
            let idx_change = prev_idx != idx;
            let sub_idx_change = idx_change || prev_sub_idx != sub_idx;

            if sub_idx_change {
                let sub_score = scaler * (sub_scores.iter().copied().sum::<f32>());
                if sub_score > cutoff {
                    scored_results.push((sub_score, prev_idx, prev_sub_idx));
                }
                vmax_inplace(&mut doc_scores, &sub_scores);
                sub_scores.copy_from_slice(&doc_scores);
            }
            if idx_change {
                doc_scores.copy_from_slice(&missing_similarities);
                sub_scores.copy_from_slice(&missing_similarities);
            }
        }

        if is_beyond_end {
            break;
        }

        let row = row_at(pos);
        vmax_inplace(&mut sub_scores, row);

        assert!(i == 0 || prev_idx <= idx);
        assert!(i == 0 || (prev_idx != idx || prev_sub_idx <= sub_idx));
        prev_idx = idx;
        prev_sub_idx = sub_idx;
    }

    scored_results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());


    let results = match sql_filter {
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
        filtered
        }
        None => {
            scored_results.truncate(top_k);
            scored_results
        }
    };

    debug!(
        "match_centroids: {} embeddings in {} ms.",
        count,
        total_start.elapsed().as_millis(),
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
            WHERE chunk.hash IS NULL AND length(document.body) > 0
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
        WHERE chunk.hash IS NULL AND length(document.body) > 0
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
                return Err(anyhow::anyhow!("add_chunk failed: {v}"));
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
    let total_chunk_embeddings = count_chunk_embeddings(db)?;
    let indexed = count_indexed_embeddings(db)?;
    Ok(total_chunk_embeddings.saturating_sub(indexed))
}

/// Count total embeddings across all chunks by summing the counts column
/// (not byte length, which is haar-packed and compressed).
fn count_chunk_embeddings(db: &DB) -> Result<usize> {
    let mut query = db.query("SELECT counts FROM chunk")?;
    let mut total = 0usize;
    let results = query.query_map((), |row| row.get::<_, String>(0))?;
    for result in results {
        let counts_str = result?;
        let n: usize = counts_str
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .sum();
        total += n;
    }
    Ok(total)
}

fn count_indexed_embeddings(db: &DB) -> Result<usize> {
    let count: usize = db
        .query(&format!(
            "SELECT IFNULL(SUM(length(residuals)/{RESIDUAL_BYTES}), 0) FROM bucket"
        ))?
        .query_row((), |row| row.get::<_, usize>(0))?;
    Ok(count)
}

fn sample_embeddings_for_kmeans(db: &DB, sql: &str, device: &Device) -> Result<(Tensor, usize)> {
    let mut kmeans_query = db.query(sql)?;
    let mut total_embeddings = 0;
    #[cfg(feature = "deterministic")]
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    #[cfg(not(feature = "deterministic"))]
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
    if all_embeddings.is_empty() {
        return Ok((Tensor::zeros(&[0, EMBEDDING_DIM], DType::F32, device)?, 0));
    }
    let matrix = Tensor::stack(&all_embeddings, 0)?.to_device(device)?;
    Ok((matrix, total_embeddings))
}

fn pq_groups_log2_for_level(level: u32) -> u32 {
    std::env::var("WARP_PQ_GROUPS_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(level / 3)
}

fn run_kmeans_for_index(matrix: &Tensor, total_embeddings: usize, level: u32) -> Result<(Vec<Tensor>, usize, usize, u32)> {
    let now = std::time::Instant::now();
    let pq_groups_log2 = pq_groups_log2_for_level(level);
    let pq_groups = 1usize << pq_groups_log2;
    assert!(EMBEDDING_DIM % pq_groups == 0, "EMBEDDING_DIM must be divisible by pq_groups");
    let sub_dim = EMBEDDING_DIM / pq_groups;

    let pg = pq_groups as f64;
    let mut k_sub = (16.0f64.powf(1.0 / pg) * (total_embeddings as f64).powf(0.5 / pg)).ceil() as usize;
    k_sub = k_sub.max(2);
    let (n_rows, _) = matrix.dims2()?;
    if k_sub > n_rows / 2 {
        k_sub = (n_rows / 2).max(2);
    }
    debug!("total_embeddings={} pq_groups={} k_sub={} k_eff={}",
           total_embeddings, pq_groups, k_sub,
           k_sub.checked_pow(pq_groups as u32).unwrap_or(usize::MAX));

    let scale = 1.0 / (pq_groups as f64).sqrt();
    let mut sub_centers = Vec::with_capacity(pq_groups);
    for s in 0..pq_groups {
        let sub = matrix.narrow(1, s * sub_dim, sub_dim)?.contiguous()?;
        sub_centers.push((kmeans(&sub, k_sub, 5)? * scale)?);
    }
    debug!("pq kmeans took {} ms.", now.elapsed().as_millis());
    Ok((sub_centers, k_sub, pq_groups, pq_groups_log2))
}

fn level_capacity(level: u32) -> usize {
    L0_CAPACITY * LSM_FANOUT.pow(level + 1)
}

/// Build one generation for a range of chunks, reading original embeddings from
/// the chunk table and running full k-means.
fn build_layer(
    db: &DB,
    device: &Device,
    level: u32,
    min_rowid: i64,
    max_rowid: i64,
) -> Result<()> {
    let sql = format!(
        "SELECT chunk.embeddings FROM chunk
         WHERE chunk.rowid >= {} AND chunk.rowid <= {}",
        min_rowid, max_rowid
    );
    let (matrix, total_embeddings) = sample_embeddings_for_kmeans(db, &sql, device)?;
    if total_embeddings == 0 {
        return Ok(());
    }

    info!(
        "building L{} with {} embeddings (chunks {}..={})",
        level, total_embeddings, min_rowid, max_rowid
    );
    let (sub_centers, k_sub, _pq_groups, pq_groups_log2) = run_kmeans_for_index(&matrix, total_embeddings, level)?;
    drop(matrix);

    let (tmpfiles, sub_centers_cpu) =
        write_buckets_for_range(db, &sub_centers, k_sub, device, total_embeddings as u64, min_rowid, max_rowid)?;

    let gen_id = db.add_generation(level, pq_groups_log2, total_embeddings as u64, min_rowid, max_rowid)?;
    merge_and_write_buckets(db, tmpfiles, &sub_centers_cpu, k_sub, gen_id)?;

    Ok(())
}

fn write_buckets_for_range(
    db: &DB,
    sub_centers: &[Tensor],
    k_sub: usize,
    device: &Device,
    expected_count: u64,
    min_rowid: i64,
    max_rowid: i64,
) -> Result<(Vec<tempfile::NamedTempFile>, Vec<Tensor>)> {
    let _priority_mgr = PriorityManager::new();
    let pq_groups = sub_centers.len();
    let sub_dim = EMBEDDING_DIM / pq_groups;
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let bar = progress::new_with_label(expected_count, "indexing");

    let mut document_indices = Vec::<(u32, u32)>::new();
    let mut all_embeddings = vec![];

    let embeddings_sql = format!(
        "SELECT document.rowid, chunk.embeddings, chunk.counts
         FROM document, chunk
         WHERE document.hash = chunk.hash
         AND chunk.rowid >= {} AND chunk.rowid <= {}
         ORDER BY document.rowid",
        min_rowid, max_rowid
    );

    let mut query = db.query(&embeddings_sql)?;

    let mut results = query.query_map((), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    let mut done = false;
    let mut batch = 0;
    let mut tmpfiles = vec![];
    let sub_centers_cpu: Vec<Tensor> = sub_centers.iter()
        .map(|c| c.to_device(&Device::Cpu))
        .collect::<candle_core::Result<_>>()?;
    let mut packed_subs = Vec::with_capacity(pq_groups);
    for c in sub_centers {
        packed_subs.push(fast_ops::PackedRight::new(c)?);
    }
    while !done {
        match results.next() {
            Some(result) => {
                let (id, embeddings, counts) = result?;

                let t = Tensor::embeddings_from_packed(&embeddings, EMBEDDING_DIM, &Device::Cpu)?;
                let split = split_tensor(&t);
                let n_emb = split.len();

                for (i, count) in counts
                    .split(',')
                    .filter_map(|s| s.parse::<u32>().ok())
                    .enumerate()
                {
                    for _ in 0..count {
                        document_indices.push((id, i as u32));
                    }
                }
                all_embeddings.extend(split);
                batch += n_emb;
            }
            None => {
                done = true;
            }
        }

        let batch_size = 0x10000;

        if batch >= batch_size || done {
            let now = std::time::Instant::now();

            let take = batch.min(batch_size);
            let remaining = batch - take;

            let embeddings = all_embeddings.split_off(remaining);
            let indices = document_indices.split_off(remaining);
            let data = Tensor::cat(&embeddings, 0)?.to_device(device)?;

            // Compute per-subspace assignments and combine into bucket IDs.
            let mut sub_assignments: Vec<Vec<u32>> = Vec::with_capacity(pq_groups);
            for s in 0..pq_groups {
                let data_sub = data.narrow(1, s * sub_dim, sub_dim)?.contiguous()?;
                let assignments = matmul_argmax_batched(&data_sub, &packed_subs[s], 1024)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?;
                sub_assignments.push(assignments);
            }
            let cluster_assignments: Vec<u32> = (0..take).map(|i| {
                let subs: Vec<usize> = (0..pq_groups).map(|s| sub_assignments[s][i] as usize).collect();
                encode_bucket_id(&subs, k_sub)
            }).collect();
            mmuls_total += now.elapsed().as_millis();

            let now = std::time::Instant::now();
            let mut writer = merger::Writer::new(RESIDUAL_BYTES)?;

            let mut pairs: Vec<(usize, u32)> = cluster_assignments
                .iter()
                .enumerate()
                .map(|(i, &bucket)| (i, bucket))
                .collect();
            pairs.sort_by_key(|&(_, bucket)| bucket);

            let mut keys: Vec<(u32, u32)> = Vec::with_capacity(take);
            let mut residuals_bytes: Vec<u8> = Vec::with_capacity(take * RESIDUAL_BYTES);
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

                let sub_indices = decode_bucket_id(bucket, k_sub, pq_groups);
                let parts: Vec<Tensor> = sub_indices.iter().enumerate()
                    .map(|(s, &si)| sub_centers_cpu[s].get(si))
                    .collect::<candle_core::Result<_>>()?;
                let center = Tensor::cat(&parts, 0)?;
                let residual = (embeddings[sample].get(0) - &center)?;
                let residual_quantized = residual.compand()?.quantize(4)?.to_q4_bytes()?;
                residuals_bytes.extend(&residual_quantized);
            }
            tmpfiles.push(writer.finish()?);
            writes_total += now.elapsed().as_millis();
            bar.inc(take as u64);

            batch = remaining;
        }
    }
    bar.finish();

    debug!("mmuls took {} ms.", mmuls_total);
    debug!("writes took {} ms.", writes_total);

    Ok((tmpfiles, sub_centers_cpu))
}

pub fn full_index(db: &DB, device: &Device) -> Result<()> {
    db.execute("DELETE FROM bucket")?;
    db.execute("DELETE FROM generation")?;
    invalidate_center_cache();
    index_chunks(db, device)
}

pub fn index_chunks(db: &DB, device: &Device) -> Result<()> {
    let x = count_unindexed_embeddings(db)?;
    if x == 0 {
        return Ok(());
    }

    let indexed = count_indexed_embeddings(db)?;
    info!("database has {} unindexed embeddings ({} indexed)", x, indexed);

    if x < L0_CAPACITY {
        debug!("buffering {} unindexed embeddings (< {} threshold)", x, L0_CAPACITY);
        return Ok(());
    }

    // Find the max indexed chunk rowid — unindexed chunks are above this
    let max_indexed_rowid: i64 = db
        .query("SELECT IFNULL(MAX(max_chunk_rowid), 0) FROM generation")?
        .query_row((), |row| row.get(0))?;
    let max_chunk_rowid: i64 = db
        .query("SELECT MAX(rowid) FROM chunk")?
        .query_row((), |row| row.get(0))?;

    // Cascade: accumulate x through levels until we find one with room
    let mut total = x;
    let mut target_level = 0u32;

    loop {
        let cap = level_capacity(target_level);
        let level_size: usize = db
            .query(&format!(
                "SELECT IFNULL(SUM(num_embeddings), 0) FROM generation WHERE level = {}",
                target_level
            ))?
            .query_row((), |row| row.get::<_, usize>(0))?;
        total += level_size;
        if total <= cap {
            break;
        }
        target_level += 1;
    }

    // Find the min chunk rowid across all levels being merged + unindexed
    let min_rowid: i64 = db
        .query(&format!(
            "SELECT IFNULL(MIN(min_chunk_rowid), {}) FROM generation WHERE level <= {}",
            max_indexed_rowid + 1,
            target_level
        ))?
        .query_row((), |row| row.get(0))?;

    info!(
        "cascading {} embeddings into L{} (chunks {}..={})",
        total, target_level, min_rowid, max_chunk_rowid
    );

    db.begin_transaction()?;

    // Delete all generations at levels 0..=target_level (they get merged)
    let delete_sql = format!(
        "DELETE FROM bucket WHERE generation_id IN \
         (SELECT id FROM generation WHERE level <= {})",
        target_level
    );
    db.execute(&delete_sql)?;
    let delete_sql = format!("DELETE FROM generation WHERE level <= {}", target_level);
    db.execute(&delete_sql)?;

    build_layer(db, device, target_level, min_rowid, max_chunk_rowid)?;

    db.commit_transaction()?;
    invalidate_center_cache();
    db.checkpoint();
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
        reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
    } else {
        sem_idxs
    };
    fused.truncate(top_k);

    let mut results = vec![];
    let mut body_query = db.query("SELECT metadata,body,lens,date FROM document WHERE rowid = ?1")?;
    for (idx, sub_idx) in fused {
        let tuple : DocPtr = (idx, sub_idx);
        let score = match scores.get(&tuple) {
            Some(score) => *score,
            None => 0.0f32,
        };
        let (metadata, bodies, date) = body_query.query_row((idx,), |row| {
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
        })?;

        let sub = (sub_idx as usize).min(bodies.len().saturating_sub(1)) as u32;
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
