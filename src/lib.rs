use log::{debug, info, warn};
use once_cell::sync::Lazy;
#[cfg(any(test, feature = "deterministic"))]
use rand::SeedableRng;
#[cfg(any(feature = "sqlite", feature = "capi-embed-cache"))]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

mod app_id;

// Conditionally compile encoder backend based on features
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

#[cfg(feature = "modernbert")]
pub mod modernbert;
#[cfg(feature = "modernbert")]
use modernbert as t5_encoder;

#[cfg(feature = "modernbert-quantized")]
pub mod quantized_modernbert;
#[cfg(feature = "modernbert-quantized")]
use quantized_modernbert as t5_encoder;

// Compile-time checks: exactly one encoder backend required.
#[cfg(not(any(
    feature = "t5-quantized",
    feature = "t5-openvino",
    feature = "modernbert",
    feature = "modernbert-quantized",
)))]
compile_error!("Must enable exactly one encoder backend: t5-quantized, t5-openvino, modernbert, or modernbert-quantized");

#[cfg(any(
    all(feature = "t5-quantized", feature = "t5-openvino"),
    all(feature = "t5-quantized", feature = "modernbert"),
    all(feature = "t5-quantized", feature = "modernbert-quantized"),
    all(feature = "t5-openvino", feature = "modernbert"),
    all(feature = "t5-openvino", feature = "modernbert-quantized"),
    all(feature = "modernbert", feature = "modernbert-quantized"),
))]
compile_error!("Cannot enable multiple encoder backends simultaneously");

// hybrid-dequant is a CPU-only optimization and cannot be used with Metal
#[cfg(all(feature = "hybrid-dequant", feature = "metal"))]
compile_error!("hybrid-dequant is incompatible with metal (use accelerate only for CPU, or metal without hybrid-dequant for GPU)");

#[cfg(feature = "sqlite")]
mod db;
#[cfg(feature = "sqlite")]
pub use db::DB;

mod embedding_cache;
pub use embedding_cache::{CachedEmbeddings, EmbeddingCache, FileEmbeddingCache};

mod embedder;
pub use embedder::Embedder;

pub mod assets;

mod packops;
use packops::TensorPackOps;

mod haarops;

pub mod rans64;

mod merger;

mod file_writer;
use file_writer::NewFileWriter;

mod file_index;
use file_index::{
    active_rowid_records, nway_merge_rowid_records, read_rowid_records,
    rowid_records_embedding_count, sort_dedup_rowid_records, sync_parent_dir,
    write_rowid_records_to_writer, FileBackedIndex, FileIndexGeneration, RowidRecord,
};

mod priority;
use priority::PriorityManager;

#[cfg(feature = "sqlite")]
mod progress_reporter;

pub mod types;
pub use types::SqlStatementInternal;

#[cfg(feature = "sqlite")]
pub mod sql_generator;

#[cfg(feature = "sqlite")]
mod sqlite_index;
#[cfg(feature = "sqlite")]
pub use sqlite_index::{
    count_unindexed_cached_embeddings, count_unindexed_embeddings,
    count_unindexed_embeddings_with_cache, embed_chunks, embed_chunks_with_cache, fulltext_search,
    index_chunks, index_chunks_with_cache, index_chunks_with_cache_and_options,
    index_chunks_with_options, match_centroids, match_centroids_with_cache, search,
    search_cached_rowids_with_cache, search_rowids, semantic_index_unavailable_reason,
};
#[cfg(all(test, feature = "sqlite"))]
pub(crate) use sqlite_index::fts5_query;

#[cfg(feature = "napi")]
#[allow(dead_code)]
mod napi;

#[cfg(feature = "python")]
mod python;

mod capi;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};

const DEFAULT_EMBEDDING_DIM: usize = 128;
#[cfg(any(feature = "sqlite", feature = "capi-embed-cache"))]
const DOCUMENT_CACHE_HASH_CHARS: usize = 32;
const BUCKET_DATA_APP_ID: u32 = file_index::GENERATION_DATA_APP_ID;
const BUCKET_DATA_VERSION: u32 = file_index::GENERATION_DATA_VERSION;
const BUCKET_SCALAR_META_BYTES: usize = 20;
const BUCKET_DATA_HEADER_BYTES: usize = file_index::GENERATION_DATA_HEADER_BYTES;
pub(crate) const BUCKET_CENTER_FORMAT: u32 = 3;
#[cfg(not(test))]
const L0_CAPACITY: usize = 1024;
#[cfg(test)]
const L0_CAPACITY: usize = 4;

#[cfg(not(test))]
const LSM_FANOUT: usize = 16;
#[cfg(test)]
const LSM_FANOUT: usize = 2;
const COARSE_TREE_BRANCHING: usize = 16;
const INDEX_KMEANS_BRANCHING: usize = 128;
const INDEX_ASSIGNMENT_BEAM: usize = 2;
const KMEANS_MATMUL_BATCH: usize = 4096;
const INDEX_KMEANS_ITERATIONS: usize = 5;
const INDEX_BATCH_SIZE: usize = 0x10000;
const INDEX_TARGET_BUCKET_VECTORS: usize = INDEX_BATCH_SIZE / INDEX_KMEANS_BRANCHING;
const BUCKET_READ_COALESCE_GAP_BYTES: usize = 16 * 1024;

/// A document pointer combining document ID and sub-chunk index
/// Allows precise location of results within subdivided documents
pub type DocPtr = (u32, u32);

#[derive(Clone, Copy, Debug)]
pub struct IndexOptions {
    residual_quant_bits: u8,
}

impl IndexOptions {
    pub fn new(residual_quant_bits: u8) -> Result<Self> {
        packops::validate_residual_quant_bits(residual_quant_bits)?;
        Ok(Self {
            residual_quant_bits,
        })
    }

    pub fn residual_quant_bits(&self) -> u8 {
        self.residual_quant_bits
    }
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            residual_quant_bits: packops::DEFAULT_RESIDUAL_QUANT_BITS,
        }
    }
}

fn f32_center_bytes_for_dim(dim: usize) -> usize {
    dim * std::mem::size_of::<f32>()
}

fn center_block_bytes_for_count(
    centroid_count: usize,
    dim: usize,
    center_format: u32,
) -> Result<usize> {
    match center_format {
        BUCKET_CENTER_FORMAT => {
            let scale_len = centroid_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| anyhow::anyhow!("bucket center scale block length overflow"))?;
            let row_bytes = packops::signed_q4_row_bytes(dim);
            centroid_count
                .checked_mul(row_bytes)
                .and_then(|code_len| scale_len.checked_add(code_len))
                .ok_or_else(|| anyhow::anyhow!("bucket center block length overflow"))
        }
        _ => anyhow::bail!("bucket center format {center_format} is not supported"),
    }
}

fn residual_bytes_for_dim(dim: usize, residual_quant_bits: u8) -> Result<usize> {
    packops::temp_residual_bytes_for_dim(dim, residual_quant_bits)
}

fn model_id_prefix() -> &'static str {
    #[cfg(any(feature = "t5-quantized", feature = "t5-openvino"))]
    {
        "xtr-base-en"
    }
    #[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
    {
        "modernbert"
    }
}

fn model_id_for_dim(dim: usize) -> String {
    let prefix = model_id_prefix();
    let prefix = if document_token_conv_stride2() {
        format!("{prefix}-conv2")
    } else {
        prefix.to_string()
    };
    if dim == DEFAULT_EMBEDDING_DIM {
        prefix
    } else {
        format!("{prefix}-d{dim}")
    }
}

fn document_token_conv_stride2() -> bool {
    std::env::var("WARP_DOC_TOKEN_CONV_STRIDE2")
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

pub fn default_embedding_cache_dir() -> PathBuf {
    PathBuf::from(".embeddings-cache").join(model_id_prefix())
}

pub fn default_embedding_cache() -> FileEmbeddingCache {
    FileEmbeddingCache::new(default_embedding_cache_dir())
}

fn dim_from_model_id(model: &str) -> usize {
    model
        .rsplit_once("-d")
        .and_then(|(_, dim)| dim.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBEDDING_DIM)
}

fn cached_embeddings_match_current_encoder(embeddings: &CachedEmbeddings) -> bool {
    embeddings.model == model_id_for_dim(dim_from_model_id(&embeddings.model))
}

#[cfg(any(feature = "sqlite", feature = "capi-embed-cache"))]
pub(crate) fn document_cache_hash(body: &str, lens: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    hasher.update(lens.as_bytes());
    let hash: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    hash[..DOCUMENT_CACHE_HASH_CHARS].to_string()
}

fn load_cached_embeddings(
    cache: &dyn EmbeddingCache,
    rowid: u64,
    hash: &str,
) -> Result<Option<CachedEmbeddings>> {
    match cache.get_for_document(rowid, hash)? {
        Some(embeddings) if cached_embeddings_match_current_encoder(&embeddings) => {
            debug!("embedding cache hit for chunk {hash}");
            Ok(Some(embeddings))
        }
        Some(embeddings) => {
            debug!(
                "embedding cache entry for chunk {hash} has stale model {}; recomputing",
                embeddings.model
            );
            Ok(None)
        }
        None => Ok(None),
    }
}

pub fn make_device() -> Device {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        let previous_panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let metal_device = std::panic::catch_unwind(|| Device::new_metal(0));
        std::panic::set_hook(previous_panic_hook);
        match metal_device {
            Ok(Ok(device)) => device,
            Ok(Err(v)) => {
                warn!("unable to create metal device: {v}");
                Device::Cpu
            }
            Err(_) => {
                warn!("unable to create metal device: initialization panicked");
                Device::Cpu
            }
        }
    } else if cfg!(feature = "cuda") {
        match Device::new_cuda(0) {
            Ok(device) => device,
            Err(v) => {
                warn!("unable to create cuda device: {v}");
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

fn matmul_argmax_scores_batched(
    t: &Tensor,
    centers: &fast_ops::PackedRight,
    batch_size: usize,
) -> Result<Vec<(u32, f32)>> {
    let (m, _n) = t.dims2()?;
    let mut assignments = Vec::with_capacity(m);

    for start in (0..m).step_by(batch_size) {
        let end = (start + batch_size).min(m);
        let batch_len = end - start;
        let batch = t.narrow(0, start, batch_len)?;
        let sim = centers.matmul(&batch)?.to_device(&Device::Cpu)?;
        for row in sim.to_vec2::<f32>()? {
            let mut best_idx = 0u32;
            let mut best_score = f32::NEG_INFINITY;
            for (idx, score) in row.into_iter().enumerate() {
                if score > best_score {
                    best_idx = idx as u32;
                    best_score = score;
                }
            }
            assignments.push((best_idx, best_score));
        }
    }

    Ok(assignments)
}

fn matmul_top2_batched(
    t: &Tensor,
    centers: &fast_ops::PackedRight,
    batch_size: usize,
) -> Result<Vec<(u32, u32)>> {
    let (m, _n) = t.dims2()?;
    let mut assignments = Vec::with_capacity(m);

    for start in (0..m).step_by(batch_size) {
        let end = (start + batch_size).min(m);
        let batch_len = end - start;
        let batch = t.narrow(0, start, batch_len)?;
        let sim = centers.matmul(&batch)?.to_device(&Device::Cpu)?;
        for row in sim.to_vec2::<f32>()? {
            let mut best_idx = 0u32;
            let mut second_idx = 0u32;
            let mut best_score = f32::NEG_INFINITY;
            let mut second_score = f32::NEG_INFINITY;
            for (idx, score) in row.into_iter().enumerate() {
                if score > best_score {
                    second_score = best_score;
                    second_idx = best_idx;
                    best_score = score;
                    best_idx = idx as u32;
                } else if score > second_score {
                    second_score = score;
                    second_idx = idx as u32;
                }
            }
            assignments.push((best_idx, second_idx));
        }
    }

    Ok(assignments)
}

fn kmeans_inner(
    data: &Tensor,
    k: usize,
    max_iter: usize,
    bar: Option<&progress::Bar>,
) -> Result<Tensor> {
    let (m, n) = data.dims2()?;
    debug!("kmeans k={} m={} n={}...", k, m, n);
    if k == 1 {
        let data_flat = data.flatten_all()?.to_vec1::<f32>()?;
        let mut center = vec![0f32; n];
        for row in data_flat.chunks_exact(n) {
            for (dst, value) in center.iter_mut().zip(row) {
                *dst += value;
            }
        }
        let norm: f32 = center.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut center {
                *value /= norm;
            }
        }
        if let Some(bar) = bar {
            bar.inc(1);
        }
        return Ok(Tensor::from_vec(center, (1, n), data.device())?);
    }

    let _priority_mgr = PriorityManager::new();
    let device = data.device();

    #[cfg(any(test, feature = "deterministic"))]
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    #[cfg(not(any(test, feature = "deterministic")))]
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
        let cluster_assignments = matmul_argmax_batched(data, &packed_centers, KMEANS_MATMUL_BATCH)?;
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
            assert!(norm > 0.0);
            for (d, e) in dst.iter_mut().zip(emb) {
                *d = e / norm;
            }
        }

        centers = Tensor::from_vec(centers_flat, (k, n), device)?;
        if let Some(bar) = bar {
            bar.inc(k as u64);
        }
    }
    Ok(centers)
}

fn kmeans(data: &Tensor, k: usize, max_iter: usize) -> Result<Tensor> {
    let total = if k == 1 { 1 } else { max_iter * k };
    let bar = progress::new_with_label(total as u64, "kmeans");
    let centers = kmeans_inner(data, k, max_iter, Some(&bar))?;
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

struct BucketSidecarMeta {
    size: usize,
    data_offset: u64,
    indices_len: usize,
    residual_len: usize,
}

struct BucketDataHeader {
    centroid_count: usize,
    embedding_dim: usize,
    residual_quant_bits: u8,
    center_format: u32,
    meta_offset: usize,
    centers_offset: usize,
    payload_offset: usize,
    rowids_offset: usize,
}

struct CenterSidecarData {
    bucket_indices: Vec<usize>,
    stored_center_block: Vec<u8>,
    residual_centers_cpu: Tensor,
    coarse_tree: CoarseTree,
}

fn write_bucket_data_header(
    writer: &mut impl Write,
    centroid_count: usize,
    embedding_dim: usize,
    residual_quant_bits: u8,
    center_format: u32,
    meta_offset: usize,
    payload_offset: usize,
    rowids_offset: usize,
) -> Result<()> {
    writer.write_all(&BUCKET_DATA_APP_ID.to_le_bytes())?;
    writer.write_all(&BUCKET_DATA_VERSION.to_le_bytes())?;
    writer.write_all(&(residual_quant_bits as u32).to_le_bytes())?;
    writer.write_all(&u32::try_from(centroid_count)?.to_le_bytes())?;
    writer.write_all(&u32::try_from(embedding_dim)?.to_le_bytes())?;
    writer.write_all(&center_format.to_le_bytes())?;
    writer.write_all(&u32::try_from(meta_offset)?.to_le_bytes())?;
    writer.write_all(&u32::try_from(payload_offset)?.to_le_bytes())?;
    writer.write_all(&u64::try_from(rowids_offset)?.to_le_bytes())?;
    Ok(())
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    anyhow::ensure!(
        offset + std::mem::size_of::<u32>() <= bytes.len(),
        "u32 field at offset {offset} is truncated"
    );
    Ok(u32::from_le_bytes(bytes[offset..offset + 4].try_into()?))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64> {
    anyhow::ensure!(
        offset + std::mem::size_of::<u64>() <= bytes.len(),
        "u64 field at offset {offset} is truncated"
    );
    Ok(u64::from_le_bytes(bytes[offset..offset + 8].try_into()?))
}

fn coarse_tree_leaf_parent_centers(
    tree: &CoarseTree,
    leaf_count: usize,
    dim: usize,
) -> Result<Option<(Vec<usize>, Vec<u8>)>> {
    if tree.levels.is_empty() {
        return Ok(None);
    }
    anyhow::ensure!(
        tree.levels.len() == 1,
        "bucket center residuals require a single-level coarse tree"
    );
    let level = &tree.levels[0];
    let (parent_count, parent_dim) = level.centers.dims2()?;
    anyhow::ensure!(
        parent_count == level.children.len(),
        "coarse tree parent count does not match child lists"
    );
    anyhow::ensure!(
        parent_dim == dim,
        "coarse tree parent dimension {parent_dim} does not match center dimension {dim}"
    );
    let mut parents = vec![usize::MAX; leaf_count];
    for (parent_idx, children) in level.children.iter().enumerate() {
        for &leaf_idx in children {
            anyhow::ensure!(
                leaf_idx < leaf_count,
                "coarse tree leaf {leaf_idx} is outside leaf count {leaf_count}"
            );
            anyhow::ensure!(
                parents[leaf_idx] == usize::MAX,
                "coarse tree leaf {leaf_idx} has multiple parents"
            );
            parents[leaf_idx] = parent_idx;
        }
    }
    for (leaf_idx, &parent_idx) in parents.iter().enumerate() {
        anyhow::ensure!(
            parent_idx != usize::MAX,
            "coarse tree is missing parent for leaf {leaf_idx}"
        );
    }
    let parent_bytes = level.centers.to_device(&Device::Cpu)?.to_f32_bytes()?;
    Ok(Some((parents, parent_bytes)))
}

fn bucket_leaf_indices(
    bucket_indices: &[usize],
    centroid_count: usize,
) -> Result<Vec<usize>> {
    let mut leaf_by_bucket = vec![usize::MAX; centroid_count];
    for (leaf_idx, &bucket_idx) in bucket_indices.iter().enumerate() {
        anyhow::ensure!(
            bucket_idx < centroid_count,
            "bucket index {bucket_idx} is outside centroid count {centroid_count}"
        );
        anyhow::ensure!(
            leaf_by_bucket[bucket_idx] == usize::MAX,
            "bucket index {bucket_idx} appears more than once"
        );
        leaf_by_bucket[bucket_idx] = leaf_idx;
    }
    Ok(leaf_by_bucket)
}

fn parent_value(
    parent_info: Option<&(Vec<usize>, Vec<u8>)>,
    leaf_idx: usize,
    dim: usize,
    dim_idx: usize,
) -> Result<f32> {
    let Some((parents, parent_bytes)) = parent_info else {
        return Ok(0.0);
    };
    let parent_idx = parents[leaf_idx];
    let offset = parent_idx
        .checked_mul(f32_center_bytes_for_dim(dim))
        .and_then(|offset| offset.checked_add(dim_idx * std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("parent center offset overflow"))?;
    Ok(f32::from_le_bytes(parent_bytes[offset..offset + 4].try_into()?))
}

fn encode_q4_center_residual_block(
    f32_block: &[u8],
    bucket_indices: &[usize],
    coarse_tree: &CoarseTree,
    centroid_count: usize,
    dim: usize,
) -> Result<Vec<u8>> {
    let f32_row_bytes = f32_center_bytes_for_dim(dim);
    let expected_f32_len = centroid_count
        .checked_mul(f32_row_bytes)
        .ok_or_else(|| anyhow::anyhow!("f32 center block length overflow"))?;
    anyhow::ensure!(
        f32_block.len() == expected_f32_len,
        "f32 center block length is invalid"
    );
    let scale_len = centroid_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("q4 center scale block length overflow"))?;
    let packed_row_bytes = packops::signed_q4_row_bytes(dim);
    let code_len = centroid_count
        .checked_mul(packed_row_bytes)
        .ok_or_else(|| anyhow::anyhow!("q4 center code block length overflow"))?;
    let encoded_len = scale_len
        .checked_add(code_len)
        .ok_or_else(|| anyhow::anyhow!("q4 center block length overflow"))?;
    let mut encoded = vec![0; encoded_len];
    let leaf_by_bucket = bucket_leaf_indices(bucket_indices, centroid_count)?;
    let parent_info = coarse_tree_leaf_parent_centers(coarse_tree, bucket_indices.len(), dim)?;

    for centroid_idx in 0..centroid_count {
        let f32_start = centroid_idx
            .checked_mul(f32_row_bytes)
            .ok_or_else(|| anyhow::anyhow!("f32 center row offset overflow"))?;
        let leaf_idx = leaf_by_bucket[centroid_idx];
        let mut max_abs = 0.0f32;
        for dim_idx in 0..dim {
            let offset = f32_start + dim_idx * std::mem::size_of::<f32>();
            let value = f32::from_le_bytes(f32_block[offset..offset + 4].try_into()?);
            anyhow::ensure!(value.is_finite(), "bucket center contains non-finite value");
            let base = if leaf_idx == usize::MAX {
                0.0
            } else {
                parent_value(parent_info.as_ref(), leaf_idx, dim, dim_idx)?
            };
            max_abs = max_abs.max((value - base).abs());
        }

        let scale = packops::signed_q4_scale(max_abs);
        let scale_offset = centroid_idx * std::mem::size_of::<f32>();
        encoded[scale_offset..scale_offset + 4].copy_from_slice(&scale.to_le_bytes());
        if scale == 0.0 {
            continue;
        }

        let code_start = scale_len + centroid_idx * packed_row_bytes;
        let code_end = code_start + packed_row_bytes;
        let codes = &mut encoded[code_start..code_end];
        for dim_idx in 0..dim {
            let offset = f32_start + dim_idx * std::mem::size_of::<f32>();
            let value = f32::from_le_bytes(f32_block[offset..offset + 4].try_into()?);
            let base = if leaf_idx == usize::MAX {
                0.0
            } else {
                parent_value(parent_info.as_ref(), leaf_idx, dim, dim_idx)?
            };
            let packed = packops::quantize_signed_q4(value - base, scale);
            packops::write_signed_q4_code(codes, dim_idx, packed);
        }
    }

    Ok(encoded)
}

fn decode_q4_center_residual_block(
    q4_block: &[u8],
    bucket_indices: &[usize],
    coarse_tree: &CoarseTree,
    centroid_count: usize,
    dim: usize,
) -> Result<Vec<u8>> {
    let expected_len = center_block_bytes_for_count(centroid_count, dim, BUCKET_CENTER_FORMAT)?;
    anyhow::ensure!(
        q4_block.len() == expected_len,
        "q4 center block length is invalid"
    );
    let scale_len = centroid_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("q4 center scale block length overflow"))?;
    let f32_row_bytes = f32_center_bytes_for_dim(dim);
    let f32_len = centroid_count
        .checked_mul(f32_row_bytes)
        .ok_or_else(|| anyhow::anyhow!("f32 center block length overflow"))?;
    let mut decoded = vec![0; f32_len];
    let leaf_by_bucket = bucket_leaf_indices(bucket_indices, centroid_count)?;
    let parent_info = coarse_tree_leaf_parent_centers(coarse_tree, bucket_indices.len(), dim)?;
    let packed_row_bytes = packops::signed_q4_row_bytes(dim);

    for centroid_idx in 0..centroid_count {
        let scale_offset = centroid_idx * std::mem::size_of::<f32>();
        let scale = f32::from_le_bytes(q4_block[scale_offset..scale_offset + 4].try_into()?);
        anyhow::ensure!(
            scale.is_finite() && scale >= 0.0,
            "q4 center scale is invalid"
        );
        let code_start = scale_len + centroid_idx * packed_row_bytes;
        let f32_start = centroid_idx
            .checked_mul(f32_row_bytes)
            .ok_or_else(|| anyhow::anyhow!("f32 center row offset overflow"))?;
        let leaf_idx = leaf_by_bucket[centroid_idx];
        for (byte_idx, &byte) in q4_block[code_start..code_start + packed_row_bytes]
            .iter()
            .enumerate()
        {
            let values = packops::signed_q4_pair_values(byte);
            let dim_idx = byte_idx * 2;
            for lane in 0..2 {
                let dim_idx = dim_idx + lane;
                if dim_idx >= dim {
                    break;
                }
                let base = if leaf_idx == usize::MAX {
                    0.0
                } else {
                    parent_value(parent_info.as_ref(), leaf_idx, dim, dim_idx)?
                };
                let value = base + values[lane] * scale;
                let offset = f32_start + dim_idx * std::mem::size_of::<f32>();
                decoded[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }

    Ok(decoded)
}

fn center_block_to_f32_bytes(
    center_block: &[u8],
    bucket_indices: &[usize],
    coarse_tree: &CoarseTree,
    centroid_count: usize,
    dim: usize,
    center_format: u32,
) -> Result<Vec<u8>> {
    match center_format {
        BUCKET_CENTER_FORMAT => {
            decode_q4_center_residual_block(
                center_block,
                bucket_indices,
                coarse_tree,
                centroid_count,
                dim,
            )
        }
        _ => anyhow::bail!("bucket center format {center_format} is not supported"),
    }
}

fn write_u64_le(bytes: &mut Vec<u8>, value: usize) -> Result<()> {
    bytes.write_all(&u64::try_from(value)?.to_le_bytes())?;
    Ok(())
}

fn read_tree_u64(bytes: &[u8], cursor: &mut usize) -> Result<usize> {
    let end = cursor
        .checked_add(std::mem::size_of::<u64>())
        .ok_or_else(|| anyhow::anyhow!("coarse tree cursor overflow"))?;
    anyhow::ensure!(end <= bytes.len(), "coarse tree data is truncated");
    let value = u64::from_le_bytes(bytes[*cursor..end].try_into()?);
    *cursor = end;
    Ok(value.try_into()?)
}

#[cfg(all(not(unix), windows))]
fn positioned_read(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Storage::FileSystem::ReadFile;
    use windows::Win32::System::IO::OVERLAPPED;

    let read_len = buf.len().min(u32::MAX as usize);
    let buf = &mut buf[..read_len];
    let mut bytes_read = 0u32;
    let mut overlapped = OVERLAPPED::default();
    unsafe {
        overlapped.Anonymous.Anonymous.Offset = offset as u32;
        overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        ReadFile(
            HANDLE(file.as_raw_handle()),
            Some(buf),
            Some(&mut bytes_read),
            Some(&mut overlapped),
        )
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
    }
    Ok(bytes_read as usize)
}

#[cfg(not(any(unix, windows)))]
fn positioned_read(_file: &File, _buf: &mut [u8], _offset: u64) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "bucket positioned reads are only implemented on Unix and Windows",
    ))
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
    while !buf.is_empty() {
        let n = positioned_read(file, buf, offset)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positioned read",
            ));
        }
        offset += n as u64;
        buf = &mut buf[n..];
    }
    Ok(())
}

#[cfg(unix)]
fn max_iov() -> usize {
    let n = unsafe { libc::sysconf(libc::_SC_IOV_MAX) };
    if n > 0 {
        n as usize
    } else {
        1024
    }
}

#[cfg(unix)]
fn preadv_once(file: &File, offset: u64, buffers: &mut [&mut [u8]]) -> io::Result<usize> {
    use std::os::fd::AsRawFd;
    let mut iovecs: Vec<libc::iovec> = buffers
        .iter_mut()
        .map(|buf| libc::iovec {
            iov_base: (*buf).as_mut_ptr().cast(),
            iov_len: buf.len(),
        })
        .collect();
    let n = unsafe {
        libc::preadv(
            file.as_raw_fd(),
            iovecs.as_mut_ptr(),
            iovecs.len().try_into().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "too many iovecs")
            })?,
            offset.try_into().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "preadv offset overflow")
            })?,
        )
    };
    if n < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

#[cfg(unix)]
fn readv_one_exact_at(file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
    while !buf.is_empty() {
        let n = {
            let mut single = [&mut *buf];
            preadv_once(file, offset, &mut single)?
        };
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short vectored positioned read",
            ));
        }
        offset += n as u64;
        let (_, rest) = buf.split_at_mut(n);
        buf = rest;
    }
    Ok(())
}

#[cfg(unix)]
fn readv_group_exact_at(file: &File, offset: u64, buffers: &mut [&mut [u8]]) -> io::Result<()> {
    let total = buffers.iter().map(|buf| buf.len()).sum::<usize>();
    if total == 0 {
        return Ok(());
    }
    let n = preadv_once(file, offset, buffers)?;
    if n == total {
        return Ok(());
    }
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short vectored positioned read",
        ));
    }

    let mut skipped = n;
    let mut cursor = offset;
    for buf in buffers.iter_mut() {
        if skipped >= buf.len() {
            skipped -= buf.len();
            cursor += buf.len() as u64;
            continue;
        }
        if skipped > 0 {
            let skip = skipped;
            skipped = 0;
            readv_one_exact_at(file, cursor + skip as u64, &mut buf[skip..])?;
        } else {
            readv_one_exact_at(file, cursor, buf)?;
        }
        cursor += buf.len() as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn readv_exact_at(file: &File, mut offset: u64, buffers: &mut [&mut [u8]]) -> io::Result<()> {
    let mut rest = buffers;
    let max_iov = max_iov();
    while !rest.is_empty() {
        let take = rest.len().min(max_iov);
        let (group, tail) = rest.split_at_mut(take);
        readv_group_exact_at(file, offset, group)?;
        offset += group.iter().map(|buf| buf.len() as u64).sum::<u64>();
        rest = tail;
    }
    Ok(())
}

#[cfg(not(unix))]
fn readv_exact_at(file: &File, mut offset: u64, buffers: &mut [&mut [u8]]) -> io::Result<()> {
    for buf in buffers {
        read_exact_at(file, offset, buf)?;
        offset += buf.len() as u64;
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_contiguous_exact_at(file: &File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut data = vec![0; len];
    read_exact_at(file, offset, &mut data)?;
    Ok(data)
}

fn bucket_data_header_from_prefix(prefix: &[u8], sidecar_len: usize) -> Result<BucketDataHeader> {
    anyhow::ensure!(
        sidecar_len >= BUCKET_DATA_HEADER_BYTES,
        "bucket sidecar is too small: {} bytes",
        sidecar_len
    );
    anyhow::ensure!(
        prefix.len() >= std::mem::size_of::<u32>(),
        "bucket sidecar header is too small"
    );
    let app_id = read_u32_le(prefix, 0)?;
    anyhow::ensure!(
        app_id == BUCKET_DATA_APP_ID,
        "bucket sidecar app id {app_id:#x} is not supported"
    );
    let version = read_u32_le(prefix, std::mem::size_of::<u32>())?;
    anyhow::ensure!(
        version == BUCKET_DATA_VERSION,
        "bucket sidecar version {version} is not supported"
    );
    anyhow::ensure!(
        BUCKET_DATA_HEADER_BYTES <= prefix.len(),
        "bucket sidecar header read was too short"
    );
    let mut offset = 2 * std::mem::size_of::<u32>();
    let residual_quant_bits: u8 = read_u32_le(prefix, offset)?.try_into()?;
    packops::validate_residual_quant_bits(residual_quant_bits)?;
    offset += std::mem::size_of::<u32>();
    let centroid_count: usize = read_u32_le(prefix, offset)?.try_into()?;
    offset += std::mem::size_of::<u32>();
    let embedding_dim: usize = read_u32_le(prefix, offset)?.try_into()?;
    anyhow::ensure!(embedding_dim > 0, "bucket sidecar embedding dimension is zero");
    offset += std::mem::size_of::<u32>();
    let center_format = read_u32_le(prefix, offset)?;
    match center_format {
        BUCKET_CENTER_FORMAT => {}
        _ => anyhow::bail!("bucket center format {center_format} is not supported"),
    }
    offset += std::mem::size_of::<u32>();
    let meta_offset: usize = read_u32_le(prefix, offset)?.try_into()?;
    offset += std::mem::size_of::<u32>();
    let payload_offset: usize = read_u32_le(prefix, offset)?.try_into()?;
    offset += std::mem::size_of::<u32>();
    let rowids_offset: usize = read_u64_le(prefix, offset)?.try_into()?;
    let scalar_meta_len = centroid_count
        .checked_mul(BUCKET_SCALAR_META_BYTES)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar metadata length overflow"))?;
    let centers_offset = meta_offset
        .checked_add(scalar_meta_len)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar centers offset overflow"))?;
    let centers_len = center_block_bytes_for_count(centroid_count, embedding_dim, center_format)?;
    let expected_payload_offset = centers_offset
        .checked_add(centers_len)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar payload offset overflow"))?;
    anyhow::ensure!(
        meta_offset >= BUCKET_DATA_HEADER_BYTES
            && payload_offset == expected_payload_offset
            && rowids_offset >= payload_offset
            && rowids_offset <= sidecar_len,
        "bucket sidecar layout is invalid"
    );
    Ok(BucketDataHeader {
        centroid_count,
        embedding_dim,
        residual_quant_bits,
        center_format,
        meta_offset,
        centers_offset,
        payload_offset,
        rowids_offset,
    })
}

fn bucket_data_header_from_file(file: &File, sidecar_len: usize) -> Result<BucketDataHeader> {
    let prefix_len = BUCKET_DATA_HEADER_BYTES.min(sidecar_len);
    let mut prefix = vec![0; prefix_len];
    let mut buffers = [prefix.as_mut_slice()];
    readv_exact_at(file, 0, &mut buffers)?;
    bucket_data_header_from_prefix(&prefix, sidecar_len)
}

fn validate_residual_radius_levels(levels: &[f32], residual_quant_bits: u8) -> Result<()> {
    packops::validate_residual_quant_bits(residual_quant_bits)?;
    anyhow::ensure!(
        residual_quant_bits != 0,
        "Lloyd-Max radius levels require polar residual quantization"
    );
    let expected = 1usize << usize::from(residual_quant_bits);
    anyhow::ensure!(
        levels.len() == expected,
        "Lloyd-Max radius level count {} does not match expected {}",
        levels.len(),
        expected
    );
    let mut previous = f32::NEG_INFINITY;
    for &level in levels {
        anyhow::ensure!(level.is_finite(), "Lloyd-Max radius level is not finite");
        anyhow::ensure!(
            level >= 0.0,
            "Lloyd-Max radius level {level} must be non-negative"
        );
        anyhow::ensure!(
            level >= previous,
            "Lloyd-Max radius levels must be sorted"
        );
        previous = level;
    }
    Ok(())
}

fn residual_radius_levels_byte_len(residual_quant_bits: u8) -> Result<usize> {
    packops::validate_residual_quant_bits(residual_quant_bits)?;
    if residual_quant_bits == 0 {
        return Ok(0);
    }
    Ok((1usize << usize::from(residual_quant_bits)) * std::mem::size_of::<f32>())
}

fn residual_radius_levels_to_bytes(
    levels: Option<&[f32]>,
    residual_quant_bits: u8,
) -> Result<Vec<u8>> {
    let Some(levels) = levels else {
        return Ok(vec![]);
    };
    validate_residual_radius_levels(levels, residual_quant_bits)?;
    let mut bytes = Vec::with_capacity(residual_radius_levels_byte_len(residual_quant_bits)?);
    for &level in levels {
        bytes.write_all(&level.to_le_bytes())?;
    }
    Ok(bytes)
}

fn bucket_data_residual_radius_levels_from_file(
    file: &File,
    header: &BucketDataHeader,
) -> Result<Option<Vec<f32>>> {
    let levels_len = header
        .meta_offset
        .checked_sub(BUCKET_DATA_HEADER_BYTES)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar radius level range is invalid"))?;
    if levels_len == 0 {
        return Ok(None);
    }
    let expected_len = residual_radius_levels_byte_len(header.residual_quant_bits)?;
    anyhow::ensure!(
        levels_len == expected_len,
        "bucket sidecar Lloyd-Max radius block has {levels_len} bytes, expected {expected_len}"
    );
    let mut bytes = vec![0; levels_len];
    let mut buffers = [bytes.as_mut_slice()];
    readv_exact_at(file, u64::try_from(BUCKET_DATA_HEADER_BYTES)?, &mut buffers)?;
    let levels: Vec<f32> = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    validate_residual_radius_levels(&levels, header.residual_quant_bits)?;
    Ok(Some(levels))
}

fn bucket_data_bucket_meta_from_file(
    file: &File,
    header: &BucketDataHeader,
) -> Result<(Vec<BucketSidecarMeta>, Vec<u8>)> {
    let meta_len = header
        .centers_offset
        .checked_sub(header.meta_offset)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar metadata range is invalid"))?;
    let mut meta_block = vec![0; meta_len];
    let centers_len = header
        .payload_offset
        .checked_sub(header.centers_offset)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar center range is invalid"))?;
    let mut center_block = vec![0; centers_len];
    let mut buffers = [meta_block.as_mut_slice(), center_block.as_mut_slice()];
    readv_exact_at(file, u64::try_from(header.meta_offset)?, &mut buffers)?;
    let metas = bucket_data_bucket_meta_from_block(&meta_block, header)?;
    Ok((metas, center_block))
}

fn bucket_data_bucket_meta_from_block(
    meta_block: &[u8],
    header: &BucketDataHeader,
) -> Result<Vec<BucketSidecarMeta>> {
    anyhow::ensure!(
        meta_block.len() == header.centers_offset - header.meta_offset,
        "bucket sidecar metadata block length is invalid"
    );
    let mut metas = Vec::with_capacity(header.centroid_count);
    for bucket_idx in 0..header.centroid_count {
        let offset = bucket_idx * BUCKET_SCALAR_META_BYTES;
        let size = read_u32_le(meta_block, offset)? as usize;
        let data_offset: u64 = read_u64_le(meta_block, offset + 4)?;
        let indices_len = read_u32_le(meta_block, offset + 12)? as usize;
        let residual_len = read_u32_le(meta_block, offset + 16)? as usize;
        anyhow::ensure!(
            data_offset >= u64::try_from(header.payload_offset)?,
            "bucket {bucket_idx} data offset is before payload block"
        );
        let bucket_end = data_offset
            .checked_add(u64::try_from(indices_len)?)
            .and_then(|offset| offset.checked_add(u64::try_from(residual_len).ok()?))
            .ok_or_else(|| anyhow::anyhow!("bucket {bucket_idx} data length overflow"))?;
        anyhow::ensure!(
            bucket_end <= u64::try_from(header.rowids_offset)?,
            "bucket {bucket_idx} data ends beyond bucket payload"
        );
        metas.push(BucketSidecarMeta {
            size,
            data_offset,
            indices_len,
            residual_len,
        });
    }
    Ok(metas)
}

fn bucket_data_payload_end(
    header: &BucketDataHeader,
    metas: &[BucketSidecarMeta],
) -> Result<usize> {
    let mut payload_end = header.payload_offset;
    for (bucket_idx, meta) in metas.iter().enumerate() {
        let data_offset: usize = meta.data_offset.try_into()?;
        let bucket_end = data_offset
            .checked_add(meta.indices_len)
            .and_then(|offset| offset.checked_add(meta.residual_len))
            .ok_or_else(|| anyhow::anyhow!("bucket {bucket_idx} data length overflow"))?;
        payload_end = payload_end.max(bucket_end);
    }
    anyhow::ensure!(
        payload_end <= header.rowids_offset,
        "bucket payload ends beyond rowid offset"
    );
    Ok(payload_end)
}

fn bucket_data_coarse_tree_from_file(
    file: &File,
    header: &BucketDataHeader,
    metas: &[BucketSidecarMeta],
    device: &Device,
) -> Result<CoarseTree> {
    let payload_end = bucket_data_payload_end(header, metas)?;
    let tree_len = header
        .rowids_offset
        .checked_sub(payload_end)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar tree range is invalid"))?;
    anyhow::ensure!(
        tree_len >= std::mem::size_of::<u64>(),
        "bucket sidecar is missing coarse tree data; run ./warp-cli reindex"
    );
    let mut tree_bytes = vec![0; tree_len];
    let mut buffers = [tree_bytes.as_mut_slice()];
    readv_exact_at(file, u64::try_from(payload_end)?, &mut buffers)?;
    deserialize_coarse_tree(&tree_bytes, device)
}

fn write_bucket_meta(writer: &mut impl Write, meta: &BucketSidecarMeta) -> Result<()> {
    writer.write_all(&u32::try_from(meta.size)?.to_le_bytes())?;
    writer.write_all(&meta.data_offset.to_le_bytes())?;
    writer.write_all(&u32::try_from(meta.indices_len)?.to_le_bytes())?;
    writer.write_all(&u32::try_from(meta.residual_len)?.to_le_bytes())?;
    Ok(())
}

fn build_coarse_tree_from_bucket_indices(
    bucket_indices: &[usize],
    center_block: &[u8],
    center_dim: usize,
) -> Result<CoarseTree> {
    let center_bytes = f32_center_bytes_for_dim(center_dim);
    anyhow::ensure!(
        center_block.len() % center_bytes == 0,
        "bucket center block length is invalid"
    );
    let centroid_count = center_block.len() / center_bytes;
    for &bucket_idx in bucket_indices {
        anyhow::ensure!(
            bucket_idx < centroid_count,
            "bucket index {bucket_idx} is outside centroid count {centroid_count}"
        );
    }
    if bucket_indices.is_empty() {
        Ok(CoarseTree { levels: vec![] })
    } else {
        let centers = Tensor::from_f32_bytes(center_block, center_dim, &Device::Cpu)?;
        let indices: Vec<u32> = bucket_indices
            .iter()
            .map(|&idx| u32::try_from(idx))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let index = Tensor::from_slice(indices.as_slice(), (indices.len(),), &Device::Cpu)?;
        let centers = centers.index_select(&index, 0)?;
        build_coarse_tree_from_centers(&centers, &Device::Cpu)
    }
}

fn prepare_center_sidecar_data(
    centers_cpu: &Tensor,
    bucket_indices: Vec<usize>,
) -> Result<CenterSidecarData> {
    let (centroid_count, center_dim) = centers_cpu.dims2()?;
    let center_block = centers_cpu.to_f32_bytes()?;
    let expected_center_block_len = centroid_count
        .checked_mul(f32_center_bytes_for_dim(center_dim))
        .ok_or_else(|| anyhow::anyhow!("bucket center block length overflow"))?;
    anyhow::ensure!(
        center_block.len() == expected_center_block_len,
        "bucket center block length is invalid"
    );
    let coarse_tree =
        build_coarse_tree_from_bucket_indices(&bucket_indices, &center_block, center_dim)?;
    let stored_center_block = encode_q4_center_residual_block(
        &center_block,
        &bucket_indices,
        &coarse_tree,
        centroid_count,
        center_dim,
    )?;
    let residual_center_block = center_block_to_f32_bytes(
        &stored_center_block,
        &bucket_indices,
        &coarse_tree,
        centroid_count,
        center_dim,
        BUCKET_CENTER_FORMAT,
    )?;
    let residual_centers_cpu =
        Tensor::from_f32_bytes(&residual_center_block, center_dim, &Device::Cpu)?;
    Ok(CenterSidecarData {
        bucket_indices,
        stored_center_block,
        residual_centers_cpu,
        coarse_tree,
    })
}

fn coarse_tree_leaf_order(tree: &CoarseTree, leaf_count: usize) -> Result<Vec<usize>> {
    let mut order: Vec<usize> = (0..leaf_count).collect();
    if tree.levels.is_empty() {
        return Ok(order);
    }

    let mut keys = vec![vec![usize::MAX; tree.levels.len()]; leaf_count];
    for (level_idx, level) in tree.levels.iter().enumerate() {
        for (node_idx, leaves) in level.children.iter().enumerate() {
            for &leaf in leaves {
                anyhow::ensure!(
                    leaf < leaf_count,
                    "coarse tree leaf {leaf} is outside leaf count {leaf_count}"
                );
                keys[leaf][level_idx] = node_idx;
            }
        }
    }
    for (leaf, key) in keys.iter().enumerate() {
        anyhow::ensure!(
            key.iter().all(|&node| node != usize::MAX),
            "coarse tree is missing leaf {leaf}"
        );
    }

    order.sort_by(|&a, &b| {
        for level_idx in 0..tree.levels.len() {
            let cmp = keys[a][level_idx].cmp(&keys[b][level_idx]);
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        a.cmp(&b)
    });
    Ok(order)
}

fn copy_exact_bytes(reader: &mut File, writer: &mut impl Write, len: usize) -> Result<()> {
    let mut limited = reader.take(u64::try_from(len)?);
    let copied = std::io::copy(&mut limited, writer)?;
    anyhow::ensure!(
        copied == u64::try_from(len)?,
        "bucket payload copy ended after {copied} bytes, expected {len}"
    );
    Ok(())
}

fn merge_and_write_buckets_to_path(
    tmpfiles: Vec<tempfile::NamedTempFile>,
    center_sidecar: &CenterSidecarData,
    rowid_records: &[RowidRecord],
    final_path: &Path,
    residual_quant_bits: u8,
    residual_radius_levels: Option<&[f32]>,
) -> Result<()> {
    let mut tmp_path = final_path.as_os_str().to_os_string();
    tmp_path.push(format!(
        ".{}.tmp",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    let tmp_path = PathBuf::from(tmp_path);

    let temp_dir = final_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let data_tmp = tempfile::NamedTempFile::new_in(temp_dir)?;
    let mut data_writer = data_tmp.reopen()?;
    let (centroid_count, center_dim) = center_sidecar.residual_centers_cpu.dims2()?;
    let residual_bytes = residual_bytes_for_dim(center_dim, residual_quant_bits)?;
    let mut bucket_meta: Vec<BucketSidecarMeta> = (0..centroid_count)
        .map(|_| BucketSidecarMeta {
            size: 0,
            data_offset: 0,
            indices_len: 0,
            residual_len: 0,
        })
        .collect();
    let mut data_offset = 0u64;

    let mut merger = merger::Merger::from_tempfiles(tmpfiles, residual_bytes)?;
    for result in &mut merger {
        let entry = result?;
        let bucket_idx = entry.value as usize;
        anyhow::ensure!(
            bucket_idx < centroid_count,
            "bucket {} is outside centroid count {centroid_count}",
            entry.value
        );
        let compressed_keys = compress_keys(&entry.keys);
        anyhow::ensure!(
            entry.data.len() % residual_bytes == 0,
            "bucket {} residual byte length {} is not divisible by residual width {}",
            entry.value,
            entry.data.len(),
            residual_bytes
        );
        data_writer.write_all(&compressed_keys)?;
        data_writer.write_all(&entry.data)?;
        let meta = &mut bucket_meta[bucket_idx];
        meta.size = entry.data.len() / residual_bytes;
        meta.data_offset = data_offset;
        meta.indices_len = compressed_keys.len();
        meta.residual_len = entry.data.len();
        let payload_len = compressed_keys
            .len()
            .checked_add(entry.data.len())
            .ok_or_else(|| anyhow::anyhow!("bucket sidecar payload length overflow"))?;
        data_offset = data_offset
            .checked_add(u64::try_from(payload_len)?)
            .ok_or_else(|| anyhow::anyhow!("bucket sidecar data offset overflow"))?;
    }
    data_writer.flush()?;
    drop(data_writer);

    let nonempty_buckets: Vec<usize> = bucket_meta
        .iter()
        .enumerate()
        .filter_map(|(bucket_idx, meta)| (meta.size > 0).then_some(bucket_idx))
        .collect();
    anyhow::ensure!(
        nonempty_buckets == center_sidecar.bucket_indices,
        "routed bucket set changed while writing bucket payloads"
    );
    let coarse_tree_bytes = serialize_coarse_tree(&center_sidecar.coarse_tree)?;
    let leaf_order = coarse_tree_leaf_order(&center_sidecar.coarse_tree, nonempty_buckets.len())?;
    let mut payload_chunks = Vec::with_capacity(nonempty_buckets.len());
    for leaf_idx in leaf_order {
        let bucket_idx = nonempty_buckets[leaf_idx];
        let meta = &bucket_meta[bucket_idx];
        let len = meta
            .indices_len
            .checked_add(meta.residual_len)
            .ok_or_else(|| anyhow::anyhow!("bucket sidecar payload length overflow"))?;
        payload_chunks.push((bucket_idx, meta.data_offset, len));
    }

    let meta_block_len = centroid_count
        .checked_mul(BUCKET_SCALAR_META_BYTES)
        .ok_or_else(|| anyhow::anyhow!("bucket metadata block length overflow"))?;
    let residual_radius_level_block =
        residual_radius_levels_to_bytes(residual_radius_levels, residual_quant_bits)?;
    let meta_offset = BUCKET_DATA_HEADER_BYTES
        .checked_add(residual_radius_level_block.len())
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar metadata offset overflow"))?;
    let centers_offset = meta_offset
        .checked_add(meta_block_len)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar center block offset overflow"))?;
    let payload_start = centers_offset
        .checked_add(center_sidecar.stored_center_block.len())
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar payload start overflow"))?;
    let payload_start = u64::try_from(payload_start)?;
    let payload_len = payload_chunks
        .iter()
        .try_fold(0u64, |total, &(_, _, len)| {
            total
                .checked_add(u64::try_from(len).ok()?)
        })
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar payload length overflow"))?;
    anyhow::ensure!(
        payload_len == data_offset,
        "bucket sidecar payload reorder length {payload_len} does not match merged length {data_offset}"
    );
    let rowids_offset = payload_start
        .checked_add(payload_len)
        .and_then(|offset| offset.checked_add(u64::try_from(coarse_tree_bytes.len()).ok()?))
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar rowid offset overflow"))?;

    let mut bucket_data_writer = NewFileWriter::create_new(&tmp_path)?;

    write_bucket_data_header(
        &mut bucket_data_writer,
        centroid_count,
        center_dim,
        residual_quant_bits,
        BUCKET_CENTER_FORMAT,
        meta_offset,
        payload_start.try_into()?,
        rowids_offset.try_into()?,
    )?;
    bucket_data_writer.write_all(&residual_radius_level_block)?;
    for meta in &mut bucket_meta {
        if meta.size == 0 {
            meta.data_offset = payload_start;
        }
    }
    let mut reordered_offset = payload_start;
    for &(bucket_idx, _, len) in &payload_chunks {
        let meta = &mut bucket_meta[bucket_idx];
        meta.data_offset = reordered_offset;
        reordered_offset = reordered_offset
            .checked_add(u64::try_from(len)?)
            .ok_or_else(|| anyhow::anyhow!("bucket sidecar absolute offset overflow"))?;
    }
    anyhow::ensure!(
        reordered_offset == payload_start + payload_len,
        "bucket sidecar reordered payload length mismatch"
    );
    for meta in &bucket_meta {
        write_bucket_meta(&mut bucket_data_writer, meta)?;
    }
    bucket_data_writer.write_all(&center_sidecar.stored_center_block)?;
    let mut data_reader = data_tmp.reopen()?;
    for &(_, temp_offset, len) in &payload_chunks {
        data_reader.seek(SeekFrom::Start(temp_offset))?;
        copy_exact_bytes(&mut data_reader, &mut bucket_data_writer, len)?;
    }
    bucket_data_writer.write_all(&coarse_tree_bytes)?;
    write_rowid_records_to_writer(&mut bucket_data_writer, rowid_records)?;
    bucket_data_writer.finish()?;
    std::fs::rename(&tmp_path, &final_path)?;
    sync_parent_dir(final_path)?;
    Ok(())
}

const HYBRID_FULLTEXT_RRF_WEIGHT: f64 = 0.075;
const HYBRID_SEMANTIC_RRF_WEIGHT: f64 = 1.0;

pub fn reciprocal_rank_fusion(list1: &[DocPtr], list2: &[DocPtr], k: f64) -> Vec<DocPtr> {
    weighted_reciprocal_rank_fusion(list1, 1.0, list2, 1.0, k)
}

pub fn hybrid_reciprocal_rank_fusion(
    fulltext: &[DocPtr],
    semantic: &[DocPtr],
    k: f64,
) -> Vec<DocPtr> {
    weighted_reciprocal_rank_fusion(
        fulltext,
        HYBRID_FULLTEXT_RRF_WEIGHT,
        semantic,
        HYBRID_SEMANTIC_RRF_WEIGHT,
        k,
    )
}

pub fn weighted_reciprocal_rank_fusion(
    list1: &[DocPtr],
    list1_weight: f64,
    list2: &[DocPtr],
    list2_weight: f64,
    k: f64,
) -> Vec<DocPtr> {
    #[derive(Clone, Copy)]
    struct FusedDoc {
        score: f64,
        best_ptr: DocPtr,
        best_contribution: f64,
    }

    fn add_contribution(scores: &mut HashMap<u32, FusedDoc>, doc_ptr: DocPtr, contribution: f64) {
        scores
            .entry(doc_ptr.0)
            .and_modify(|entry| {
                entry.score += contribution;
                if contribution > entry.best_contribution
                    || (contribution == entry.best_contribution && doc_ptr < entry.best_ptr)
                {
                    entry.best_ptr = doc_ptr;
                    entry.best_contribution = contribution;
                }
            })
            .or_insert(FusedDoc {
                score: contribution,
                best_ptr: doc_ptr,
                best_contribution: contribution,
            });
    }

    fn add_ranked_list(
        scores: &mut HashMap<u32, FusedDoc>,
        list: &[DocPtr],
        weight: f64,
        k: f64,
    ) {
        if weight == 0.0 {
            return;
        }
        for (rank, &doc_id) in list.iter().enumerate() {
            let score = weight / (k + rank as f64 + 1.0);
            add_contribution(scores, doc_id, score);
        }
    }

    let mut scores: HashMap<u32, FusedDoc> = HashMap::new();
    add_ranked_list(&mut scores, list1, list1_weight, k);
    add_ranked_list(&mut scores, list2, list2_weight, k);

    let mut results: Vec<(u32, FusedDoc)> = scores.into_iter().collect();
    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap()
            .then_with(|| a.0.cmp(&b.0))
    });
    results.iter().map(|(_, fused)| fused.best_ptr).collect()
}

#[cfg(test)]
mod reciprocal_rank_fusion_tests {
    use super::*;

    #[test]
    fn reciprocal_rank_fusion_is_symmetric_between_rankers() {
        let a = [(1, 0), (2, 0), (3, 0)];
        let b = [(3, 0), (2, 0), (1, 0)];

        assert_eq!(
            reciprocal_rank_fusion(&a, &b, 60.0),
            reciprocal_rank_fusion(&b, &a, 60.0)
        );
    }

    #[test]
    fn reciprocal_rank_fusion_promotes_documents_found_by_both_rankers() {
        let fulltext = [(1, 0), (2, 0), (3, 0)];
        let semantic = [(4, 0), (3, 0), (5, 0)];

        assert_eq!(
            reciprocal_rank_fusion(&fulltext, &semantic, 60.0)[0],
            (3, 0)
        );
    }

    #[test]
    fn reciprocal_rank_fusion_merges_different_subdocs_for_the_same_document() {
        let fulltext = [(7, 2), (1, 0), (2, 0)];
        let semantic = [(3, 0), (7, 5), (4, 0)];

        let results = reciprocal_rank_fusion(&fulltext, &semantic, 60.0);

        assert_eq!(results[0].0, 7);
        assert_eq!(results.iter().filter(|(rowid, _)| *rowid == 7).count(), 1);
    }

    #[test]
    fn hybrid_reciprocal_rank_fusion_weights_semantic_above_fulltext() {
        let fulltext = [(1, 0)];
        let semantic = [(2, 0)];

        assert_eq!(
            hybrid_reciprocal_rank_fusion(&fulltext, &semantic, 60.0)[0],
            (2, 0)
        );
    }
}

struct BucketRead {
    bucket_idx: usize,
    start: usize,
    len: usize,
}

struct BucketReadBatch {
    offset: usize,
    len: usize,
    buckets: Vec<BucketRead>,
}

#[derive(Clone, Copy, Default)]
struct BucketReadStats {
    batches: usize,
    buckets: usize,
    bytes: usize,
}

impl BucketReadStats {
    fn from_batches(batches: &[BucketReadBatch]) -> Self {
        Self {
            batches: batches.len(),
            buckets: batches.iter().map(|batch| batch.buckets.len()).sum(),
            bytes: batches.iter().map(|batch| batch.len).sum(),
        }
    }
}

struct BucketPayload<'a> {
    keys: &'a [u8],
    residuals: &'a [u8],
}

struct CoarseTreeLevel {
    centers: Tensor,
    children: Vec<Vec<usize>>,
}

struct CoarseTree {
    levels: Vec<CoarseTreeLevel>,
}

struct IndexKMeans {
    centers: Tensor,
    routing: IndexKMeansNode,
}

struct IndexKMeansNode {
    packed_centers: fast_ops::PackedRight,
    children: Vec<IndexKMeansNode>,
    leaf_offset: usize,
}

#[derive(Default)]
struct MatchIoCounters {
    queries: usize,
    candidate_buckets: usize,
    selected_buckets: usize,
    selected_slots: usize,
    read: BucketReadStats,
}

impl MatchIoCounters {
    fn add_read_stats(&mut self, read: BucketReadStats) {
        self.read.batches += read.batches;
        self.read.buckets += read.buckets;
        self.read.bytes += read.bytes;
    }
}

thread_local! {
    static BUCKET_IO_COUNTERS: std::cell::RefCell<MatchIoCounters> =
        std::cell::RefCell::new(MatchIoCounters::default());
}

fn record_bucket_io_counters(counters: MatchIoCounters) {
    BUCKET_IO_COUNTERS.with(|cell| {
        let mut total = cell.borrow_mut();
        total.queries += counters.queries;
        total.candidate_buckets += counters.candidate_buckets;
        total.selected_buckets += counters.selected_buckets;
        total.selected_slots += counters.selected_slots;
        total.read.batches += counters.read.batches;
        total.read.buckets += counters.read.buckets;
        total.read.bytes += counters.read.bytes;
    });
}

pub fn reset_bucket_io_counters() {
    BUCKET_IO_COUNTERS.with(|cell| {
        let _ = cell.replace(MatchIoCounters::default());
    });
}

pub fn log_bucket_io_counters() {
    BUCKET_IO_COUNTERS.with(|cell| {
        let counters = cell.replace(MatchIoCounters::default());
        if counters.queries == 0 {
            return;
        }
        let q = counters.queries as f64;
        info!(
            "bucket io counters: queries={} avg_candidates={:.1} avg_selected_buckets={:.1} avg_selected_slots={:.1} avg_read_batches={:.1} avg_read_buckets={:.1} avg_read_mb={:.2}",
            counters.queries,
            counters.candidate_buckets as f64 / q,
            counters.selected_buckets as f64 / q,
            counters.selected_slots as f64 / q,
            counters.read.batches as f64 / q,
            counters.read.buckets as f64 / q,
            counters.read.bytes as f64 / q / (1024.0 * 1024.0),
        );
    });
}

/// Per-generation centroid data loaded from sidecar files.
pub struct GenerationCentroids {
    dim: usize,
    residual_quant_bits: u8,
    residual_bytes: usize,
    residual_dequant_table: packops::ResidualDequantTable,
    bucket_indices: Vec<usize>,
    sizes: Vec<usize>,
    data_offsets: Vec<usize>,
    indices_lens: Vec<usize>,
    residual_lens: Vec<usize>,
    sidecar_len: usize,
    data_file: Arc<File>,
    centers_matrix: Tensor,
    coarse_tree: CoarseTree,
}

static GENERATIONS_CACHE: Lazy<RwLock<HashMap<Vec<PathBuf>, Arc<Vec<GenerationCentroids>>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn invalidate_generations_cache(paths: &[PathBuf]) {
    GENERATIONS_CACHE.write().unwrap().remove(paths);
}

fn clear_generations_cache() {
    GENERATIONS_CACHE.write().unwrap().clear();
}

fn build_coarse_tree_from_centers(
    leaf_centers: &Tensor,
    device: &Device,
) -> Result<CoarseTree> {
    let (leaf_count, _) = leaf_centers.dims2()?;
    if leaf_count <= COARSE_TREE_BRANCHING {
        return Ok(CoarseTree { levels: vec![] });
    }

    let root_count = (leaf_count as f64)
        .sqrt()
        .ceil()
        .max(2.0) as usize;
    let root_count = root_count.min(leaf_count - 1);
    let leaf_centers = leaf_centers.to_device(&Device::Cpu)?;
    debug!(
        "building coarse tree root: {} leaves -> {} root buckets",
        leaf_count, root_count
    );
    let root_centers = kmeans(&leaf_centers, root_count, 5)?;
    let packed = fast_ops::PackedRight::new(&root_centers)?;
    let assignments = matmul_argmax_batched(&leaf_centers, &packed, KMEANS_MATMUL_BATCH)?;
    let assignments = assignments.to_vec1::<u32>()?;

    let mut children = vec![vec![]; root_count];
    for (leaf_idx, &root_idx) in assignments.iter().enumerate() {
        children[root_idx as usize].push(leaf_idx);
    }

    Ok(CoarseTree {
        levels: vec![CoarseTreeLevel {
            centers: root_centers.to_device(device)?,
            children,
        }],
    })
}

fn serialize_coarse_tree(tree: &CoarseTree) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    write_u64_le(&mut bytes, tree.levels.len())?;

    for level in &tree.levels {
        let centers = level.centers.to_device(&Device::Cpu)?;
        let (center_count, dim) = centers.dims2()?;
        anyhow::ensure!(
            center_count == level.children.len(),
            "coarse tree level has {} centers but {} child lists",
            center_count,
            level.children.len()
        );
        write_u64_le(&mut bytes, center_count)?;
        write_u64_le(&mut bytes, dim)?;
        bytes.write_all(&centers.to_f32_bytes()?)?;
        for children in &level.children {
            write_u64_le(&mut bytes, children.len())?;
            for &child in children {
                write_u64_le(&mut bytes, child)?;
            }
        }
    }

    Ok(bytes)
}

fn deserialize_coarse_tree(bytes: &[u8], device: &Device) -> Result<CoarseTree> {
    anyhow::ensure!(
        bytes.len() >= std::mem::size_of::<u64>(),
        "coarse tree data is too small"
    );
    let mut cursor = 0;
    let level_count = read_tree_u64(bytes, &mut cursor)?;
    anyhow::ensure!(
        level_count <= bytes.len() / std::mem::size_of::<u64>(),
        "coarse tree level count is invalid"
    );
    let mut levels = Vec::with_capacity(level_count);

    for _ in 0..level_count {
        let center_count = read_tree_u64(bytes, &mut cursor)?;
        let dim = read_tree_u64(bytes, &mut cursor)?;
        anyhow::ensure!(
            center_count <= bytes.len() / std::mem::size_of::<u64>(),
            "coarse tree center count is invalid"
        );
        let center_bytes_len = center_count
            .checked_mul(dim)
            .and_then(|len| len.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("coarse tree center byte length overflow"))?;
        let center_end = cursor
            .checked_add(center_bytes_len)
            .ok_or_else(|| anyhow::anyhow!("coarse tree center cursor overflow"))?;
        anyhow::ensure!(center_end <= bytes.len(), "coarse tree center data is truncated");
        let centers = Tensor::from_f32_bytes(&bytes[cursor..center_end], dim, device)?;
        cursor = center_end;

        let mut children = Vec::with_capacity(center_count);
        for _ in 0..center_count {
            let child_count = read_tree_u64(bytes, &mut cursor)?;
            anyhow::ensure!(
                child_count <= bytes.len() / std::mem::size_of::<u64>(),
                "coarse tree child count is invalid"
            );
            let mut child_list = Vec::with_capacity(child_count);
            for _ in 0..child_count {
                child_list.push(read_tree_u64(bytes, &mut cursor)?);
            }
            children.push(child_list);
        }

        levels.push(CoarseTreeLevel { centers, children });
    }

    anyhow::ensure!(
        cursor == bytes.len(),
        "coarse tree sidecar data has trailing bytes"
    );
    Ok(CoarseTree { levels })
}

impl GenerationCentroids {
    fn routed_centroid_candidates(&self, query_embeddings: &Tensor) -> Result<Vec<usize>> {
        let tree = &self.coarse_tree;
        if tree.levels.is_empty() {
            return Ok((0..self.sizes.len()).collect());
        }

        let (query_rows, _) = query_embeddings.dims2()?;
        let mut candidate_set = vec![false; self.sizes.len()];
        for level in tree.levels.iter().rev() {
            let k_level = level.children.len();
            if k_level == 0 {
                continue;
            }
            let n_expand = (k_level as f64).sqrt().ceil() as usize;
            let level_sim = fast_ops::matmul_t(query_embeddings, &level.centers)?;
            let level_sim = level_sim.to_device(&Device::Cpu)?;
            let level_sorted = level_sim.arg_sort_last_dim(false)?
                .to_device(&Device::Cpu)?
                .to_vec2::<u32>()?;

            for qi in 0..query_rows {
                for &node_idx in level_sorted[qi].iter().take(n_expand) {
                    if let Some(leaves) = level.children.get(node_idx as usize) {
                        for &leaf_idx in leaves {
                            if let Some(selected) = candidate_set.get_mut(leaf_idx) {
                                *selected = true;
                            }
                        }
                    }
                }
            }
        }

        let candidates: Vec<usize> = candidate_set
            .iter()
            .enumerate()
            .filter_map(|(idx, selected)| selected.then_some(idx))
            .collect();
        if candidates.is_empty() {
            Ok((0..self.sizes.len()).collect())
        } else {
            Ok(candidates)
        }
    }

    fn bucket_range(&self, bucket_idx: usize) -> Result<(usize, usize, usize, usize)> {
        let bucket_id = *self
            .bucket_indices
            .get(bucket_idx)
            .ok_or_else(|| anyhow::anyhow!("bucket index {bucket_idx} is out of range"))?;
        let data_offset = self.data_offsets[bucket_idx];
        let indices_len = self.indices_lens[bucket_idx];
        let residual_len = self.residual_lens[bucket_idx];
        let total_len = indices_len
            .checked_add(residual_len)
            .ok_or_else(|| anyhow::anyhow!("bucket {bucket_id} data length overflow"))?;
        let end = data_offset
            .checked_add(total_len)
            .ok_or_else(|| anyhow::anyhow!("bucket {bucket_id} data range overflow"))?;
        anyhow::ensure!(
            end <= self.sidecar_len,
            "bucket {bucket_id} data range {}..{} exceeds sidecar length {}",
            data_offset,
            end,
            self.sidecar_len
        );
        Ok((bucket_id, data_offset, indices_len, residual_len))
    }

    fn bucket_read_batches(&self, bucket_indices: &[usize]) -> Result<Vec<BucketReadBatch>> {
        let mut reads = Vec::with_capacity(bucket_indices.len());
        for &bucket_idx in bucket_indices {
            let (_, data_offset, indices_len, residual_len) = self.bucket_range(bucket_idx)?;
            let len = indices_len
                .checked_add(residual_len)
                .ok_or_else(|| anyhow::anyhow!("bucket data length overflow"))?;
            if len != 0 {
                reads.push((data_offset, bucket_idx, len));
            }
        }
        reads.sort_by_key(|&(data_offset, bucket_idx, _)| (data_offset, bucket_idx));

        let mut batches: Vec<BucketReadBatch> = vec![];
        for (data_offset, bucket_idx, len) in reads {
            if let Some(batch) = batches.last_mut() {
                let batch_end = batch
                    .len
                    .checked_add(batch.offset)
                    .ok_or_else(|| anyhow::anyhow!("bucket read batch range overflow"))?;
                if data_offset <= batch_end + BUCKET_READ_COALESCE_GAP_BYTES {
                    let start = data_offset
                        .checked_sub(batch.offset)
                        .ok_or_else(|| anyhow::anyhow!("bucket read batch range underflow"))?;
                    batch.buckets.push(BucketRead {
                        bucket_idx,
                        start,
                        len,
                    });
                    batch.len = batch.len.max(
                        start
                            .checked_add(len)
                            .ok_or_else(|| anyhow::anyhow!("bucket read batch length overflow"))?,
                    );
                    continue;
                }
            }
            batches.push(BucketReadBatch {
                offset: data_offset,
                len,
                buckets: vec![BucketRead {
                    bucket_idx,
                    start: 0,
                    len,
                }],
            });
        }
        Ok(batches)
    }

    #[cfg(unix)]
    fn read_bucket_batch(file: &File, batch: &BucketReadBatch) -> Result<Vec<Vec<u8>>> {
        let mut data = vec![0; batch.len];
        let mut buffers = [data.as_mut_slice()];
        readv_exact_at(file, batch.offset as u64, &mut buffers)?;
        batch
            .buckets
            .iter()
            .map(|bucket| {
                let end = bucket
                    .start
                    .checked_add(bucket.len)
                    .ok_or_else(|| anyhow::anyhow!("bucket read batch slice overflow"))?;
                anyhow::ensure!(
                    end <= data.len(),
                    "bucket read batch slice {}..{} exceeds batch length {}",
                    bucket.start,
                    end,
                    data.len()
                );
                Ok(data[bucket.start..end].to_vec())
            })
            .collect()
    }

    #[cfg(not(unix))]
    fn read_bucket_batch(file: &File, batch: &BucketReadBatch) -> Result<Vec<Vec<u8>>> {
        let data = read_contiguous_exact_at(file, batch.offset as u64, batch.len)?;
        let mut payloads = Vec::with_capacity(batch.buckets.len());
        for bucket in &batch.buckets {
            let end = bucket
                .start
                .checked_add(bucket.len)
                .ok_or_else(|| anyhow::anyhow!("bucket read batch slice overflow"))?;
            anyhow::ensure!(
                end <= data.len(),
                "bucket read batch slice {}..{} exceeds batch length {}",
                bucket.start,
                end,
                data.len()
            );
            payloads.push(data[bucket.start..end].to_vec());
        }
        Ok(payloads)
    }

    fn prefetch_bucket_payloads(
        &self,
        bucket_indices: &[usize],
    ) -> Result<(Vec<Option<Vec<u8>>>, BucketReadStats)> {
        let batches = self.bucket_read_batches(bucket_indices)?;
        let stats = BucketReadStats::from_batches(&batches);
        let mut payloads = vec![None; self.bucket_indices.len()];
        if batches.is_empty() {
            return Ok((payloads, stats));
        }

        for batch in &batches {
            let data = Self::read_bucket_batch(&self.data_file, batch)?;
            anyhow::ensure!(
                data.len() == batch.buckets.len(),
                "bucket read batch returned {} buffers for {} buckets",
                data.len(),
                batch.buckets.len()
            );
            for (bucket, data) in batch.buckets.iter().zip(data) {
                payloads[bucket.bucket_idx] = Some(data);
            }
        }
        Ok((payloads, stats))
    }

    fn bucket_payload<'a>(
        &'a self,
        bucket_idx: usize,
        prefetched: &'a [Option<Vec<u8>>],
    ) -> Result<BucketPayload<'a>> {
        let (_, _, indices_len, residual_len) = self.bucket_range(bucket_idx)?;
        let data = prefetched
            .get(bucket_idx)
            .and_then(|data| data.as_deref())
            .ok_or_else(|| anyhow::anyhow!("bucket {bucket_idx} was not prefetched"))?;
        anyhow::ensure!(
            data.len() == indices_len + residual_len,
            "bucket {bucket_idx} prefetched length {} does not match expected {}",
            data.len(),
            indices_len + residual_len
        );
        Ok(BucketPayload {
            keys: &data[..indices_len],
            residuals: &data[indices_len..],
        })
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

/// Load generation centroid data from sidecar files (cached).
pub fn load_generations(paths: &[PathBuf], device: &Device) -> Result<Arc<Vec<GenerationCentroids>>> {
    let key = paths.to_vec();
    {
        let cache = GENERATIONS_CACHE.read().unwrap();
        if let Some(cached) = cache.get(&key) {
            return Ok(cached.clone());
        }
    }

    let mut all = Vec::with_capacity(paths.len());
    for path in paths {
        let file = File::open(path)?;
        let sidecar_len: usize = file.metadata()?.len().try_into()?;
        let header = bucket_data_header_from_file(&file, sidecar_len)?;
        let residual_radius_levels =
            bucket_data_residual_radius_levels_from_file(&file, &header)?;
        let residual_dequant_table = packops::make_residual_dequant_table_with_radius_levels(
            header.residual_quant_bits,
            residual_radius_levels.as_deref(),
        )?;
        let (metas, center_block) = bucket_data_bucket_meta_from_file(&file, &header)?;
        let coarse_tree = bucket_data_coarse_tree_from_file(&file, &header, &metas, device)?;
        let residual_bytes =
            residual_bytes_for_dim(header.embedding_dim, header.residual_quant_bits)?;

        let mut bucket_indices = vec![];
        let mut sizes = vec![];
        let mut data_offsets = vec![];
        let mut indices_lens = vec![];
        let mut residual_lens = vec![];

        for (bucket_idx, meta) in metas.into_iter().enumerate() {
            if meta.size == 0 {
                continue;
            }
            bucket_indices.push(bucket_idx);
            sizes.push(meta.size);
            data_offsets.push(meta.data_offset.try_into()?);
            indices_lens.push(meta.indices_len);
            residual_lens.push(meta.residual_len);
        }

        let centers_matrix = if !bucket_indices.is_empty() {
            let center_f32_bytes = center_block_to_f32_bytes(
                &center_block,
                &bucket_indices,
                &coarse_tree,
                header.centroid_count,
                header.embedding_dim,
                header.center_format,
            )?;
            let centers =
                Tensor::from_f32_bytes(&center_f32_bytes, header.embedding_dim, &Device::Cpu)?;
            if bucket_indices.len() == header.centroid_count {
                centers.to_device(device)?
            } else {
                let indices: Vec<u32> = bucket_indices
                    .iter()
                    .map(|&idx| u32::try_from(idx))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let index = Tensor::from_slice(indices.as_slice(), (indices.len(),), &Device::Cpu)?;
                centers.index_select(&index, 0)?.to_device(device)?
            }
        } else {
            Tensor::zeros(&[0, header.embedding_dim], DType::F32, device)?
        };
        all.push(GenerationCentroids {
            dim: header.embedding_dim,
            residual_quant_bits: header.residual_quant_bits,
            residual_bytes,
            residual_dequant_table,
            bucket_indices,
            sizes,
            data_offsets,
            indices_lens,
            residual_lens,
            sidecar_len,
            data_file: Arc::new(file),
            centers_matrix,
            coarse_tree,
        });
    }

    let result = Arc::new(all);
    GENERATIONS_CACHE.write().unwrap().insert(key, result.clone());
    Ok(result)
}

/// Pure index search: scores query embeddings against generation sidecar files.
/// No database access — loads generation metadata and reads bucket payloads from sidecar files.
/// `unindexed` contains (doc_rowid, embeddings) for documents not yet in any generation.
pub fn match_centroids_raw(
    generation_files: &[PathBuf],
    query_embeddings: &Tensor,
    unindexed: &[(Vec<DocPtr>, Tensor)],
    threshold: f32,
    top_k: usize,
) -> Result<Vec<(f32, u32, u32)>> {
    let device = query_embeddings.device();
    let generations = load_generations(generation_files, device)?;
    let total_start = std::time::Instant::now();

    let k = 32;
    let t_prime = 40000;
    let device = query_embeddings.device();
    let (m, query_dim) = query_embeddings.dims2()?;

    let mut all_residuals = vec![];
    let mut document_clusters: Vec<(usize, usize)> = vec![];
    let mut gen_centroid_scores_all: Vec<Vec<Vec<f32>>> = vec![];
    let mut gen_centroid_score_ranges_all: Vec<Vec<(f32, f32)>> = vec![];
    let mut all = vec![];
    let mut count = 0;
    let mut missing = vec![0.0f32; m];
    let mut io_counters = MatchIoCounters {
        queries: 1,
        ..MatchIoCounters::default()
    };

    for gen in generations.iter() {
        if gen.sizes.is_empty() {
            continue;
        }
        anyhow::ensure!(
            gen.dim == query_dim,
            "index embedding dimension {} does not match query dimension {}; re-embed and reindex with the selected encoder",
            gen.dim,
            query_dim
        );

        let gen_idx = gen_centroid_scores_all.len();
        let n_centroids = gen.sizes.len();

        let candidates = gen.routed_centroid_candidates(query_embeddings)?;
        io_counters.candidate_buckets += candidates.len();
        let candidate_indices: Vec<u32> = candidates.iter().map(|&idx| idx as u32).collect();
        let candidate_index_tensor =
            Tensor::from_slice(candidate_indices.as_slice(), (candidate_indices.len(),), device)?;
        let candidate_centers = gen.centers_matrix.index_select(&candidate_index_tensor, 0)?;
        let query_centroid_similarity =
            fast_ops::matmul_t(query_embeddings, &candidate_centers)?;
        let query_centroid_similarity = query_centroid_similarity.to_device(&Device::Cpu)?;

        let candidate_centroid_scores = query_centroid_similarity.to_vec2::<f32>()?;
        let mut gen_centroid_scores = vec![vec![0.0f32; n_centroids]; m];
        for (candidate_pos, &centroid_idx) in candidates.iter().enumerate() {
            for query_idx in 0..m {
                gen_centroid_scores[query_idx][centroid_idx] =
                    candidate_centroid_scores[query_idx][candidate_pos];
            }
        }
        gen_centroid_scores_all.push(gen_centroid_scores);

        let sorted_indices = query_centroid_similarity.arg_sort_last_dim(false)?;

        let mut topk_clusters = Vec::with_capacity(k);
        let mut gen_centroid_score_ranges = Vec::with_capacity(m);
        for i in 0..m {
            let row = sorted_indices.get(i)?;
            let row_scores_sorted =
                query_centroid_similarity.get(i)?.gather(&row, D::Minus1)?;
            let row_scores_sorted = row_scores_sorted.to_vec1::<f32>()?;
            let row = row.to_vec1::<u32>()?;
            let mut cumsum = 0;
            let selection_limit = candidates.len().min(k);
            if selection_limit == 0 {
                continue;
            }
            let mut tail_rank = 0;
            for j in 0..selection_limit {
                let idx = candidates[row[j] as usize];
                topk_clusters.push(idx);
                io_counters.selected_slots += 1;
                cumsum += gen.sizes[idx];
                tail_rank = j;
                if cumsum >= t_prime {
                    break;
                }
            }
            let confidence_tail_rank = (selection_limit / 2).max(1) - 1;
            let residual_tail_rank = tail_rank.min(confidence_tail_rank);
            gen_centroid_score_ranges.push((
                row_scores_sorted[0],
                row_scores_sorted[residual_tail_rank],
            ));
            if cumsum < t_prime {
                missing[i] = missing[i].max(row_scores_sorted[selection_limit - 1]);
            }
        }
        gen_centroid_score_ranges_all.push(gen_centroid_score_ranges);
        topk_clusters.sort_unstable();
        topk_clusters.dedup();
        io_counters.selected_buckets += topk_clusters.len();

        debug!(
            "hierarchical centroid routing: candidates={} selected={} / {} buckets",
            candidates.len(),
            topk_clusters.len(),
            n_centroids
        );

        let (prefetched_payloads, read_stats) = gen.prefetch_bucket_payloads(&topk_clusters)?;
        io_counters.add_read_stats(read_stats);
        for &i in &topk_clusters {
            let bucket_idx = i as usize;
            let bucket_id = gen.bucket_indices[bucket_idx];
            let payload = gen.bucket_payload(bucket_idx, &prefetched_payloads)?;
            let keys_compressed = payload.keys;
            let residual_bytes = payload.residuals;

            let document_indices = decompress_keys(keys_compressed)?;
            anyhow::ensure!(
                residual_bytes.len() % gen.residual_bytes == 0,
                "bucket {bucket_id} residual byte length {} is not divisible by residual width {}; run ./warp-cli reindex with matching feature flags",
                residual_bytes.len(),
                gen.residual_bytes
            );
            let residuals = packops::residuals_from_bytes(
                residual_bytes,
                gen.dim,
                &gen.residual_dequant_table,
                gen.residual_quant_bits,
                &Device::Cpu,
            )?;
            let (num_docs, _) = residuals.dims2()?;
            anyhow::ensure!(
                num_docs == document_indices.len(),
                "bucket {bucket_id} has {num_docs} residual vectors but {} document indices; run ./warp-cli reindex with matching feature flags",
                document_indices.len()
            );
            all_residuals.push(residuals);
            for idx in &document_indices[..num_docs] {
                document_clusters.push((gen_idx, i as usize));
                all.push((*idx, count));
                count += 1;
            }
        }
    }

    if count == 0 && unindexed.is_empty() {
        record_bucket_io_counters(io_counters);
        return Ok(vec![]);
    }

    let n = m;
    let mut sim: Vec<f32> = Vec::with_capacity(count * n);

    // Process indexed embeddings: query·residuals + centroid scores
    if !all_residuals.is_empty() {
        let all_residuals = Tensor::cat(&all_residuals, 0)?;
        let all_residuals = all_residuals.to_device(device)?;

        let residual_sims =
            fast_ops::matmul_t(query_embeddings, &all_residuals)?.transpose(0, 1)?;
        let residual_sims = residual_sims.to_device(&Device::Cpu)?;
        let residual_sims = residual_sims.to_dtype(DType::F32)?.contiguous()?;

        let mut residual_sims_flat = residual_sims.flatten_all()?.to_vec1::<f32>()?;
        for (doc_idx, &(gen_idx, cluster_idx)) in
            document_clusters.iter().enumerate()
        {
            let gen = &generations[gen_idx];
            let centroid_scores = &gen_centroid_scores_all[gen_idx];
            let centroid_score_ranges = &gen_centroid_score_ranges_all[gen_idx];
            for (query_idx, scores) in centroid_scores.iter().enumerate().take(n) {
                let offset = doc_idx * n + query_idx;
                let centroid_score = scores[cluster_idx];
                let residual_weight = packops::residual_centroid_confidence_weight(
                    centroid_score,
                    centroid_score_ranges[query_idx],
                    gen.residual_quant_bits,
                );
                residual_sims_flat[offset] =
                    centroid_score + residual_weight * residual_sims_flat[offset];
            }
        }
        sim.extend_from_slice(&residual_sims_flat);
    }

    // Process unindexed embeddings: full similarities (no centroid boost)
    if !unindexed.is_empty() {
        let mut unindexed_tensors = vec![];
        for (indices, embeddings) in unindexed {
            let (num_docs, dim) = embeddings.dims2()?;
            anyhow::ensure!(
                dim == query_dim,
                "unindexed embedding dimension {dim} does not match query dimension {query_dim}; re-embed with the selected encoder"
            );
            anyhow::ensure!(
                indices.len() == num_docs,
                "unindexed embedding has {num_docs} rows but {} document indices",
                indices.len()
            );
            unindexed_tensors.push(embeddings.clone());
            for idx in indices {
                all.push((*idx, count));
                count += 1;
            }
        }
        let all_unindexed = Tensor::cat(&unindexed_tensors, 0)?;
        let all_unindexed = all_unindexed.to_device(device)?;

        let unindexed_sims =
            fast_ops::matmul_t(query_embeddings, &all_unindexed)?.transpose(0, 1)?;
        let unindexed_sims = unindexed_sims.to_device(&Device::Cpu)?;
        let unindexed_sims = unindexed_sims.to_dtype(DType::F32)?.contiguous()?;
        let unindexed_sims_flat = unindexed_sims.flatten_all()?.to_vec1::<f32>()?;
        sim.extend_from_slice(&unindexed_sims_flat);
    }

    if count == 0 {
        record_bucket_io_counters(io_counters);
        return Ok(vec![]);
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
    let mut best_sub_score = f32::NEG_INFINITY;
    let mut best_sub_idx = 0u32;

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
                if sub_score > best_sub_score {
                    best_sub_score = sub_score;
                    best_sub_idx = prev_sub_idx;
                }
                vmax_inplace(&mut doc_scores, &sub_scores);
                sub_scores.copy_from_slice(&missing_similarities);
            }
            if idx_change {
                let doc_score = scaler * (doc_scores.iter().copied().sum::<f32>());
                if doc_score > cutoff {
                    scored_results.push((doc_score, prev_idx, best_sub_idx));
                }
                doc_scores.copy_from_slice(&missing_similarities);
                best_sub_score = f32::NEG_INFINITY;
                best_sub_idx = sub_idx;
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
    scored_results.truncate(top_k);

    debug!(
        "match_centroids_raw: {} embeddings in {} ms.",
        count,
        total_start.elapsed().as_millis()
    );
    record_bucket_io_counters(io_counters);
    Ok(scored_results)
}

fn split_tensor(tensor: &Tensor) -> Vec<Tensor> {
    let dims = tensor.dims();
    let num_rows = dims[0];

    // Collect each row as a separate one-row tensor.
    (0..num_rows)
        .map(|i| {
            let row_tensor = tensor.i(i).unwrap();
            row_tensor.unsqueeze(0).unwrap()
        })
        .collect()
}

fn docptrs_for_counts(id: u32, counts: &str) -> Vec<DocPtr> {
    let mut document_indices = Vec::new();
    for (i, count) in counts
        .split(',')
        .filter_map(|s| s.parse::<u32>().ok())
        .enumerate()
    {
        for _ in 0..count {
            document_indices.push((id, i as u32));
        }
    }
    document_indices
}

fn cached_embeddings_for_rowid(
    cache: &dyn EmbeddingCache,
    rowid: u64,
) -> Result<CachedEmbeddings> {
    load_cached_embeddings(cache, rowid, "")?
        .ok_or_else(|| anyhow::anyhow!("missing embeddings for rowid {rowid}"))
}

fn sample_embeddings_for_rowids(
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
) -> Result<(Tensor, usize)> {
    let mut total_embeddings = 0;
    #[cfg(any(test, feature = "deterministic"))]
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    #[cfg(not(any(test, feature = "deterministic")))]
    let mut rng = rand::rng();
    let mut all_embeddings = vec![];

    for record in active_rowid_records(records) {
        let embeddings = cached_embeddings_for_rowid(cache, record.rowid)?;
        anyhow::ensure!(
            embeddings.embedding_count == record.rows as usize,
            "rowid {} catalog says {} vectors but embedding blob has {}",
            record.rowid,
            record.rows,
            embeddings.embedding_count
        );
        let t = Tensor::embeddings_from_packed(
            &embeddings.embeddings,
            dim_from_model_id(&embeddings.model),
            &Device::Cpu,
        )?;
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
        return Ok((Tensor::zeros(&[0, DEFAULT_EMBEDDING_DIM], DType::F32, &Device::Cpu)?, 0));
    }
    let matrix = Tensor::stack(&all_embeddings, 0)?;
    Ok((matrix, total_embeddings))
}

fn token_counts_for_lens(lens: &str, offsets: &[(usize, usize)]) -> Result<Vec<u32>> {
    let mut lengths: Vec<usize> = lens
        .split(',')
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();
    anyhow::ensure!(!lengths.is_empty(), "document chunk lens are empty");

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
    anyhow::ensure!(count == 0, "unfinished token count while splitting chunks");
    anyhow::ensure!(
        counts.iter().sum::<u32>() == offsets.len() as u32,
        "token counts do not cover all offsets"
    );
    Ok(counts)
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

fn gaussian_stride2_document_embeddings(
    embeddings: &Tensor,
    counts: &[u32],
) -> Result<(Tensor, Vec<u32>)> {
    let (rows, cols) = embeddings.dims2()?;
    anyhow::ensure!(
        counts.iter().map(|c| *c as usize).sum::<usize>() == rows,
        "document chunk counts do not match embedding rows"
    );

    let rows_vec = embeddings.to_vec2::<f32>()?;
    let mut downsampled_rows = Vec::with_capacity((rows + 1) / 2);
    let mut downsampled_counts = Vec::with_capacity(counts.len());
    let mut offset = 0usize;

    for &count in counts {
        let count = count as usize;
        let mut kept = 0u32;
        for local_idx in (0..count).step_by(2) {
            let row_idx = offset + local_idx;
            let mut out = vec![0.0f32; cols];
            for (delta, weight) in [(-1isize, 0.25f32), (0, 0.5), (1, 0.25)] {
                let neighbor_local_idx = local_idx as isize + delta;
                if !(0..count as isize).contains(&neighbor_local_idx) {
                    continue;
                }
                let neighbor_idx = offset + neighbor_local_idx as usize;
                for (dst, src) in out.iter_mut().zip(rows_vec[neighbor_idx].iter()) {
                    *dst += weight * *src;
                }
            }

            let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv_norm = 1.0 / norm;
                for value in &mut out {
                    *value *= inv_norm;
                }
                downsampled_rows.push(out);
            } else {
                downsampled_rows.push(rows_vec[row_idx].clone());
            }
            kept += 1;
        }
        downsampled_counts.push(kept);
        offset += count;
    }

    let downsampled_count = downsampled_rows.len();
    let mut flat = Vec::with_capacity(downsampled_count * cols);
    for row in downsampled_rows {
        flat.extend(row);
    }
    let downsampled = Tensor::from_vec(flat, (downsampled_count, cols), embeddings.device())?;
    debug!(
        "document token gaussian stride-2 downsampled {downsampled_count}/{rows} tokens ({:.1}%)",
        100.0 * (downsampled_count as f64) / (rows as f64)
    );
    Ok((downsampled, downsampled_counts))
}

pub(crate) fn compute_cached_embeddings(
    embedder: &Embedder,
    body: &str,
    lens: &str,
) -> Result<CachedEmbeddings> {
    let now = std::time::Instant::now();
    let (embeddings, offsets) = embedder.embed(body)?;
    let mut embeddings = embeddings.squeeze(0)?.to_device(&Device::Cpu)?;
    let dt = now.elapsed().as_secs_f64();
    let (original_rows, _original_cols) = embeddings.dims2()?;
    debug!(
        "embedder took {} ms ({} rows/s).",
        now.elapsed().as_millis(),
        ((original_rows as f64) / dt).round()
    );

    let mut counts = token_counts_for_lens(lens, &offsets)?;
    if document_token_conv_stride2() {
        let downsampled = gaussian_stride2_document_embeddings(&embeddings, &counts)?;
        embeddings = downsampled.0;
        counts = downsampled.1;
    }
    let (rows, cols) = embeddings.dims2()?;
    debug!(
        "got embedding for chunk {:?} {:?}",
        embeddings.dims2()?,
        counts,
    );

    let now = std::time::Instant::now();
    let bytes = embeddings.embeddings_to_packed()?;
    let pct = 100.0 * (bytes.len() as f32) / ((rows * cols) as f32);
    let bpe = 8.0 * (bytes.len() as f32) / ((rows * cols) as f32);
    debug!(
        "compressing to {pct:.2}% {bpe:.2}bpe took {} ms.",
        now.elapsed().as_millis()
    );

    #[cfg(debug_assertions)]
    {
        let t = Tensor::embeddings_from_packed(&bytes, cols, &Device::Cpu)?;
        let min_acc = rowwise_cosine_min(&embeddings, &t)?;

        let n = bpe.ceil() as u32;
        let qn = stretch_rows(&embeddings)?.quantize(n)?.dequantize(n)?;
        let min_qn_acc = rowwise_cosine_min(&embeddings, &qn)?;
        debug!("haar reconstruction accuracy={min_acc} compare at q{n}_acc={min_qn_acc}");
    }

    Ok(CachedEmbeddings {
        model: model_id_for_dim(cols),
        counts: counts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(","),
        embedding_count: rows,
        embeddings: bytes,
    })
}

#[cfg(any(feature = "sqlite", feature = "capi-embed-cache"))]
pub(crate) fn load_or_compute_cached_embeddings(
    cache: &dyn EmbeddingCache,
    rowid: u64,
    hash: &str,
    body: &str,
    lens: &str,
    embedder: &Embedder,
) -> Result<(CachedEmbeddings, bool)> {
    if let Some(embeddings) = load_cached_embeddings(cache, rowid, hash)? {
        return Ok((embeddings, false));
    }

    let embeddings = compute_cached_embeddings(embedder, body, lens)?;
    if load_cached_embeddings(cache, rowid, hash)?.is_none() {
        cache.put(hash, &embeddings)?;
    }
    Ok((embeddings, true))
}

fn sample_centers(data: &Tensor, k: usize) -> Result<Tensor> {
    let (m, _) = data.dims2()?;
    anyhow::ensure!(k > 0 && k <= m, "cannot sample {k} centers from {m} rows");
    let indices: Vec<u32> = (0..k).map(|i| ((i * m) / k).min(m - 1) as u32).collect();
    let index_tensor = Tensor::from_slice(indices.as_slice(), (k,), data.device())?;
    Ok(data.index_select(&index_tensor, 0)?)
}

fn allocate_child_kmeans_targets(counts: &[usize], target_k: usize) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total == 0 || target_k == 0 {
        return vec![0; counts.len()];
    }

    let target_k = target_k.min(total);
    let mut child_ks = vec![0usize; counts.len()];
    let mut remainders = Vec::new();
    let mut assigned = 0usize;

    for (idx, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let exact = (count as f64 / total as f64) * target_k as f64;
        let base = (exact.floor() as usize).max(1).min(count);
        child_ks[idx] = base;
        assigned += base;
        remainders.push((exact - exact.floor(), idx));
    }

    if assigned > target_k {
        remainders.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut excess = assigned - target_k;
        while excess > 0 {
            let mut changed = false;
            for &(_, idx) in &remainders {
                if child_ks[idx] > 1 {
                    child_ks[idx] -= 1;
                    excess -= 1;
                    changed = true;
                    if excess == 0 {
                        break;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    } else if assigned < target_k {
        remainders.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut remaining = target_k - assigned;
        while remaining > 0 {
            let mut changed = false;
            for &(_, idx) in &remainders {
                if child_ks[idx] < counts[idx] {
                    child_ks[idx] += 1;
                    remaining -= 1;
                    changed = true;
                    if remaining == 0 {
                        break;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    child_ks
}

fn select_tensor_rows(data: &Tensor, rows: &[u32]) -> Result<Tensor> {
    let index_tensor = Tensor::from_slice(rows, (rows.len(),), data.device())?;
    Ok(data.index_select(&index_tensor, 0)?)
}

fn hierarchical_kmeans_for_index(
    data: &Tensor,
    target_k: usize,
    max_iter: usize,
    bar: Option<&progress::Bar>,
    next_leaf_offset: &mut usize,
) -> Result<IndexKMeans> {
    let (m, _) = data.dims2()?;
    let target_k = target_k.min(m);
    anyhow::ensure!(target_k > 0, "cannot build index kmeans with zero centers");

    if target_k <= INDEX_KMEANS_BRANCHING || m <= INDEX_KMEANS_BRANCHING {
        let centers = kmeans_inner(data, target_k, max_iter, None)?;
        let packed_centers = fast_ops::PackedRight::new(&centers)?;
        let leaf_offset = *next_leaf_offset;
        *next_leaf_offset += target_k;
        if let Some(bar) = bar {
            bar.inc(target_k as u64);
        }
        return Ok(IndexKMeans {
            centers: centers.clone(),
            routing: IndexKMeansNode {
                packed_centers,
                children: vec![],
                leaf_offset,
            },
        });
    }

    let branch_k = INDEX_KMEANS_BRANCHING.min(target_k).min(m);
    let coarse = kmeans_inner(data, branch_k, max_iter, None)?;
    let packed = fast_ops::PackedRight::new(&coarse)?;
    let assignments = matmul_argmax_batched(data, &packed, KMEANS_MATMUL_BATCH)?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;

    let mut row_groups = vec![Vec::<u32>::new(); branch_k];
    for (row, &child) in assignments.iter().enumerate() {
        row_groups[child as usize].push(row as u32);
    }
    let counts: Vec<usize> = row_groups.iter().map(Vec::len).collect();
    let active_children = counts.iter().filter(|&&count| count > 0).count();
    if active_children <= 1 {
        let centers = sample_centers(data, target_k)?;
        let packed_centers = fast_ops::PackedRight::new(&centers)?;
        let leaf_offset = *next_leaf_offset;
        *next_leaf_offset += target_k;
        if let Some(bar) = bar {
            bar.inc(target_k as u64);
        }
        return Ok(IndexKMeans {
            centers: centers.clone(),
            routing: IndexKMeansNode {
                packed_centers,
                children: vec![],
                leaf_offset,
            },
        });
    }

    let child_targets = allocate_child_kmeans_targets(&counts, target_k);
    let mut routing_centers = Vec::new();
    let mut child_nodes = Vec::new();
    let mut child_centers = Vec::new();

    for child_idx in 0..branch_k {
        let child_target = child_targets[child_idx];
        if child_target == 0 {
            continue;
        }

        let child_data = select_tensor_rows(data, &row_groups[child_idx])?;
        let child = hierarchical_kmeans_for_index(
            &child_data,
            child_target,
            max_iter,
            bar,
            next_leaf_offset,
        )?;
        routing_centers.push(coarse.get(child_idx)?);
        child_centers.push(child.centers);
        child_nodes.push(child.routing);
    }

    let routing = Tensor::stack(&routing_centers, 0)?.to_device(data.device())?;
    let packed_centers = fast_ops::PackedRight::new(&routing)?;
    Ok(IndexKMeans {
        centers: Tensor::cat(&child_centers, 0)?,
        routing: IndexKMeansNode {
            packed_centers,
            children: child_nodes,
            leaf_offset: 0,
        },
    })
}

fn run_kmeans_for_index(matrix: &Tensor, total_embeddings: usize) -> Result<IndexKMeans> {
    let now = std::time::Instant::now();
    let sqrt_k = (16.0 * (total_embeddings as f64).sqrt()).round() as usize;
    let size_k = total_embeddings.div_ceil(INDEX_TARGET_BUCKET_VECTORS);
    let mut k = sqrt_k.max(size_k);
    k = k.max(1);
    debug!(
        "total_embeddings={} k={} sqrt_k={} size_k={} target_bucket_vectors={}",
        total_embeddings, k, sqrt_k, size_k, INDEX_TARGET_BUCKET_VECTORS
    );
    let (m, _) = matrix.dims2()?;
    if m < k {
        k = (m / 4).max(1);
    }
    let bar = progress::new_with_label(k as u64, "kmeans");
    let mut next_leaf_offset = 0;
    let centers = hierarchical_kmeans_for_index(
        matrix,
        k,
        INDEX_KMEANS_ITERATIONS,
        Some(&bar),
        &mut next_leaf_offset,
    )?;
    bar.finish();
    anyhow::ensure!(
        centers.centers.dims2()?.0 == next_leaf_offset,
        "index kmeans produced inconsistent leaf offsets"
    );
    anyhow::ensure!(
        next_leaf_offset > 0,
        "index kmeans produced no centers"
    );
    debug!("kmeans took {} ms.", now.elapsed().as_millis());
    Ok(centers)
}

fn assign_with_index_kmeans_node(
    data: &Tensor,
    node: &IndexKMeansNode,
    row_indices: &[usize],
    assignments: &mut [u32],
    best_scores: &mut [f32],
) -> Result<()> {
    let (m, _) = data.dims2()?;
    if m == 0 {
        return Ok(());
    }

    if node.children.is_empty() {
        let local_assignments =
            matmul_argmax_scores_batched(data, &node.packed_centers, KMEANS_MATMUL_BATCH)?;
        for (row, &bucket) in local_assignments.iter().enumerate() {
            let global_row = row_indices[row];
            if bucket.1 > best_scores[global_row] {
                best_scores[global_row] = bucket.1;
                assignments[global_row] = (node.leaf_offset + bucket.0 as usize).try_into()?;
            }
        }
        return Ok(());
    }

    debug_assert_eq!(INDEX_ASSIGNMENT_BEAM, 2);
    let local_assignments = matmul_top2_batched(data, &node.packed_centers, KMEANS_MATMUL_BATCH)?;
    let mut child_rows = vec![Vec::<u32>::new(); node.children.len()];
    let mut child_row_indices = vec![Vec::<usize>::new(); node.children.len()];
    for (row, &(best_child, second_child)) in local_assignments.iter().enumerate() {
        for child in [best_child, second_child] {
            let child = child as usize;
            anyhow::ensure!(
                child < node.children.len(),
                "index kmeans routed row to missing child {child}"
            );
            child_rows[child].push(row as u32);
            child_row_indices[child].push(row_indices[row]);
        }
    }

    for (child_idx, rows) in child_rows.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let child_data = select_tensor_rows(data, rows)?;
        assign_with_index_kmeans_node(
            &child_data,
            &node.children[child_idx],
            &child_row_indices[child_idx],
            assignments,
            best_scores,
        )?;
    }

    Ok(())
}

fn assign_with_index_kmeans(data: &Tensor, routing: &IndexKMeansNode) -> Result<Vec<u32>> {
    let (m, _) = data.dims2()?;
    let mut assignments = vec![0u32; m];
    let mut best_scores = vec![f32::NEG_INFINITY; m];
    let row_indices: Vec<usize> = (0..m).collect();
    assign_with_index_kmeans_node(
        data,
        routing,
        &row_indices,
        &mut assignments,
        &mut best_scores,
    )?;
    Ok(assignments)
}

#[cfg(feature = "polar-quant")]
fn learn_residual_radius_levels(
    embeddings: &[Tensor],
    index_kmeans: &IndexKMeans,
    residual_centers_cpu: &Tensor,
    residual_quant_bits: u8,
) -> Result<Option<Vec<f32>>> {
    packops::validate_residual_quant_bits(residual_quant_bits)?;
    if residual_quant_bits == 0 || embeddings.is_empty() {
        return Ok(None);
    }

    let data_cpu = Tensor::cat(embeddings, 0)?;
    let cluster_assignments = assign_with_index_kmeans(&data_cpu, &index_kmeans.routing)?;
    let (rows, dim) = data_cpu.dims2()?;
    let (center_count, center_dim) = residual_centers_cpu.dims2()?;
    anyhow::ensure!(
        dim == center_dim,
        "residual training data dimension {dim} does not match center dimension {center_dim}"
    );
    anyhow::ensure!(
        dim % 2 == 0,
        "polar residual radius training requires an even embedding dimension"
    );
    let data = data_cpu.flatten_all()?.to_vec1::<f32>()?;
    let centers = residual_centers_cpu.flatten_all()?.to_vec1::<f32>()?;
    let mut radii = Vec::with_capacity(rows * (dim / 2));

    for (row, &bucket) in cluster_assignments.iter().enumerate() {
        let bucket: usize = bucket.try_into()?;
        anyhow::ensure!(
            bucket < center_count,
            "assigned bucket {bucket} is outside centroid count {center_count}"
        );
        let row_offset = row * dim;
        let center_offset = bucket * dim;
        for dim_idx in (0..dim).step_by(2) {
            let x = data[row_offset + dim_idx] - centers[center_offset + dim_idx];
            let y = data[row_offset + dim_idx + 1] - centers[center_offset + dim_idx + 1];
            radii.push(x.mul_add(x, y * y).sqrt());
        }
    }

    // We tried learning angle levels too, with residual radius^2 as the weight.
    // It converged to nearly uniform bins and regressed nfcorpus at 2 and 3 bits,
    // so keep angle quantization analytic and only learn the residual radii.
    let levels = packops::lloyd_max_radius_levels(&radii, residual_quant_bits)?;
    info!(
        "Lloyd-Max polar radius levels bits={} samples={} levels={:?}",
        residual_quant_bits,
        radii.len(),
        levels
    );
    Ok(Some(levels))
}

#[cfg(not(feature = "polar-quant"))]
fn learn_residual_radius_levels(
    _embeddings: &[Tensor],
    _index_kmeans: &IndexKMeans,
    _residual_centers_cpu: &Tensor,
    _residual_quant_bits: u8,
) -> Result<Option<Vec<f32>>> {
    Ok(None)
}

fn write_bucket_batch(
    embeddings: Vec<Tensor>,
    indices: Vec<DocPtr>,
    index_kmeans: &IndexKMeans,
    residual_centers_cpu: &Tensor,
    residual_bytes: usize,
    residual_quant_bits: u8,
    residual_radius_levels: Option<&[f32]>,
) -> Result<(tempfile::NamedTempFile, u128, u128)> {
    let now = std::time::Instant::now();
    let data_cpu = Tensor::cat(&embeddings, 0)?;
    let cluster_assignments = assign_with_index_kmeans(&data_cpu, &index_kmeans.routing)?;
    let mmuls_ms = now.elapsed().as_millis();

    let now = std::time::Instant::now();
    let mut writer = merger::Writer::new(residual_bytes)?;

    let mut pairs: Vec<(usize, u32)> = cluster_assignments
        .iter()
        .enumerate()
        .map(|(i, &bucket)| (i, bucket))
        .collect();
    pairs.sort_by_key(|&(_, bucket)| bucket);

    let mut keys: Vec<(u32, u32)> = Vec::with_capacity(indices.len());
    let mut residuals_bytes: Vec<u8> = Vec::with_capacity(indices.len() * residual_bytes);
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

        let center = residual_centers_cpu.get(bucket as usize)?;
        let residual = (&data_cpu.get(sample)? - &center)?;
        let residual_quantized = packops::residual_to_temp_bytes_with_radius_levels(
            &residual,
            residual_quant_bits,
            residual_radius_levels,
        )?;
        residuals_bytes.extend(&residual_quantized);
    }

    let tmpfile = writer.finish()?;
    Ok((tmpfile, mmuls_ms, now.elapsed().as_millis()))
}

fn mark_assigned_bucket_batch(
    embeddings: Vec<Tensor>,
    assigned: &mut [bool],
    index_kmeans: &IndexKMeans,
) -> Result<()> {
    if embeddings.is_empty() {
        return Ok(());
    }

    let centroid_count = assigned.len();
    let data_cpu = Tensor::cat(&embeddings, 0)?;
    let assignments = assign_with_index_kmeans(&data_cpu, &index_kmeans.routing)?;
    for bucket in assignments {
        let bucket: usize = bucket.try_into()?;
        anyhow::ensure!(
            bucket < centroid_count,
            "assigned bucket {bucket} is outside centroid count {centroid_count}"
        );
        assigned[bucket] = true;
    }
    Ok(())
}

fn assigned_bucket_indices_for_rowids(
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    index_kmeans: &IndexKMeans,
    expected_count: u64,
) -> Result<Vec<usize>> {
    let (centroid_count, center_dim) = index_kmeans.centers.dims2()?;
    let mut assigned = vec![false; centroid_count];
    let bar = progress::new_with_label(expected_count, "routing");
    let mut all_embeddings = vec![];
    let mut batch = 0usize;

    for record in active_rowid_records(records) {
        let embeddings = cached_embeddings_for_rowid(cache, record.rowid)?;
        anyhow::ensure!(
            embeddings.embedding_count == record.rows as usize,
            "rowid {} catalog says {} vectors but embedding blob has {}",
            record.rowid,
            record.rows,
            embeddings.embedding_count
        );

        let dim = dim_from_model_id(&embeddings.model);
        anyhow::ensure!(
            dim == center_dim,
            "document embedding dimension {dim} does not match index center dimension {center_dim}"
        );
        let t = Tensor::embeddings_from_packed(&embeddings.embeddings, dim, &Device::Cpu)?;
        let (m, _) = t.dims2()?;

        if batch > 0 && batch + m > INDEX_BATCH_SIZE {
            mark_assigned_bucket_batch(
                std::mem::take(&mut all_embeddings),
                &mut assigned,
                index_kmeans,
            )?;
            bar.inc(batch as u64);
            batch = 0;
        }

        all_embeddings.push(t);
        batch += m;

        if batch >= INDEX_BATCH_SIZE {
            mark_assigned_bucket_batch(
                std::mem::take(&mut all_embeddings),
                &mut assigned,
                index_kmeans,
            )?;
            bar.inc(batch as u64);
            batch = 0;
        }
    }

    if batch > 0 {
        mark_assigned_bucket_batch(
            std::mem::take(&mut all_embeddings),
            &mut assigned,
            index_kmeans,
        )?;
        bar.inc(batch as u64);
    }
    bar.finish();

    Ok(assigned
        .iter()
        .enumerate()
        .filter_map(|(bucket_idx, &seen)| seen.then_some(bucket_idx))
        .collect())
}

fn level_capacity(level: u32) -> usize {
    L0_CAPACITY * LSM_FANOUT.pow(level + 1)
}

fn write_buckets_for_rowids(
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    index_kmeans: &IndexKMeans,
    expected_count: u64,
    residual_quant_bits: u8,
) -> Result<(Vec<tempfile::NamedTempFile>, CenterSidecarData, Option<Vec<f32>>)> {
    let _priority_mgr = PriorityManager::new();
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let mut document_indices = Vec::<(u32, u32)>::new();
    let mut all_embeddings = vec![];

    let mut done = false;
    let mut batch = 0;
    let mut tmpfiles = vec![];
    let centers = &index_kmeans.centers;
    let centers_cpu = centers.to_device(&Device::Cpu)?;
    let (_, center_dim) = centers_cpu.dims2()?;
    let residual_bytes = packops::temp_residual_bytes_for_dim(center_dim, residual_quant_bits)?;
    let bucket_indices =
        assigned_bucket_indices_for_rowids(records, cache, index_kmeans, expected_count)?;
    let center_sidecar = prepare_center_sidecar_data(&centers_cpu, bucket_indices)?;
    let residual_centers_cpu = &center_sidecar.residual_centers_cpu;
    let bar = progress::new_with_label(expected_count, "indexing");
    let mut records = active_rowid_records(records);
    let mut residual_radius_levels = None;
    let mut attempted_radius_training = false;
    while !done {
        match records.next() {
            Some(record) => {
                let id: u32 = record.rowid.try_into().map_err(|_| {
                    anyhow::anyhow!(
                        "rowid {} exceeds current bucket key limit {}",
                        record.rowid,
                        u32::MAX
                    )
                })?;
                let embeddings = cached_embeddings_for_rowid(cache, record.rowid)?;
                anyhow::ensure!(
                    embeddings.embedding_count == record.rows as usize,
                    "rowid {} catalog says {} vectors but embedding blob has {}",
                    record.rowid,
                    record.rows,
                    embeddings.embedding_count
                );

                let dim = dim_from_model_id(&embeddings.model);
                anyhow::ensure!(
                    dim == center_dim,
                    "document embedding dimension {dim} does not match index center dimension {center_dim}"
                );
                let t = Tensor::embeddings_from_packed(&embeddings.embeddings, dim, &Device::Cpu)?;
                let (m, _) = t.dims2()?;

                let docptrs = docptrs_for_counts(id, &embeddings.counts);
                anyhow::ensure!(
                    docptrs.len() == m,
                    "document {id} has {m} embedding rows but {} document indices",
                    docptrs.len()
                );
                if batch > 0 && batch + m > INDEX_BATCH_SIZE {
                    if !attempted_radius_training {
                        residual_radius_levels = learn_residual_radius_levels(
                            &all_embeddings,
                            index_kmeans,
                            residual_centers_cpu,
                            residual_quant_bits,
                        )?;
                        attempted_radius_training = true;
                    }
                    let (tmpfile, mmuls_ms, writes_ms) = write_bucket_batch(
                        std::mem::take(&mut all_embeddings),
                        std::mem::take(&mut document_indices),
                        index_kmeans,
                        residual_centers_cpu,
                        residual_bytes,
                        residual_quant_bits,
                        residual_radius_levels.as_deref(),
                    )?;
                    tmpfiles.push(tmpfile);
                    mmuls_total += mmuls_ms;
                    writes_total += writes_ms;
                    bar.inc(batch as u64);
                    batch = 0;
                }

                document_indices.extend(docptrs);
                all_embeddings.push(t);
                batch += m;
            }
            None => {
                done = true;
            }
        }

        if batch >= INDEX_BATCH_SIZE || done {
            if batch == 0 {
                continue;
            }
            let flushed = batch;
            if !attempted_radius_training {
                residual_radius_levels = learn_residual_radius_levels(
                    &all_embeddings,
                    index_kmeans,
                    residual_centers_cpu,
                    residual_quant_bits,
                )?;
                attempted_radius_training = true;
            }
            let (tmpfile, mmuls_ms, writes_ms) = write_bucket_batch(
                std::mem::take(&mut all_embeddings),
                std::mem::take(&mut document_indices),
                index_kmeans,
                residual_centers_cpu,
                residual_bytes,
                residual_quant_bits,
                residual_radius_levels.as_deref(),
            )?;
            tmpfiles.push(tmpfile);
            mmuls_total += mmuls_ms;
            writes_total += writes_ms;
            bar.inc(flushed as u64);
            batch = 0;
        }
    }
    bar.finish();

    debug!("mmuls took {} ms.", mmuls_total);
    debug!("writes took {} ms.", writes_total);

    Ok((tmpfiles, center_sidecar, residual_radius_levels))
}

fn build_index_generation(
    index: &FileBackedIndex,
    cache: &dyn EmbeddingCache,
    level: u32,
    records: &[RowidRecord],
    options: IndexOptions,
) -> Result<Option<FileIndexGeneration>> {
    let active_embeddings = rowid_records_embedding_count(records);
    if active_embeddings == 0 {
        return Ok(None);
    }

    let (matrix, total_embeddings) = sample_embeddings_for_rowids(records, cache)?;
    if total_embeddings == 0 {
        return Ok(None);
    }

    info!(
        "building standalone L{} with {} embeddings",
        level, total_embeddings
    );
    let centers = run_kmeans_for_index(&matrix, total_embeddings)?;
    drop(matrix);

    let data_file = index.generation_file_name(level);
    let data_path = index.path_for(&data_file);
    let residual_quant_bits = options.residual_quant_bits();

    let (tmpfiles, center_sidecar, residual_radius_levels) = write_buckets_for_rowids(
        records,
        cache,
        &centers,
        total_embeddings as u64,
        residual_quant_bits,
    )?;
    if let Err(err) = merge_and_write_buckets_to_path(
        tmpfiles,
        &center_sidecar,
        records,
        &data_path,
        residual_quant_bits,
        residual_radius_levels.as_deref(),
    ) {
        let _ = std::fs::remove_file(&data_path);
        return Err(err);
    }

    Ok(Some(FileIndexGeneration {
        level,
        num_embeddings: total_embeddings,
        data_file,
    }))
}

pub(crate) fn index_buffered_embeddings(
    index: &FileBackedIndex,
    cache: &dyn EmbeddingCache,
) -> Result<()> {
    index_buffered_embeddings_with_options(index, cache, IndexOptions::default())
}

pub(crate) fn index_buffered_embeddings_with_options(
    index: &FileBackedIndex,
    cache: &dyn EmbeddingCache,
    options: IndexOptions,
) -> Result<()> {
    let buffer = sort_dedup_rowid_records(read_rowid_records(index.rowid_buffer_path())?);
    let x = rowid_records_embedding_count(&buffer);
    if x == 0 {
        return Ok(());
    }

    let generations = index.read_manifest()?;
    let indexed: usize = generations
        .iter()
        .map(|generation| generation.num_embeddings)
        .sum();
    info!("standalone index has {} buffered embeddings ({} indexed)", x, indexed);

    if x < L0_CAPACITY {
        debug!("buffering {} embeddings (< {} threshold)", x, L0_CAPACITY);
        return Ok(());
    }

    let mut total = x;
    let mut target_level = 0u32;
    loop {
        let cap = level_capacity(target_level);
        let level_size: usize = generations
            .iter()
            .filter(|generation| generation.level == target_level)
            .map(|generation| generation.num_embeddings)
            .sum();
        total += level_size;
        if total <= cap {
            break;
        }
        target_level += 1;
    }

    let mut inputs = vec![buffer];
    let mut stale_generations = vec![];
    let mut kept_generations = vec![];
    for generation in generations {
        if generation.level <= target_level {
            inputs.push(index.generation_rowid_records(&generation)?);
            stale_generations.push(generation);
        } else {
            kept_generations.push(generation);
        }
    }

    let mut merged = nway_merge_rowid_records(&inputs);
    if kept_generations.is_empty() {
        merged.retain(|record| record.rows > 0);
    }
    if let Some(generation) =
        build_index_generation(index, cache, target_level, &merged, options)?
    {
        kept_generations.push(generation);
    }
    kept_generations.sort_by_key(|generation| generation.level);
    index.write_manifest(&kept_generations)?;
    let _ = std::fs::remove_file(index.rowid_buffer_path());
    index.remove_generation_files(&stale_generations);
    clear_generations_cache();
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

#[cfg(all(test, feature = "sqlite"))]
mod tests;
