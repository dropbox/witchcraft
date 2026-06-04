use log::{debug, info, warn};
use memmap2::Mmap;
use once_cell::sync::Lazy;
#[cfg(any(test, feature = "deterministic"))]
use rand::SeedableRng;
#[cfg(feature = "sqlite")]
use rusqlite::OptionalExtension;
#[cfg(any(feature = "sqlite", feature = "capi-embed-cache"))]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
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
mod modernbert;
#[cfg(feature = "modernbert")]
use modernbert as t5_encoder;

#[cfg(feature = "modernbert-quantized")]
mod quantized_modernbert;
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

#[cfg(all(feature = "polar-quant-2bit", feature = "polar-quant-3bit"))]
compile_error!("polar-quant-2bit and polar-quant-3bit are mutually exclusive");

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
#[cfg(feature = "sqlite")]
use progress_reporter::ProgressReporter;

pub mod types;
pub use types::SqlStatementInternal;

#[cfg(feature = "sqlite")]
pub mod sql_generator;
#[cfg(feature = "sqlite")]
use sql_generator::build_filter_sql_and_params;

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
const BUCKET_DATA_VERSION: u32 = file_index::GENERATION_DATA_VERSION;
const BUCKET_DATA_MAGIC: [u8; 8] = file_index::GENERATION_DATA_MAGIC;
const LEGACY_BUCKET_DATA_VERSION: u32 = 2;
const LEGACY_BUCKET_DATA_MAGIC: [u8; 8] = *b"WRPBKT02";
const BUCKET_META_PREFIX_BYTES: usize = 20;
const LEGACY_BUCKET_DATA_HEADER_BYTES: usize =
    std::mem::size_of::<u32>() + BUCKET_DATA_MAGIC.len() + 3 * std::mem::size_of::<u64>();
const BUCKET_DATA_HEADER_BYTES: usize = file_index::GENERATION_DATA_HEADER_BYTES;
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

fn center_bytes_for_dim(dim: usize) -> usize {
    dim * std::mem::size_of::<f32>()
}

fn bucket_meta_bytes_for_dim(dim: usize) -> usize {
    BUCKET_META_PREFIX_BYTES + center_bytes_for_dim(dim)
}

fn residual_bytes_for_dim(dim: usize) -> usize {
    #[cfg(feature = "polar-quant")]
    {
        packops::polar_row_bytes(dim)
    }
    #[cfg(not(feature = "polar-quant"))]
    {
        assert!(
            dim % 2 == 0,
            "embedding dimension must be even for q4 residuals"
        );
        dim / 2
    }
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
    if dim == DEFAULT_EMBEDDING_DIM {
        prefix.to_string()
    } else {
        format!("{prefix}-d{dim}")
    }
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

fn kmeans(data: &Tensor, k: usize, max_iter: usize) -> Result<Tensor> {
    let (m, n) = data.dims2()?;
    debug!("kmeans k={} m={} n={}...", k, m, n);

    let _priority_mgr = PriorityManager::new();
    let total: u64 = (max_iter * k).try_into()?;
    let bar = progress::new_with_label(total, "kmeans");
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

struct BucketSidecarMeta {
    size: usize,
    data_offset: u64,
    indices_len: usize,
    residual_len: usize,
    center: Vec<u8>,
}

struct BucketDataHeader {
    centroid_count: usize,
    embedding_dim: usize,
    meta_bytes: usize,
    meta_offset: usize,
    payload_offset: usize,
    rowids_offset: usize,
}

fn write_bucket_data_header(
    writer: &mut impl Write,
    centroid_count: usize,
    meta_offset: usize,
    payload_offset: usize,
    rowids_offset: usize,
) -> Result<()> {
    writer.write_all(&BUCKET_DATA_VERSION.to_le_bytes())?;
    writer.write_all(&BUCKET_DATA_MAGIC)?;
    writer.write_all(&(centroid_count as u64).to_le_bytes())?;
    writer.write_all(&(meta_offset as u64).to_le_bytes())?;
    writer.write_all(&(payload_offset as u64).to_le_bytes())?;
    writer.write_all(&(rowids_offset as u64).to_le_bytes())?;
    Ok(())
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes[offset..offset + 4].try_into()?))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(bytes[offset..offset + 8].try_into()?))
}

fn bucket_data_header(bucket_data: &[u8]) -> Result<BucketDataHeader> {
    anyhow::ensure!(
        bucket_data.len() >= LEGACY_BUCKET_DATA_HEADER_BYTES,
        "bucket sidecar is too small: {} bytes",
        bucket_data.len()
    );
    let version = read_u32_le(bucket_data, 0)?;
    let header_bytes = match version {
        BUCKET_DATA_VERSION => BUCKET_DATA_HEADER_BYTES,
        LEGACY_BUCKET_DATA_VERSION => LEGACY_BUCKET_DATA_HEADER_BYTES,
        _ => anyhow::bail!(
            "bucket sidecar version {version} is not supported; run ./warp-cli reindex"
        ),
    };
    anyhow::ensure!(
        bucket_data.len() >= header_bytes,
        "bucket sidecar is too small: {} bytes",
        bucket_data.len()
    );
    anyhow::ensure!(
        if version == BUCKET_DATA_VERSION {
            &bucket_data[4..4 + BUCKET_DATA_MAGIC.len()] == BUCKET_DATA_MAGIC.as_slice()
        } else {
            &bucket_data[4..4 + LEGACY_BUCKET_DATA_MAGIC.len()]
                == LEGACY_BUCKET_DATA_MAGIC.as_slice()
        },
        "bucket sidecar has an invalid header; run ./warp-cli reindex"
    );
    let centroid_count: usize = read_u64_le(bucket_data, 12)?.try_into()?;
    let meta_offset: usize = read_u64_le(bucket_data, 20)?.try_into()?;
    let payload_offset: usize = read_u64_le(bucket_data, 28)?.try_into()?;
    let rowids_offset: usize = if version == BUCKET_DATA_VERSION {
        read_u64_le(bucket_data, 36)?.try_into()?
    } else {
        bucket_data.len()
    };
    anyhow::ensure!(
        meta_offset >= header_bytes
            && payload_offset >= meta_offset
            && rowids_offset >= payload_offset
            && rowids_offset <= bucket_data.len(),
        "bucket sidecar layout is invalid"
    );
    let meta_block_len = payload_offset - meta_offset;
    let meta_bytes = if centroid_count == 0 {
        bucket_meta_bytes_for_dim(DEFAULT_EMBEDDING_DIM)
    } else {
        anyhow::ensure!(
            meta_block_len % centroid_count == 0,
            "bucket sidecar metadata block is not divisible by centroid count"
        );
        meta_block_len / centroid_count
    };
    anyhow::ensure!(
        meta_bytes >= BUCKET_META_PREFIX_BYTES,
        "bucket sidecar metadata entry is too small"
    );
    let center_bytes = meta_bytes - BUCKET_META_PREFIX_BYTES;
    anyhow::ensure!(
        center_bytes % std::mem::size_of::<f32>() == 0,
        "bucket sidecar center byte length is not a multiple of f32"
    );
    let embedding_dim = center_bytes / std::mem::size_of::<f32>();
    anyhow::ensure!(
        payload_offset <= rowids_offset,
        "bucket sidecar layout is invalid"
    );
    Ok(BucketDataHeader {
        centroid_count,
        embedding_dim,
        meta_bytes,
        meta_offset,
        payload_offset,
        rowids_offset,
    })
}

fn bucket_data_bucket_meta(
    bucket_data: &[u8],
    header: &BucketDataHeader,
) -> Result<Vec<BucketSidecarMeta>> {
    let mut metas = Vec::with_capacity(header.centroid_count);
    for bucket_idx in 0..header.centroid_count {
        let offset = header.meta_offset + bucket_idx * header.meta_bytes;
        let size = read_u32_le(bucket_data, offset)? as usize;
        let data_offset: u64 = read_u64_le(bucket_data, offset + 4)?;
        let indices_len = read_u32_le(bucket_data, offset + 12)? as usize;
        let residual_len = read_u32_le(bucket_data, offset + 16)? as usize;
        let center_start = offset + BUCKET_META_PREFIX_BYTES;
        let center_end = offset + header.meta_bytes;
        anyhow::ensure!(
            data_offset >= header.payload_offset as u64,
            "bucket {bucket_idx} data offset is before payload block"
        );
        let bucket_end = data_offset
            .checked_add(u64::try_from(indices_len)?)
            .and_then(|offset| offset.checked_add(u64::try_from(residual_len).ok()?))
            .ok_or_else(|| anyhow::anyhow!("bucket {bucket_idx} data length overflow"))?;
        anyhow::ensure!(
            bucket_end <= header.rowids_offset as u64,
            "bucket {bucket_idx} data ends beyond bucket payload"
        );
        metas.push(BucketSidecarMeta {
            size,
            data_offset,
            indices_len,
            residual_len,
            center: bucket_data[center_start..center_end].to_vec(),
        });
    }
    Ok(metas)
}

fn write_bucket_meta(writer: &mut impl Write, meta: &BucketSidecarMeta) -> Result<()> {
    let center_bytes = meta.center.len();
    anyhow::ensure!(
        center_bytes % std::mem::size_of::<f32>() == 0,
        "bucket center byte length {center_bytes} is not a multiple of f32"
    );
    writer.write_all(&u32::try_from(meta.size)?.to_le_bytes())?;
    writer.write_all(&meta.data_offset.to_le_bytes())?;
    writer.write_all(&u32::try_from(meta.indices_len)?.to_le_bytes())?;
    writer.write_all(&u32::try_from(meta.residual_len)?.to_le_bytes())?;
    writer.write_all(&meta.center)?;
    Ok(())
}

fn merge_and_write_buckets_to_path(
    tmpfiles: Vec<tempfile::NamedTempFile>,
    centers_cpu: &Tensor,
    rowid_records: &[RowidRecord],
    final_path: &Path,
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
    let (centroid_count, center_dim) = centers_cpu.dims2()?;
    let center_bytes = center_bytes_for_dim(center_dim);
    let bucket_meta_bytes = bucket_meta_bytes_for_dim(center_dim);
    let residual_bytes = residual_bytes_for_dim(center_dim);
    let mut bucket_meta: Vec<BucketSidecarMeta> = (0..centroid_count)
        .map(|_| BucketSidecarMeta {
            size: 0,
            data_offset: 0,
            indices_len: 0,
            residual_len: 0,
            center: vec![0; center_bytes],
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
        let center = centers_cpu.get(bucket_idx)?;
        let center_bytes = center.to_f32_bytes()?;
        anyhow::ensure!(
            center_bytes.len() == center_bytes_for_dim(center_dim),
            "bucket {} center byte length {} does not match expected {}",
            entry.value,
            center_bytes.len(),
            center_bytes_for_dim(center_dim)
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
        meta.center = center_bytes;
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

    let meta_block_len = centroid_count
        .checked_mul(bucket_meta_bytes)
        .ok_or_else(|| anyhow::anyhow!("bucket metadata block length overflow"))?;
    let meta_offset = BUCKET_DATA_HEADER_BYTES;
    let payload_start = meta_offset
        .checked_add(meta_block_len)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar payload start overflow"))?;
    let payload_start = u64::try_from(payload_start)?;
    let rowids_offset = payload_start
        .checked_add(data_offset)
        .ok_or_else(|| anyhow::anyhow!("bucket sidecar rowid offset overflow"))?;

    let mut bucket_data_writer = NewFileWriter::create_new(&tmp_path)?;

    write_bucket_data_header(
        &mut bucket_data_writer,
        centroid_count,
        meta_offset,
        payload_start.try_into()?,
        rowids_offset.try_into()?,
    )?;
    for meta in &mut bucket_meta {
        meta.data_offset = payload_start
            .checked_add(meta.data_offset)
            .ok_or_else(|| anyhow::anyhow!("bucket sidecar absolute offset overflow"))?;
        write_bucket_meta(&mut bucket_data_writer, meta)?;
    }
    let mut data_reader = data_tmp.reopen()?;
    std::io::copy(&mut data_reader, &mut bucket_data_writer)?;
    write_rowid_records_to_writer(&mut bucket_data_writer, rowid_records)?;
    bucket_data_writer.finish()?;
    std::fs::rename(&tmp_path, &final_path)?;
    sync_parent_dir(final_path)?;
    Ok(())
}

#[cfg(feature = "sqlite")]
fn fts5_query(q: &str) -> Option<(String, String)> {
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

#[cfg(feature = "sqlite")]
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

/// Per-generation centroid data loaded from the database.
pub struct GenerationCentroids {
    dim: usize,
    residual_bytes: usize,
    bucket_indices: Vec<usize>,
    sizes: Vec<usize>,
    data_offsets: Vec<usize>,
    indices_lens: Vec<usize>,
    residual_lens: Vec<usize>,
    bucket_data: Option<Arc<Mmap>>,
    centers_matrix: Tensor,
}

static GENERATIONS_CACHE: Lazy<RwLock<HashMap<Vec<PathBuf>, Arc<Vec<GenerationCentroids>>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn invalidate_generations_cache(paths: &[PathBuf]) {
    GENERATIONS_CACHE.write().unwrap().remove(paths);
}

fn clear_generations_cache() {
    GENERATIONS_CACHE.write().unwrap().clear();
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

/// Load generation centroid data from mmap sidecar files (cached).
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
        let mmap = unsafe { Mmap::map(&file)? };
        let bucket_data = Arc::new(mmap);

        let header = bucket_data_header(&bucket_data)?;
        let residual_bytes = residual_bytes_for_dim(header.embedding_dim);
        let metas = bucket_data_bucket_meta(&bucket_data, &header)?;

        let mut bucket_indices = vec![];
        let mut sizes = vec![];
        let mut data_offsets = vec![];
        let mut indices_lens = vec![];
        let mut residual_lens = vec![];
        let mut centers = vec![];

        for (bucket_idx, meta) in metas.into_iter().enumerate() {
            if meta.size == 0 {
                continue;
            }
            bucket_indices.push(bucket_idx);
            sizes.push(meta.size);
            data_offsets.push(meta.data_offset.try_into()?);
            indices_lens.push(meta.indices_len);
            residual_lens.push(meta.residual_len);
            let t = Tensor::from_f32_bytes(&meta.center, header.embedding_dim, &Device::Cpu)?.flatten_all()?;
            centers.push(t);
        }

        let centers_matrix = if !centers.is_empty() {
            Tensor::stack(&centers, 0)?.to_device(device)?
        } else {
            Tensor::zeros(&[0, header.embedding_dim], DType::F32, device)?
        };

        all.push(GenerationCentroids {
            dim: header.embedding_dim,
            residual_bytes,
            bucket_indices,
            sizes,
            data_offsets,
            indices_lens,
            residual_lens,
            bucket_data: Some(bucket_data),
            centers_matrix,
        });
    }

    let result = Arc::new(all);
    GENERATIONS_CACHE.write().unwrap().insert(key, result.clone());
    Ok(result)
}

/// Pure index search: scores query embeddings against generation sidecar files.
/// No database access — loads generations from mmap files directly.
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

    let table = packops::make_residual_dequant_table()?;

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

        let query_centroid_similarity =
            fast_ops::matmul_t(query_embeddings, &gen.centers_matrix)?;
        let query_centroid_similarity = query_centroid_similarity.to_device(&Device::Cpu)?;

        let gen_centroid_scores = query_centroid_similarity.to_vec2::<f32>()?;
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
            let selection_limit = n_centroids.min(k);
            let mut tail_rank = 0;
            for j in 0..selection_limit {
                let idx = row[j] as usize;
                topk_clusters.push(idx);
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

        for &i in &topk_clusters {
            let bucket_idx = i as usize;
            let bucket_id = gen.bucket_indices[bucket_idx];
            let data_offset = gen.data_offsets[bucket_idx];
            let indices_len = gen.indices_lens[bucket_idx];
            let residual_len = gen.residual_lens[bucket_idx];
            let bucket_data = gen.bucket_data.as_ref().ok_or_else(|| {
                anyhow::anyhow!("generation has no mapped bucket data sidecar")
            })?;
            let indices_end = data_offset + indices_len;
            let residual_end = indices_end + residual_len;
            anyhow::ensure!(
                residual_end <= bucket_data.len(),
                "bucket {bucket_id} data range {}..{} exceeds sidecar length {}",
                data_offset,
                residual_end,
                bucket_data.len()
            );
            let keys_compressed = &bucket_data[data_offset..indices_end];
            let residual_bytes = &bucket_data[indices_end..residual_end];

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
                &table,
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
            let centroid_scores = &gen_centroid_scores_all[gen_idx];
            let centroid_score_ranges = &gen_centroid_score_ranges_all[gen_idx];
            for (query_idx, scores) in centroid_scores.iter().enumerate().take(n) {
                let offset = doc_idx * n + query_idx;
                let centroid_score = scores[cluster_idx];
                let residual_weight = packops::residual_centroid_confidence_weight(
                    centroid_score,
                    centroid_score_ranges[query_idx],
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
    scored_results.truncate(top_k);

    debug!(
        "match_centroids_raw: {} embeddings in {} ms.",
        count,
        total_start.elapsed().as_millis()
    );
    Ok(scored_results)
}

/// DB-backed wrapper: loads generations from cache and fetches any unindexed
/// documents that already have cached embeddings.
#[cfg(feature = "sqlite")]
pub fn match_centroids(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<Vec<(f32, u32, u32)>> {
    let cache = default_embedding_cache();
    match_centroids_from_cache(db, query_embeddings, threshold, top_k, sql_filter, None, &cache)
}

/// DB-backed wrapper that can demand-populate missing unindexed document
/// embeddings through the supplied cache.
#[cfg(feature = "sqlite")]
pub fn match_centroids_with_cache(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&SqlStatementInternal>,
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

#[cfg(feature = "sqlite")]
fn match_centroids_from_cache(
    db: &DB,
    query_embeddings: &Tensor,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<&SqlStatementInternal>,
    embedder: Option<&Embedder>,
    cache: &dyn EmbeddingCache,
) -> Result<Vec<(f32, u32, u32)>> {
    let index = index_for_db(db);
    let generation_files = index.generation_files()?;
    let buffered = index.buffered_rowid_records()?;
    let unindexed = if !buffered.is_empty() {
        buffered_unindexed_embeddings(db, &buffered, cache, embedder)?
    } else if generation_files.is_empty() {
        current_unindexed_embeddings(db, cache, embedder)?
    } else {
        vec![]
    };

    let scored_results = match_centroids_raw(
        &generation_files, query_embeddings, &unindexed, threshold, top_k,
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

#[cfg(feature = "sqlite")]
fn buffered_unindexed_embeddings(
    db: &DB,
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<Vec<(Vec<DocPtr>, Tensor)>> {
    let records: Vec<RowidRecord> = active_rowid_records(records).collect();
    if records.is_empty() {
        return Ok(vec![]);
    }

    let mut query = db.query(
        "SELECT body, lens FROM document
         WHERE rowid = ?1 AND length(body) > 0",
    )?;
    let mut unindexed = vec![];
    for record in records {
        let rowid: i64 = record.rowid.try_into()?;
        let row = query
            .query_row((rowid,), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .optional()?;
        let Some((body, lens)) = row else {
            continue;
        };
        let hash = document_cache_hash(&body, &lens);
        let Some(cached) =
            cached_embeddings_for_document(cache, embedder, record.rowid, &hash, &body, &lens)?
        else {
            continue;
        };
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
        let id: u32 = record.rowid.try_into()?;
        let indices = docptrs_for_counts(id, &cached.counts);
        anyhow::ensure!(
            indices.len() == cached.embedding_count,
            "rowid {} has {} embedding rows but {} document indices",
            record.rowid,
            cached.embedding_count,
            indices.len()
        );
        unindexed.push((indices, embeddings));
    }
    Ok(unindexed)
}

#[cfg(feature = "sqlite")]
fn current_unindexed_embeddings(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<Vec<(Vec<DocPtr>, Tensor)>> {
    let mut query = db.query(
        "SELECT rowid, body, lens FROM document
         WHERE length(body) > 0
         ORDER BY rowid",
    )?;
    let results = query.query_map((), |row| {
        Ok((
            row.get::<_, u32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    let mut unindexed = vec![];
    for result in results {
        let (id, body, lens) = result?;
        let hash = document_cache_hash(&body, &lens);
        let Some(cached) =
            cached_embeddings_for_document(cache, embedder, id as u64, &hash, &body, &lens)?
        else {
            continue;
        };
        let embeddings = Tensor::embeddings_from_packed(
            &cached.embeddings,
            dim_from_model_id(&cached.model),
            &Device::Cpu,
        )?;
        let indices = docptrs_for_counts(id, &cached.counts);
        anyhow::ensure!(
            indices.len() == cached.embedding_count,
            "document {id} has {} embedding rows but {} document indices",
            cached.embedding_count,
            indices.len()
        );
        unindexed.push((indices, embeddings));
    }
    Ok(unindexed)
}

#[cfg(feature = "sqlite")]
fn index_for_db(db: &DB) -> FileBackedIndex {
    FileBackedIndex::new(db.path().clone())
}

fn cached_embeddings_for_rowid(
    cache: &dyn EmbeddingCache,
    rowid: u64,
) -> Result<CachedEmbeddings> {
    load_cached_embeddings(cache, rowid, "")?
        .ok_or_else(|| anyhow::anyhow!("missing embeddings for rowid {rowid}"))
}

#[cfg(feature = "sqlite")]
fn current_document_rowid_records(
    db: &DB,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<Vec<RowidRecord>> {
    let mut query = db.query(
        "SELECT rowid, body, lens FROM document
         WHERE length(body) > 0
         ORDER BY rowid",
    )?;
    let rows = query.query_map((), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    let mut records = vec![];
    for row in rows {
        let (rowid, body, lens) = row?;
        let rowid_u64: u64 = rowid.try_into()?;
        let hash = document_cache_hash(&body, &lens);
        let Some(embeddings) =
            cached_embeddings_for_document(cache, embedder, rowid_u64, &hash, &body, &lens)?
        else {
            continue;
        };
        records.push(RowidRecord {
            rowid: rowid_u64,
            rows: embeddings.embedding_count.try_into()?,
        });
    }
    Ok(records)
}

#[cfg(feature = "sqlite")]
fn pending_rowid_records(
    db: &DB,
    index: &FileBackedIndex,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<Vec<RowidRecord>> {
    let current_records = current_document_rowid_records(db, cache, embedder)?;
    let mut known = index.all_rowid_map()?;
    let mut pending = vec![];

    for record in current_records {
        match known.remove(&record.rowid) {
            Some(rows) if rows == record.rows => {}
            _ => pending.push(record),
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

#[cfg(feature = "sqlite")]
struct DocumentEmbeddingSource<'a> {
    cache: &'a dyn EmbeddingCache,
    hashes: HashMap<u64, String>,
}

#[cfg(feature = "sqlite")]
impl<'a> DocumentEmbeddingSource<'a> {
    fn new(db: &DB, cache: &'a dyn EmbeddingCache) -> Result<Self> {
        let mut query = db.query(
            "SELECT rowid, body, lens FROM document
             WHERE length(body) > 0",
        )?;
        let rows = query.query_map((), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut hashes = HashMap::new();
        for row in rows {
            let (rowid, body, lens) = row?;
            hashes.insert(rowid.try_into()?, document_cache_hash(&body, &lens));
        }
        Ok(Self { cache, hashes })
    }
}

#[cfg(feature = "sqlite")]
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

#[cfg(feature = "sqlite")]
fn unmaterialized_embedding_count(
    db: &DB,
    index: &FileBackedIndex,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
) -> Result<usize> {
    let indexed = index.indexed_rowid_map()?;
    let mut count = 0usize;
    for record in current_document_rowid_records(db, cache, embedder)? {
        if indexed.get(&record.rowid).copied() != Some(record.rows) {
            count += record.rows as usize;
        }
    }
    Ok(count)
}

fn sample_embeddings_for_rowids(
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    device: &Device,
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
        return Ok((Tensor::zeros(&[0, DEFAULT_EMBEDDING_DIM], DType::F32, device)?, 0));
    }
    let matrix = Tensor::stack(&all_embeddings, 0)?.to_device(device)?;
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

pub(crate) fn compute_cached_embeddings(
    embedder: &Embedder,
    body: &str,
    lens: &str,
) -> Result<CachedEmbeddings> {
    let now = std::time::Instant::now();
    let (embeddings, offsets) = embedder.embed(body)?;
    let embeddings = embeddings.squeeze(0)?.to_device(&Device::Cpu)?;
    let (rows, cols) = embeddings.dims2()?;
    let dt = now.elapsed().as_secs_f64();
    debug!(
        "embedder took {} ms ({} rows/s).",
        now.elapsed().as_millis(),
        ((rows as f64) / dt).round()
    );

    let counts = token_counts_for_lens(lens, &offsets)?;
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

#[cfg(feature = "sqlite")]
pub fn embed_chunks_with_cache(
    db: &DB,
    embedder: &Embedder,
    cache: &dyn EmbeddingCache,
    limit: Option<usize>,
) -> Result<usize> {
    let _priority_mgr = PriorityManager::new();

    // Count total documents to embed for progress reporting
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
        document.rowid,document.body,document.lens
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
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    let mut count = 0;
    for result in documents.by_ref() {
        let (rowid, body, lens) = result?;
        let hash = document_cache_hash(&body, &lens);
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

#[cfg(feature = "sqlite")]
pub fn embed_chunks(db: &DB, embedder: &Embedder, limit: Option<usize>) -> Result<usize> {
    let cache = default_embedding_cache();
    embed_chunks_with_cache(db, embedder, &cache, limit)
}

#[cfg(feature = "sqlite")]
pub fn count_unindexed_embeddings(db: &DB) -> Result<usize> {
    let cache = default_embedding_cache();
    let index = index_for_db(db);
    unmaterialized_embedding_count(db, &index, &cache, None)
}

#[cfg(feature = "sqlite")]
pub fn count_unindexed_embeddings_with_cache(
    db: &DB,
    embedder: &Embedder,
    cache: &dyn EmbeddingCache,
) -> Result<usize> {
    let index = index_for_db(db);
    unmaterialized_embedding_count(
        db,
        &index,
        cache,
        Some(embedder),
    )
}

#[cfg(feature = "sqlite")]
pub fn count_unindexed_cached_embeddings(db: &DB, cache: &dyn EmbeddingCache) -> Result<usize> {
    let index = index_for_db(db);
    unmaterialized_embedding_count(db, &index, cache, None)
}

#[cfg(feature = "sqlite")]
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

fn run_kmeans_for_index(matrix: &Tensor, total_embeddings: usize) -> Result<Tensor> {
    let now = std::time::Instant::now();
    let mut k = (16.0 * (total_embeddings as f64).sqrt()).round() as usize;
    k = k.max(1);
    debug!("total_embeddings={} k={}", total_embeddings, k);
    let (m, _) = matrix.dims2()?;
    if m < k {
        k = (m / 4).max(1);
    }
    let centers = kmeans(matrix, k, 5)?;
    debug!("kmeans took {} ms.", now.elapsed().as_millis());
    Ok(centers)
}

fn level_capacity(level: u32) -> usize {
    L0_CAPACITY * LSM_FANOUT.pow(level + 1)
}

fn write_buckets_for_rowids(
    records: &[RowidRecord],
    cache: &dyn EmbeddingCache,
    centers: &Tensor,
    device: &Device,
    expected_count: u64,
) -> Result<(Vec<tempfile::NamedTempFile>, Tensor)> {
    let _priority_mgr = PriorityManager::new();
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let bar = progress::new_with_label(expected_count, "indexing");

    let mut document_indices = Vec::<(u32, u32)>::new();
    let mut all_embeddings = vec![];

    let mut records = active_rowid_records(records);
    let mut done = false;
    let mut batch = 0;
    let mut tmpfiles = vec![];
    let centers_cpu = centers.to_device(&Device::Cpu)?;
    let (_, center_dim) = centers_cpu.dims2()?;
    let residual_bytes = packops::temp_residual_bytes_for_dim(center_dim);
    let packed_centers = fast_ops::PackedRight::new(centers)?;
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
                let split = split_tensor(&t);
                let m = split.len();

                let docptrs = docptrs_for_counts(id, &embeddings.counts);
                anyhow::ensure!(
                    docptrs.len() == m,
                    "document {id} has {m} embedding rows but {} document indices",
                    docptrs.len()
                );
                document_indices.extend(docptrs);
                all_embeddings.extend(split);
                batch += m;
            }
            None => {
                done = true;
            }
        }

        let batch_size = 0x10000;

        if batch >= batch_size || done {
            if batch == 0 {
                continue;
            }
            let now = std::time::Instant::now();

            let take = batch.min(batch_size);
            let left = batch - take;

            let embeddings = all_embeddings.split_off(left);
            let indices = document_indices.split_off(left);
            let data = Tensor::cat(&embeddings, 0)?.to_device(device)?;

            let cluster_assignments =
                matmul_argmax_batched(&data, &packed_centers, 1024)?.to_device(&Device::Cpu)?;
            mmuls_total += now.elapsed().as_millis();

            let now = std::time::Instant::now();
            let mut writer = merger::Writer::new(residual_bytes)?;

            let mut pairs: Vec<(usize, u32)> = cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .map(|(i, &bucket)| (i, bucket))
                .collect();
            pairs.sort_by_key(|&(_, bucket)| bucket);

            let mut keys: Vec<(u32, u32)> = Vec::with_capacity(take);
            let mut residuals_bytes: Vec<u8> = Vec::with_capacity(take * residual_bytes);
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
                let residual_quantized = packops::residual_to_temp_bytes(&residual)?;
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

    Ok((tmpfiles, centers_cpu))
}

fn build_index_generation(
    index: &FileBackedIndex,
    device: &Device,
    cache: &dyn EmbeddingCache,
    level: u32,
    records: &[RowidRecord],
) -> Result<Option<FileIndexGeneration>> {
    let active_embeddings = rowid_records_embedding_count(records);
    if active_embeddings == 0 {
        return Ok(None);
    }

    let (matrix, total_embeddings) = sample_embeddings_for_rowids(records, cache, device)?;
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

    let (tmpfiles, centers_cpu) =
        write_buckets_for_rowids(records, cache, &centers, device, total_embeddings as u64)?;
    if let Err(err) = merge_and_write_buckets_to_path(tmpfiles, &centers_cpu, records, &data_path) {
        let _ = std::fs::remove_file(&data_path);
        return Err(err);
    }

    Ok(Some(FileIndexGeneration {
        level,
        num_embeddings: total_embeddings,
        data_file,
        rowids_file: None,
    }))
}

pub(crate) fn index_buffered_embeddings(
    index: &FileBackedIndex,
    device: &Device,
    cache: &dyn EmbeddingCache,
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
        build_index_generation(index, device, cache, target_level, &merged)?
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

#[cfg(feature = "sqlite")]
pub fn index_chunks(
    db: &DB,
    device: &Device,
    cache: &dyn EmbeddingCache,
    embedder: Option<&Embedder>,
    reset: bool,
) -> Result<()> {
    let index = index_for_db(db);
    if reset {
        index.clear()?;
        clear_generations_cache();
        db.remove_all_bucket_data_sidecars();
    }

    let unmaterialized = unmaterialized_embedding_count(db, &index, cache, embedder)?;
    let pending = pending_rowid_records(db, &index, cache, embedder)?;
    if pending.is_empty() && unmaterialized == 0 {
        return Ok(());
    }

    for record in &pending {
        index.append_rowid_record(record.rowid, record.rows)?;
    }

    let x = unmaterialized;
    let indexed = index.indexed_embedding_count()?;
    info!("database has {} unindexed embeddings ({} indexed)", x, indexed);

    let source = DocumentEmbeddingSource::new(db, cache)?;
    index_buffered_embeddings(&index, device, &source)?;
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

#[cfg(feature = "sqlite")]
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
        let embedding_cache = default_embedding_cache();
        match match_centroids_with_cache(
            db,
            &qe,
            threshold,
            top_k,
            sql_filter,
            embedder,
            &embedding_cache,
        ) {
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

#[cfg(feature = "sqlite")]
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
    let embedding_cache = default_embedding_cache();
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

#[cfg(feature = "sqlite")]
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

#[cfg(feature = "sqlite")]
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
        reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
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
