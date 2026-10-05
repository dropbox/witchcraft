use anyhow::{anyhow, bail, Context, Result};
use candle_core::Device;
use serde::Deserialize;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use witchcraft::{cached_embeddings_from_output, Embedder, EmbeddingOutput};

const CACHE_BATCH_MAGIC: [u8; 8] = *b"WECB0001";
const DEFAULT_ADDR: &str = "0.0.0.0:7860";
const DEFAULT_GPU_BATCH_SIZE: usize = 32;
#[cfg(feature = "modernbert")]
const MODEL_ASSET: &str = "modernbert.safetensors";
#[cfg(feature = "modernbert-quantized")]
const MODEL_ASSET: &str = "modernbert.gguf";
const REQUIRED_ASSETS: [&str; 3] = [
    "modernbert-config.json",
    "modernbert-tokenizer.json",
    MODEL_ASSET,
];

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TextInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct PredictRequest {
    text: TextInput,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    include_tokens: Option<bool>,
    #[serde(default)]
    include_offsets: Option<bool>,
}

fn main() -> Result<()> {
    let addr = env::var("WITCHCRAFT_KSERVE_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string());
    let assets = assets_dir()?;
    let device = device_from_env()?;
    let gpu_batch_size = gpu_batch_size_from_env();
    let pack_worker_count = pack_worker_count_from_env();
    eprintln!(
        "witchcraft-kserve: loading ModernBERT assets={} device={device:?} gpu_batch_size={} pack_workers={}",
        assets.display(),
        gpu_batch_size,
        pack_worker_count,
    );
    let embedder = Embedder::new(&device, &assets).context("load Witchcraft embedder")?;
    let listener = TcpListener::bind(&addr).with_context(|| format!("bind {addr}"))?;
    eprintln!("witchcraft-kserve: listening on {addr}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("witchcraft-kserve: accept failed: {err}");
                continue;
            }
        };
        if let Err(err) =
            handle_connection(&mut stream, &embedder, gpu_batch_size, pack_worker_count)
        {
            let body = serde_json::json!({ "error": err.to_string() }).to_string();
            let _ = write_response(
                &mut stream,
                "500 Internal Server Error",
                "application/json",
                body.as_bytes(),
            );
        }
    }
    Ok(())
}

fn handle_connection(
    stream: &mut TcpStream,
    embedder: &Embedder,
    gpu_batch_size: usize,
    pack_worker_count: usize,
) -> Result<()> {
    let request = read_http_request(stream)?;
    if request.method == "GET" {
        let body = br#"{"status":"ok"}"#;
        return write_response(stream, "200 OK", "application/json", body);
    }
    if request.method != "POST" {
        return write_response(
            stream,
            "405 Method Not Allowed",
            "text/plain",
            b"method not allowed",
        );
    }
    if !request.path.ends_with(":predict") && request.path != "/predict" {
        return write_response(stream, "404 Not Found", "text/plain", b"not found");
    }

    let started = Instant::now();
    let predict: PredictRequest =
        serde_json::from_slice(&request.body).context("parse predict request JSON")?;
    validate_predict_request(&predict)?;
    let texts = match predict.text {
        TextInput::One(text) => vec![text],
        TextInput::Many(texts) => texts,
    };
    let lens = texts
        .iter()
        .map(|text| text.chars().count().to_string())
        .collect::<Vec<_>>();
    let pipeline =
        compute_packed_entries_pipeline(embedder, texts, lens, gpu_batch_size, pack_worker_count)?;
    let entries = pipeline.entries;
    let body = encode_cache_batch(&entries)?;
    eprintln!(
        "witchcraft-kserve: docs={} gpu_batches={} gpu_batch_size={} embeddings={} embed_ms={} pack_cpu_ms={} cache_entry_bytes={} response_bytes={} elapsed_ms={}",
        entries.len(),
        pipeline.gpu_batches,
        gpu_batch_size,
        pipeline.embedding_count,
        pipeline.embed_elapsed.as_millis(),
        pipeline.pack_elapsed.as_millis(),
        pipeline.packed_bytes,
        body.len(),
        started.elapsed().as_millis()
    );
    write_response(
        stream,
        "200 OK",
        "application/vnd.dropbox.witchcraft-cache-batch",
        &body,
    )
}

struct PipelineResult {
    entries: Vec<Vec<u8>>,
    embedding_count: usize,
    packed_bytes: usize,
    gpu_batches: usize,
    embed_elapsed: Duration,
    pack_elapsed: Duration,
}

struct PackJob {
    index: usize,
    body: String,
    lens: String,
    output: EmbeddingOutput,
}

struct PackedEntry {
    index: usize,
    embedding_count: usize,
    bytes: Vec<u8>,
    elapsed: Duration,
}

struct PackResult {
    index: usize,
    result: Result<PackedEntry, String>,
}

fn compute_packed_entries_pipeline(
    embedder: &Embedder,
    texts: Vec<String>,
    lens: Vec<String>,
    gpu_batch_size: usize,
    pack_worker_count: usize,
) -> Result<PipelineResult> {
    anyhow::ensure!(
        texts.len() == lens.len(),
        "cached embedding batch has {} bodies and {} lens entries",
        texts.len(),
        lens.len()
    );
    if texts.is_empty() {
        return Ok(PipelineResult {
            entries: Vec::new(),
            embedding_count: 0,
            packed_bytes: 0,
            gpu_batches: 0,
            embed_elapsed: Duration::ZERO,
            pack_elapsed: Duration::ZERO,
        });
    }

    let gpu_batch_size = gpu_batch_size.max(1);
    let pack_worker_count = pack_worker_count.max(1);
    let doc_count = texts.len();
    let (pack_tx, pack_rx) = std_mpsc::sync_channel(pack_worker_count * 2);
    let (result_tx, result_rx) = std_mpsc::channel();
    let pack_workers = spawn_pack_workers(pack_rx, result_tx, pack_worker_count)?;

    let mut scheduled_docs = 0usize;
    let mut gpu_batches = 0usize;
    let mut embed_elapsed = Duration::ZERO;
    let mut batch = Vec::with_capacity(gpu_batch_size);
    let mut inference_error = None;

    for (index, (body, lens)) in texts.into_iter().zip(lens).enumerate() {
        batch.push((index, body, lens));
        if batch.len() == gpu_batch_size {
            match process_gpu_batch(embedder, std::mem::take(&mut batch), &pack_tx) {
                Ok((docs, elapsed)) => {
                    scheduled_docs += docs;
                    gpu_batches += 1;
                    embed_elapsed += elapsed;
                }
                Err(err) => {
                    inference_error = Some(err);
                    break;
                }
            }
        }
    }
    if inference_error.is_none() && !batch.is_empty() {
        match process_gpu_batch(embedder, batch, &pack_tx) {
            Ok((docs, elapsed)) => {
                scheduled_docs += docs;
                gpu_batches += 1;
                embed_elapsed += elapsed;
            }
            Err(err) => {
                inference_error = Some(err);
            }
        }
    }

    drop(pack_tx);
    let pack_result = collect_pack_results(result_rx, doc_count, scheduled_docs);
    join_pack_workers(pack_workers)?;
    if let Some(err) = inference_error {
        return Err(err);
    }
    let (entries, embedding_count, packed_bytes, pack_elapsed) = pack_result?;
    Ok(PipelineResult {
        entries,
        embedding_count,
        packed_bytes,
        gpu_batches,
        embed_elapsed,
        pack_elapsed,
    })
}

fn process_gpu_batch(
    embedder: &Embedder,
    batch: Vec<(usize, String, String)>,
    pack_tx: &std_mpsc::SyncSender<PackJob>,
) -> Result<(usize, Duration)> {
    let docs = batch.len();
    let bodies = batch
        .iter()
        .map(|(_, body, _)| body.clone())
        .collect::<Vec<_>>();
    let started = Instant::now();
    let outputs = embedder.embed_batch_with_gate_scores_and_tokens(&bodies)?;
    let elapsed = started.elapsed();
    anyhow::ensure!(
        outputs.len() == docs,
        "embedding batch produced {} outputs for {} docs",
        outputs.len(),
        docs
    );
    for ((index, body, lens), output) in batch.into_iter().zip(outputs) {
        pack_tx
            .send(PackJob {
                index,
                body,
                lens,
                output,
            })
            .map_err(|_| anyhow!("cache-entry pack worker stopped"))?;
    }
    Ok((docs, elapsed))
}

fn spawn_pack_workers(
    jobs: std_mpsc::Receiver<PackJob>,
    results: std_mpsc::Sender<PackResult>,
    worker_count: usize,
) -> Result<Vec<JoinHandle<()>>> {
    let jobs = Arc::new(Mutex::new(jobs));
    let mut handles = Vec::with_capacity(worker_count);
    for worker_id in 0..worker_count {
        let jobs = Arc::clone(&jobs);
        let results = results.clone();
        let handle = std::thread::Builder::new()
            .name(format!("witchcraft-pack-{worker_id}"))
            .spawn(move || pack_worker(worker_id, jobs, results))
            .with_context(|| format!("spawn cache-entry pack worker {worker_id}"))?;
        handles.push(handle);
    }
    Ok(handles)
}

fn pack_worker(
    worker_id: usize,
    jobs: Arc<Mutex<std_mpsc::Receiver<PackJob>>>,
    results: std_mpsc::Sender<PackResult>,
) {
    loop {
        let job = match jobs.lock() {
            Ok(jobs) => jobs.recv(),
            Err(err) => {
                eprintln!("witchcraft-kserve: pack_worker={worker_id} queue lock poisoned: {err}");
                break;
            }
        };
        let Ok(job) = job else {
            break;
        };
        let index = job.index;
        let result = pack_cache_entry(job).map_err(|err| err.to_string());
        if results.send(PackResult { index, result }).is_err() {
            break;
        }
    }
}

fn pack_cache_entry(job: PackJob) -> Result<PackedEntry> {
    let started = Instant::now();
    let cached = cached_embeddings_from_output(&job.body, &job.lens, job.output)?;
    let embedding_count = cached.embedding_count;
    let bytes = cached.to_cache_entry_bytes()?;
    Ok(PackedEntry {
        index: job.index,
        embedding_count,
        bytes,
        elapsed: started.elapsed(),
    })
}

fn collect_pack_results(
    result_rx: std_mpsc::Receiver<PackResult>,
    doc_count: usize,
    scheduled_docs: usize,
) -> Result<(Vec<Vec<u8>>, usize, usize, Duration)> {
    let mut entries = (0..doc_count).map(|_| None).collect::<Vec<_>>();
    let mut embedding_count = 0usize;
    let mut packed_bytes = 0usize;
    let mut pack_elapsed = Duration::ZERO;
    let mut first_error = None;
    for _ in 0..scheduled_docs {
        let packed = result_rx
            .recv()
            .context("receive packed cache-entry from worker")?;
        match packed.result {
            Ok(entry) => {
                embedding_count += entry.embedding_count;
                packed_bytes += entry.bytes.len();
                pack_elapsed += entry.elapsed;
                entries[entry.index] = Some(entry.bytes);
            }
            Err(err) if first_error.is_none() => {
                first_error = Some(format!("pack cache entry {}: {err}", packed.index));
            }
            Err(_) => {}
        }
    }
    if let Some(err) = first_error {
        bail!("{err}");
    }
    let entries = entries
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| entry.ok_or_else(|| anyhow!("missing packed cache entry {idx}")))
        .collect::<Result<Vec<_>>>()?;
    Ok((entries, embedding_count, packed_bytes, pack_elapsed))
}

fn join_pack_workers(handles: Vec<JoinHandle<()>>) -> Result<()> {
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("cache-entry pack worker panicked"))?;
    }
    Ok(())
}

fn validate_predict_request(request: &PredictRequest) -> Result<()> {
    match request
        .response_format
        .as_deref()
        .unwrap_or("cache_entries")
    {
        "cache" | "cache_entries" | "cached_embeddings" => {}
        other => bail!("unsupported response_format {other:?}; expected cache_entries"),
    }
    if request.normalize == Some(false) {
        bail!("cache_entries response is always normalized");
    }
    if request.include_tokens == Some(true) || request.include_offsets == Some(true) {
        bail!("cache_entries response does not include tokens or offsets");
    }
    Ok(())
}

fn encode_cache_batch(entries: &[Vec<u8>]) -> Result<Vec<u8>> {
    let total_len = CACHE_BATCH_MAGIC.len()
        + std::mem::size_of::<u32>()
        + entries
            .iter()
            .map(|entry| std::mem::size_of::<u64>() + entry.len())
            .sum::<usize>();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&CACHE_BATCH_MAGIC);
    write_u32(&mut out, entries.len())?;
    for entry in entries {
        write_u64(&mut out, entry.len())?;
        out.extend_from_slice(entry);
    }
    Ok(out)
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let max_request_bytes = env::var("WITCHCRAFT_MAX_REQUEST_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);
    let mut buffer = Vec::new();
    let mut scratch = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut scratch).context("read request")?;
        if n == 0 {
            bail!("connection closed before HTTP headers");
        }
        buffer.extend_from_slice(&scratch[..n]);
        if buffer.len() > max_request_bytes {
            bail!("request exceeds WITCHCRAFT_MAX_REQUEST_BYTES");
        }
        if let Some(idx) = find_header_end(&buffer) {
            break idx;
        }
    };

    let headers = std::str::from_utf8(&buffer[..header_end]).context("headers are not UTF-8")?;
    let mut lines = headers.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow!("missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| anyhow!("missing HTTP method"))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| anyhow!("missing HTTP path"))?
        .to_string();

    let mut content_length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value
                .trim()
                .parse::<usize>()
                .context("parse Content-Length")?;
        }
    }

    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let n = stream.read(&mut scratch).context("read request body")?;
        if n == 0 {
            bail!("connection closed before full request body");
        }
        buffer.extend_from_slice(&scratch[..n]);
        if buffer.len() > max_request_bytes {
            bail!("request exceeds WITCHCRAFT_MAX_REQUEST_BYTES");
        }
    }
    let body = buffer[body_start..body_start + content_length].to_vec();
    Ok(HttpRequest { method, path, body })
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn write_u32(out: &mut Vec<u8>, value: usize) -> Result<()> {
    out.extend_from_slice(&u32::try_from(value)?.to_le_bytes());
    Ok(())
}

fn write_u64(out: &mut Vec<u8>, value: usize) -> Result<()> {
    out.extend_from_slice(&u64::try_from(value)?.to_le_bytes());
    Ok(())
}

fn assets_dir() -> Result<PathBuf> {
    let mut roots = Vec::new();
    for key in [
        "WITCHCRAFT_ASSETS",
        "HF_MODEL",
        "SENTENCE_TRANSFORMERS_HOME",
    ] {
        if let Ok(path) = env::var(key) {
            roots.push(PathBuf::from(path));
        }
    }
    roots.extend([
        PathBuf::from("/mnt/models"),
        PathBuf::from("/mnt/models/assets"),
        PathBuf::from("assets"),
    ]);
    for root in roots {
        if let Some(path) = find_assets_dir(&root)? {
            return Ok(path);
        }
    }
    bail!(
        "could not find Witchcraft ModernBERT assets; set WITCHCRAFT_ASSETS to a directory containing {}",
        REQUIRED_ASSETS.join(", ")
    )
}

fn find_assets_dir(root: &Path) -> Result<Option<PathBuf>> {
    find_assets_dir_at_depth(root, 6)
}

fn find_assets_dir_at_depth(root: &Path, depth: usize) -> Result<Option<PathBuf>> {
    if has_required_assets(root) {
        return Ok(Some(root.to_path_buf()));
    }
    let nested_assets = root.join("assets");
    if has_required_assets(&nested_assets) {
        return Ok(Some(nested_assets));
    }
    if depth == 0 || !root.exists() {
        return Ok(None);
    }

    let mut dirs = Vec::new();
    for entry in
        fs::read_dir(root).with_context(|| format!("read asset root {}", root.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    for dir in dirs {
        if let Some(found) = find_assets_dir_at_depth(&dir, depth - 1)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

fn has_required_assets(path: &Path) -> bool {
    REQUIRED_ASSETS
        .iter()
        .all(|asset| path.join(asset).is_file())
}

fn gpu_batch_size_from_env() -> usize {
    env_usize("WITCHCRAFT_KSERVE_GPU_BATCH_SIZE")
        .or_else(|| env_usize("WITCHCRAFT_EMBED_BATCH_SIZE"))
        .unwrap_or(DEFAULT_GPU_BATCH_SIZE)
}

fn pack_worker_count_from_env() -> usize {
    env_usize("WITCHCRAFT_KSERVE_PACK_WORKERS").unwrap_or_else(default_pack_worker_count)
}

fn default_pack_worker_count() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get().saturating_sub(1).max(1))
}

fn env_usize(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn device_from_env() -> Result<Device> {
    match env::var("WITCHCRAFT_DEVICE")
        .unwrap_or_else(|_| "cpu".to_string())
        .to_lowercase()
        .as_str()
    {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "cuda:0" => cuda_device(0),
        other => bail!("unsupported WITCHCRAFT_DEVICE {other:?}; expected cpu or cuda"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_device(index: usize) -> Result<Device> {
    Device::new_cuda(index).context("create CUDA device")
}

#[cfg(not(feature = "cuda"))]
fn cuda_device(_index: usize) -> Result<Device> {
    bail!("this witchcraft_kserve_server binary was not built with Candle CUDA support")
}
