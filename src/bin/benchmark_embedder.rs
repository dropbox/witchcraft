use anyhow::{bail, Context, Result};
use candle_core::Device;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;
use witchcraft::Embedder;

const DEFAULT_BATCH_SIZE: usize = 64;

fn main() -> Result<()> {
    let input = input_path()?;
    let assets = env::var("WITCHCRAFT_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("assets"));
    let batch_size = env::var("WITCHCRAFT_BENCH_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_BATCH_SIZE);
    let device = device_from_env()?;

    eprintln!("input: {}", input.display());
    eprintln!("assets: {}", assets.display());
    eprintln!("batch-size: {batch_size}");
    eprintln!("device: {device:?}");

    let embedder = Embedder::new(&device, &assets).context("load Witchcraft embedder")?;
    let texts = read_texts(&input)?;
    let started = Instant::now();
    let mut total_docs = 0usize;
    let mut total_vectors = 0usize;
    let mut total_raw_bytes = 0usize;

    for (batch_number, batch) in texts.chunks(batch_size).enumerate() {
        let batch_started = Instant::now();
        let outputs = embedder
            .embed_batch_with_gate_scores_and_tokens(batch)
            .context("embed batch")?;
        let mut vectors = 0usize;
        let mut raw_bytes = 0usize;
        for output in &outputs {
            let (_batch, rows, dim) = output.embeddings.dims3()?;
            vectors += rows;
            raw_bytes += rows * dim * std::mem::size_of::<f32>();
        }
        total_docs += outputs.len();
        total_vectors += vectors;
        total_raw_bytes += raw_bytes;

        let elapsed = started.elapsed().as_secs_f64();
        let batch_s = batch_started.elapsed().as_secs_f64();
        eprintln!(
            "batch {}: docs={} vectors={} batch_s={:.3} total_docs={} docs_per_s={:.2} vectors_per_s={:.2} raw_mib={:.2}",
            batch_number + 1,
            outputs.len(),
            vectors,
            batch_s,
            total_docs,
            total_docs as f64 / elapsed,
            total_vectors as f64 / elapsed,
            raw_bytes as f64 / 1024.0 / 1024.0
        );
    }

    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "done: docs={} vectors={} elapsed_s={:.3} docs_per_s={:.2} vectors_per_s={:.2} raw_embedding_mib={:.2}",
        total_docs,
        total_vectors,
        elapsed,
        total_docs as f64 / elapsed,
        total_vectors as f64 / elapsed,
        total_raw_bytes as f64 / 1024.0 / 1024.0
    );
    Ok(())
}

fn input_path() -> Result<PathBuf> {
    if let Ok(path) = env::var("WITCHCRAFT_BENCH_INPUT") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("src/witchcraft/datasets/nfcorpus.tsv"))
}

fn read_texts(path: &PathBuf) -> Result<Vec<String>> {
    let reader = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut texts = Vec::new();
    for (line_number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_number + 1))?;
        let Some((_doc_id, text)) = line.split_once('\t') else {
            bail!("malformed TSV row at line {}", line_number + 1);
        };
        texts.push(text.to_string());
    }
    Ok(texts)
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
    bail!("this benchmark binary was not built with Candle CUDA support")
}
