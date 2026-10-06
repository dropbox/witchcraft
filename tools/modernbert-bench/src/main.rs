use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(all(feature = "modernbert", feature = "modernbert-quantized"))]
compile_error!("Enable only one ModernBERT backend feature");

#[cfg(feature = "modernbert")]
use witchcraft::modernbert as model_mod;
#[cfg(feature = "modernbert-quantized")]
use witchcraft::quantized_modernbert as model_mod;

#[cfg(feature = "modernbert")]
const BACKEND: &str = "modernbert-f32";
#[cfg(feature = "modernbert-quantized")]
const BACKEND: &str = "modernbert-quantized";

fn make_input(tokenizer: &tokenizers::Tokenizer, base: &str, min_tokens: usize) -> Vec<u32> {
    let enc = tokenizer.encode(base, true).unwrap();
    let base_ids = enc.get_ids();
    assert!(!base_ids.is_empty(), "base text tokenized to zero ids");

    let mut ids = Vec::with_capacity(min_tokens);
    while ids.len() < min_tokens {
        ids.extend_from_slice(base_ids);
    }
    ids.truncate(min_tokens);
    ids
}

fn bench_sizes() -> Result<Vec<usize>> {
    match std::env::var("WARP_BENCH_SIZES") {
        Ok(sizes) => sizes
            .split(',')
            .map(|size| {
                size.trim()
                    .parse::<usize>()
                    .map_err(|e| anyhow::anyhow!("invalid WARP_BENCH_SIZES entry {size:?}: {e}"))
            })
            .collect(),
        Err(_) => Ok(vec![32, 64, 128, 256, 512, 1024, 2048]),
    }
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn check_mixed_length_batch(
    model: &model_mod::T5EncoderModel,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    base: &str,
) -> Result<()> {
    let short = make_input(tokenizer, base, 37);
    let long = make_input(tokenizer, base, 61);
    let mut padded = short.clone();
    padded.resize(long.len(), 0);
    padded.extend_from_slice(&long);
    let input = Tensor::from_vec(padded, (2, long.len()), device)?;
    let (batched, _) = model.forward_with_gate_for_lengths(&input, &[short.len(), long.len()])?;

    let short_input = Tensor::new(short.as_slice(), device)?.unsqueeze(0)?;
    let long_input = Tensor::new(long.as_slice(), device)?.unsqueeze(0)?;
    let expected = [model.forward(&short_input)?, model.forward(&long_input)?];
    let mut max_delta = 0.0f32;
    let mut document_deltas = Vec::with_capacity(2);
    for (index, (length, expected)) in [short.len(), long.len()]
        .into_iter()
        .zip(expected)
        .enumerate()
    {
        let actual = batched.narrow(0, index, 1)?.narrow(1, 0, length)?;
        let mut document_delta = 0.0f32;
        for (actual, expected) in values(&actual)?.into_iter().zip(values(&expected)?) {
            document_delta = document_delta.max((actual - expected).abs());
        }
        document_deltas.push(document_delta);
        max_delta = max_delta.max(document_delta);
    }
    eprintln!("mixed-length document deltas: {document_deltas:?}");
    anyhow::ensure!(
        max_delta <= 1e-3,
        "mixed-length batch mismatch: max abs delta {max_delta}"
    );
    eprintln!("mixed-length batch check: max abs delta {max_delta:.6}");
    Ok(())
}

fn main() -> Result<()> {
    let assets = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "assets".into()));
    eprintln!("assets dir: {}", assets.display());
    eprintln!("backend: {BACKEND}");

    let device = witchcraft::make_device();
    let t0 = Instant::now();
    let (builder, tokenizer) = model_mod::T5ModelBuilder::load(&assets)?;
    let model = builder.build_encoder(&device, &assets)?;
    eprintln!("{BACKEND}: model loaded in {:.0?}", t0.elapsed());

    let batch_size = std::env::var("WARP_BENCH_BATCH_SIZE")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    anyhow::ensure!(
        batch_size > 0,
        "WARP_BENCH_BATCH_SIZE must be greater than zero"
    );
    eprintln!("batch size: {batch_size}");

    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    if batch_size > 1 {
        check_mixed_length_batch(&model, &tokenizer, &device, base)?;
    }

    let ids = make_input(&tokenizer, base, 32);
    let input = Tensor::from_vec(ids.repeat(batch_size), (batch_size, ids.len()), &device)?;
    let _ = model.forward(&input)?;

    let iters = std::env::var("WARP_BENCH_ITERS")
        .ok()
        .map(|iters| iters.parse::<usize>())
        .transpose()?
        .unwrap_or(7);
    anyhow::ensure!(iters > 0, "WARP_BENCH_ITERS must be greater than zero");

    for n in bench_sizes()? {
        let ids = make_input(&tokenizer, base, n);
        let input = Tensor::from_vec(ids.repeat(batch_size), (batch_size, ids.len()), &device)?;

        let _ = model.forward(&input)?;

        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let out = model.forward(&input)?;
            let _ = out.dims3()?;
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[times.len() / 2];
        eprintln!(
            "{BACKEND}: batch {batch_size:>2} x {n:>4} tokens -> median {:>7.1?}  ({:.0} tok/s, {:.1} docs/s)",
            median,
            (batch_size * n) as f64 / median.as_secs_f64(),
            batch_size as f64 / median.as_secs_f64(),
        );
    }

    Ok(())
}
