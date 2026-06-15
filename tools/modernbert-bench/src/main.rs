use anyhow::Result;
use candle_core::Tensor;
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
        Err(_) => Ok(vec![32, 64, 128, 256, 512]),
    }
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

    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    let ids = make_input(&tokenizer, base, 32);
    let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;
    let _ = model.forward(&input)?;

    let iters = std::env::var("WARP_BENCH_ITERS")
        .ok()
        .map(|iters| iters.parse::<usize>())
        .transpose()?
        .unwrap_or(7);
    anyhow::ensure!(iters > 0, "WARP_BENCH_ITERS must be greater than zero");

    for n in bench_sizes()? {
        let ids = make_input(&tokenizer, base, n);
        let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;

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
            "{BACKEND}: {:>4} tokens -> median {:>7.1?}  ({:.0} tok/s)",
            n,
            median,
            n as f64 / median.as_secs_f64()
        );
    }

    Ok(())
}
