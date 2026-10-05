use anyhow::Result;
use candle_core::Tensor;
use std::path::PathBuf;
use std::time::Instant;
use witchcraft::quantized_t5;

fn load_tokenizer(assets: &PathBuf) -> Result<tokenizers::Tokenizer> {
    let bytes = std::fs::read(assets.join("xtr-tokenizer.json"))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(tokenizer)
}

/// Repeat text to reach at least `min_tokens` after tokenization.
fn make_input(tokenizer: &tokenizers::Tokenizer, base: &str, min_tokens: usize) -> Vec<u32> {
    let mut text = base.to_string();
    loop {
        let enc = tokenizer.encode(text.as_str(), true).unwrap();
        if enc.get_ids().len() >= min_tokens {
            return enc.get_ids()[..min_tokens].to_vec();
        }
        text.push(' ');
        text.push_str(base);
    }
}

fn bench_candle(assets: &PathBuf, tokenizer: &tokenizers::Tokenizer, sizes: &[usize]) -> Result<()> {
    let device = witchcraft::make_device();
    eprintln!("candle: device {:?}", device.location());

    let cfg_bytes = std::fs::read(assets.join("xtr-config.json"))?;
    let config: quantized_t5::Config = serde_json::from_slice(&cfg_bytes)?;

    let t0 = Instant::now();
    let model_path = assets.join("xtr.gguf");
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &model_path,
        &device,
    )?;
    let model = quantized_t5::T5EncoderModel::load(vb, &config)?;
    eprintln!("candle: model loaded in {:.0?} (using mmap)", t0.elapsed());

    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    // Warmup
    let ids = make_input(tokenizer, base, 32);
    let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;
    let _ = model.forward(&input)?;

    for &n in sizes {
        let ids = make_input(tokenizer, base, n);
        let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;

        // Warmup this size
        let _ = model.forward(&input)?;

        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            let out = model.forward(&input)?;
            let _ = out.dims3()?;
            times.push(t.elapsed());
        }
        times.sort();
        let median = times[3];
        eprintln!(
            "candle:   {:>4} tokens -> median {:>7.1?}  ({:.0} tok/s)",
            n, median, n as f64 / median.as_secs_f64()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let assets = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "assets".into()));
    eprintln!("assets dir: {}", assets.display());

    let tokenizer = load_tokenizer(&assets)?;

    let sizes = vec![32, 64, 128, 256, 512];
    eprintln!("\n=== Candle (Q4K -> F32) ===");
    if let Err(e) = bench_candle(&assets, &tokenizer, &sizes) {
        eprintln!("candle error: {e}");
    }

    Ok(())
}
