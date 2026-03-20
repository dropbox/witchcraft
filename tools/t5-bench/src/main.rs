use anyhow::Result;
use candle_core::Tensor;
use std::path::PathBuf;
use std::time::Instant;
use warp::quantized_t5;

fn load_tokenizer(assets: &PathBuf) -> Result<tokenizers::Tokenizer> {
    let bytes = std::fs::read(assets.join("tokenizer.json"))?;
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
    let device = warp::make_device();

    let cfg_bytes = std::fs::read(assets.join("config.json"))?;
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

#[cfg(feature = "ov")]
fn bench_openvino(assets: &PathBuf, tokenizer: &tokenizers::Tokenizer, sizes: &[usize]) -> Result<()> {
    use openvino::{Core, DeviceType, Shape};

    #[cfg(target_os = "windows")]
    {
        let dll_file = assets.join("openvino_c.dll");
        if dll_file.exists() {
            // Get absolute path
            let abs_assets = assets.canonicalize().unwrap_or_else(|_| assets.clone());
            if let Some(assets_str) = abs_assets.to_str() {
                // Add to PATH environment variable at the front
                if let Ok(current_path) = std::env::var("PATH") {
                    let new_path = format!("{};{}", assets_str, current_path);
                    unsafe {
                        std::env::set_var("PATH", new_path);
                    }
                    eprintln!("[INFO] Added assets directory to PATH for OpenVINO DLLs: {}", assets_str);
                }
            }
        }
    }


    let base = "Bananas are berries but strawberries are not. Octopuses have three hearts and blue blood. A day on Venus is longer than a year on Venus. There are more trees on Earth than stars in the Milky Way.";

    // For each size, create a fresh OV model+infer request to avoid shape-change issues.
    // Retry on failure.
    for &n in sizes {
        let ids = make_input(tokenizer, base, n);
        let padded_len = n.next_power_of_two().max(64);

        let mut success = false;
        for attempt in 0..3 {
            let result: Result<()> = (|| {
                let mut core = Core::new().map_err(|e| anyhow::anyhow!("ov core: {e:?}"))?;
                let xml_path = assets.join("xtr-ov-int4.xml");
                let bin_path = assets.join("xtr-ov-int4.bin");

                let model = core.read_model_from_file(
                    xml_path.to_str().unwrap(),
                    bin_path.to_str().unwrap(),
                ).map_err(|e| anyhow::anyhow!("read model: {e:?}"))?;

                let mut compiled = core.compile_model(&model, DeviceType::CPU)
                    .map_err(|e| anyhow::anyhow!("compile: {e:?}"))?;
                let mut infer = compiled.create_infer_request()
                    .map_err(|e| anyhow::anyhow!("infer request: {e:?}"))?;

                let mut run_once = |ids: &[u32]| -> Result<()> {
                    let mut padded: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
                    padded.resize(padded_len, 0);
                    let ov_shape = Shape::new(&[1i64, padded_len as i64])
                        .map_err(|e| anyhow::anyhow!("shape: {e:?}"))?;
                    let mut ov_input = openvino::Tensor::new(openvino::ElementType::I64, &ov_shape)
                        .map_err(|e| anyhow::anyhow!("tensor: {e:?}"))?;
                    ov_input.get_data_mut::<i64>()
                        .map_err(|e| anyhow::anyhow!("data: {e:?}"))?
                        .copy_from_slice(&padded);
                    infer.set_input_tensor(&ov_input)
                        .map_err(|e| anyhow::anyhow!("set input: {e:?}"))?;
                    infer.infer()
                        .map_err(|e| anyhow::anyhow!("infer: {e:?}"))?;
                    let _output = infer.get_output_tensor()
                        .map_err(|e| anyhow::anyhow!("output: {e:?}"))?;
                    Ok(())
                };

                // Warmup (3 iters, same shape)
                for _ in 0..3 {
                    run_once(&ids)?;
                }

                let mut times = Vec::new();
                for _ in 0..7 {
                    let t = Instant::now();
                    run_once(&ids)?;
                    times.push(t.elapsed());
                }
                times.sort();
                let median = times[3];
                eprintln!(
                    "openvino: {:>4} tokens (padded {:>4}) -> median {:>7.1?}  ({:.0} tok/s)",
                    n, padded_len, median, n as f64 / median.as_secs_f64()
                );
                Ok(())
            })();

            match result {
                Ok(()) => { success = true; break; }
                Err(e) => {
                    eprintln!("openvino: {} tokens attempt {} failed: {e}", n, attempt + 1);
                }
            }
        }
        if !success {
            eprintln!("openvino: {} tokens FAILED after 3 attempts", n);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let assets = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "assets".into()));
    eprintln!("assets dir: {}", assets.display());

    let tokenizer = load_tokenizer(&assets)?;

    let sizes = vec![32, 64, 128, 256, 512];
    let only = std::env::args().nth(2);

    if only.as_deref() != Some("ov") {
        eprintln!("\n=== Candle (Q4K -> F32 on CPU) ===");
        if let Err(e) = bench_candle(&assets, &tokenizer, &sizes) {
            eprintln!("candle error: {e}");
        }
    }

    #[cfg(feature = "ov")]
    if only.as_deref() != Some("candle") {
        eprintln!("\n=== OpenVINO (INT4) ===");
        if let Err(e) = bench_openvino(&assets, &tokenizer, &sizes) {
            eprintln!("openvino error: {e}");
        }
    }

    Ok(())
}
