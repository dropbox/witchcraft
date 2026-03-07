mod quantized_t5;
mod fast_ops;
#[cfg(feature = "ov")]
mod openvino_t5;
#[cfg(feature = "fbgemm")]
use warp::quantized_t5 as fbgemm_t5;

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct EmbeddingComparison {
    min_similarity: f32,
    max_similarity: f32,
    avg_similarity: f32,
    total_vectors: usize,
    filtered_vectors: usize,
}

fn load_tokenizer(assets: &PathBuf) -> Result<Tokenizer> {
    let bytes = std::fs::read(assets.join("tokenizer.json"))?;
    let tokenizer = Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(tokenizer)
}

fn compute_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    // Compute cosine similarity between two vectors
    // Both should be [dim] shaped
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let norm_a = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let norm_b = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let similarity = dot / (norm_a * norm_b);
    Ok(similarity)
}

struct VectorDetail {
    position: usize,
    similarity: f32,
    norm_a: f32,
    norm_b: f32,
}

const MIN_NORM: f32 = 1.0;

fn compare_embeddings_detailed(emb1: &Tensor, emb2: &Tensor) -> Result<(EmbeddingComparison, Vec<VectorDetail>)> {
    let emb1 = emb1.squeeze(0)?;
    let emb2 = emb2.squeeze(0)?;
    let (seq_len, _dim) = emb1.dims2()?;

    let mut similarities = Vec::new();
    let mut details = Vec::new();
    let mut filtered = 0usize;

    for i in 0..seq_len {
        let v1 = emb1.get(i)?;
        let v2 = emb2.get(i)?;
        let norm_a = v1.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_b = v2.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        if norm_a < MIN_NORM || norm_b < MIN_NORM {
            filtered += 1;
            continue;
        }
        let sim = compute_similarity(&v1, &v2)?;
        similarities.push(sim);
        details.push(VectorDetail { position: i, similarity: sim, norm_a, norm_b });
    }

    if similarities.is_empty() {
        return Ok((EmbeddingComparison {
            min_similarity: 0.0,
            max_similarity: 0.0,
            avg_similarity: 0.0,
            total_vectors: 0,
            filtered_vectors: filtered,
        }, details));
    }

    let min_sim = similarities.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_sim = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let avg_sim = similarities.iter().sum::<f32>() / similarities.len() as f32;

    Ok((EmbeddingComparison {
        min_similarity: min_sim,
        max_similarity: max_sim,
        avg_similarity: avg_sim,
        total_vectors: similarities.len(),
        filtered_vectors: filtered,
    }, details))
}

fn compare_embeddings(emb1: &Tensor, emb2: &Tensor) -> Result<EmbeddingComparison> {
    let (comp, _) = compare_embeddings_detailed(emb1, emb2)?;
    Ok(comp)
}

fn compare_quantized_backends(
    assets: &PathBuf,
    tokenizer: &Tokenizer,
    tsv_path: &PathBuf,
) -> Result<()> {
    eprintln!("\n=== Comparing Quantized T5 Backend (self-check) ===");
    eprintln!("This compares the same backend with itself to verify comparison logic");

    let device = Device::Cpu;

    // Load config
    let cfg_bytes = std::fs::read(assets.join("config.json"))?;
    let config: quantized_t5::Config = serde_json::from_slice(&cfg_bytes)?;

    // Load model once (we'll use it twice to verify comparison works)
    let model_path = assets.join("xtr.gguf");
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &model_path,
        &device,
    )?;
    let model = quantized_t5::T5EncoderModel::load(vb, &config)?;

    eprintln!("Model loaded, reading dataset from {}", tsv_path.display());

    // Read TSV dataset
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(tsv_path)?;

    let mut all_min_sims = Vec::new();
    let mut all_avg_sims = Vec::new();
    let mut doc_count = 0;

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            continue;
        }

        let _doc_id = &record[0];
        let text = &record[1];

        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?;
        let ids = encoding.get_ids();

        if ids.is_empty() || ids.len() > 512 {
            continue;
        }

        let input = Tensor::new(ids, &device)?.unsqueeze(0)?;

        let emb1 = model.forward(&input)?;
        let emb2 = model.forward(&input)?;

        let comparison = compare_embeddings(&emb1, &emb2)?;

        all_min_sims.push(comparison.min_similarity);
        all_avg_sims.push(comparison.avg_similarity);

        doc_count += 1;

        if doc_count % 10 == 0 {
            eprintln!(
                "  Doc {}: {} tokens, min_sim={:.6}, avg_sim={:.6}",
                doc_count,
                comparison.total_vectors,
                comparison.min_similarity,
                comparison.avg_similarity
            );
        }

        if doc_count >= 50 {
            break; // Limit to first 50 documents for testing
        }
    }

    // Overall statistics
    let overall_min = all_min_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let overall_avg_min = all_min_sims.iter().sum::<f32>() / all_min_sims.len() as f32;
    let overall_avg = all_avg_sims.iter().sum::<f32>() / all_avg_sims.len() as f32;

    eprintln!("\n=== Results (Quantized vs Quantized - should be ~1.0) ===");
    eprintln!("Documents processed: {}", doc_count);
    eprintln!("Minimum similarity (worst vector): {:.6}", overall_min);
    eprintln!("Average of minimum similarities: {:.6}", overall_avg_min);
    eprintln!("Average of average similarities: {:.6}", overall_avg);

    if overall_avg < 0.9999 {
        eprintln!("\n⚠️  WARNING: Same backend should produce identical results (sim ~1.0)");
    } else {
        eprintln!("\n✓ Self-comparison passed (embeddings are deterministic)");
    }

    Ok(())
}

#[cfg(feature = "ov")]
fn compare_quantized_vs_openvino(
    assets: &PathBuf,
    tokenizer: &Tokenizer,
    tsv_path: &PathBuf,
) -> Result<()> {
    eprintln!("\n=== Comparing Quantized vs OpenVINO INT4 ===");

    let device = Device::Cpu;

    // Load quantized model
    let cfg_bytes = std::fs::read(assets.join("config.json"))?;
    let config_q: quantized_t5::Config = serde_json::from_slice(&cfg_bytes)?;
    let model_path = assets.join("xtr.gguf");
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &model_path,
        &device,
    )?;
    let quantized_model = quantized_t5::T5EncoderModel::load(vb, &config_q)?;

    // Load OpenVINO model
    let (builder, _) = openvino_t5::T5ModelBuilder::load(assets)?;
    let ov_model = builder.build_encoder(&device, assets)?;

    eprintln!("Both models loaded, reading dataset from {}", tsv_path.display());

    // Read TSV dataset
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(tsv_path)?;

    let mut all_min_sims = Vec::new();
    let mut all_avg_sims = Vec::new();
    let mut doc_count = 0;
    let mut total_filtered = 0usize;
    let mut total_kept = 0usize;
    let outlier_threshold = 0.92;

    // Collect all outliers across docs: (doc_id, token_text, position, sim, norm_q, norm_ov, seq_len)
    let mut outliers: Vec<(String, String, usize, f32, f32, f32, usize)> = Vec::new();
    // Histogram of similarities in 0.05-wide buckets from 0.70 to 1.00
    let mut sim_histogram = [0u64; 7]; // [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            continue;
        }

        let doc_id = &record[0];
        let text = &record[1];

        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?;
        let ids = encoding.get_ids();
        let tokens = encoding.get_tokens();

        if ids.is_empty() || ids.len() > 512 {
            continue;
        }

        let input = Tensor::new(ids, &device)?.unsqueeze(0)?;

        let emb_q = quantized_model.forward(&input)?;
        let emb_ov = ov_model.forward(&input)?;

        let (comparison, details) = compare_embeddings_detailed(&emb_q, &emb_ov)?;

        for d in &details {
            // Histogram
            let bucket = ((d.similarity - 0.70) / 0.05).floor() as i32;
            let bucket = bucket.clamp(0, 6) as usize;
            if d.similarity >= 0.70 {
                sim_histogram[bucket] += 1;
            }

            if d.similarity < outlier_threshold {
                let tok = tokens.get(d.position).cloned().unwrap_or_else(|| "?".into());
                outliers.push((
                    doc_id.to_string(), tok, d.position, d.similarity,
                    d.norm_a, d.norm_b, details.len(),
                ));
            }
        }

        total_filtered += comparison.filtered_vectors;
        total_kept += comparison.total_vectors;
        all_min_sims.push(comparison.min_similarity);
        all_avg_sims.push(comparison.avg_similarity);

        doc_count += 1;

        if doc_count % 10 == 0 {
            eprintln!(
                "  Doc {} ({}): {} tokens ({} filtered), min_sim={:.6}, avg_sim={:.6}",
                doc_count,
                doc_id,
                comparison.total_vectors,
                comparison.filtered_vectors,
                comparison.min_similarity,
                comparison.avg_similarity
            );
        }

        if doc_count >= 50 {
            break;
        }
    }

    let overall_min = all_min_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let overall_avg_min = all_min_sims.iter().sum::<f32>() / all_min_sims.len() as f32;
    let overall_avg = all_avg_sims.iter().sum::<f32>() / all_avg_sims.len() as f32;

    eprintln!("\n=== Results (Quantized vs OpenVINO INT4) ===");
    eprintln!("Documents processed: {}", doc_count);
    eprintln!("Vectors compared: {} (filtered {} low-norm tokens, {:.1}%)",
        total_kept, total_filtered,
        100.0 * total_filtered as f64 / (total_kept + total_filtered) as f64);
    eprintln!("Minimum similarity across all vectors: {:.6}", overall_min);
    eprintln!("Average of minimum similarities per doc: {:.6}", overall_avg_min);
    eprintln!("Average of average similarities per doc: {:.6}", overall_avg);

    // Similarity histogram
    eprintln!("\n--- Similarity distribution ---");
    let buckets = ["0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00", "1.00+"];
    for (i, label) in buckets.iter().enumerate() {
        if sim_histogram[i] > 0 {
            eprintln!("  {}: {} vectors", label, sim_histogram[i]);
        }
    }

    // Outlier details
    if outliers.is_empty() {
        eprintln!("\nNo outliers below {:.2} threshold", outlier_threshold);
    } else {
        outliers.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
        eprintln!("\n--- Worst {} outliers (sim < {:.2}) ---", outliers.len().min(30), outlier_threshold);
        eprintln!("{:<8} {:<6} {:<20} {:<10} {:<10} {:<10} {:<6}",
            "doc_id", "pos", "token", "sim", "norm_q", "norm_ov", "seqlen");
        for o in outliers.iter().take(30) {
            eprintln!("{:<8} {:<6} {:<20} {:<10.6} {:<10.4} {:<10.4} {:<6}",
                o.0, o.2, o.1, o.3, o.4, o.5, o.6);
        }
    }

    if overall_min < 0.90 {
        eprintln!("\n⚠️  WARNING: Low minimum similarity (<0.90) detected");
    } else if overall_min < 0.95 {
        eprintln!("\n✓ Acceptable similarity (>0.90) - some quantization differences expected");
    } else {
        eprintln!("\n✓ Excellent similarity (>0.95) between backends");
    }

    Ok(())
}

#[cfg(feature = "fbgemm")]
fn compare_vanilla_vs_fbgemm(
    assets: &PathBuf,
    tokenizer: &Tokenizer,
    tsv_path: &PathBuf,
) -> Result<()> {
    eprintln!("\n=== Comparing Vanilla Candle QMatMul vs fbgemm-rs hybrid-dequant ===");

    let device = Device::Cpu;

    // Load config
    let cfg_bytes = std::fs::read(assets.join("config.json"))?;

    // Load vanilla model (this crate's quantized_t5 — standard candle QMatMul)
    let config_v: quantized_t5::Config = serde_json::from_slice(&cfg_bytes)?;
    let vb_v = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &assets.join("xtr.gguf"),
        &device,
    )?;
    let vanilla_model = quantized_t5::T5EncoderModel::load(vb_v, &config_v)?;

    // Load fbgemm model (warp's quantized_t5 compiled with hybrid-dequant)
    let config_f: fbgemm_t5::Config = serde_json::from_slice(&cfg_bytes)?;
    let vb_f = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
        &assets.join("xtr.gguf"),
        &device,
    )?;
    let fbgemm_model = fbgemm_t5::T5EncoderModel::load(vb_f, &config_f)?;

    eprintln!("Both models loaded, reading dataset from {}", tsv_path.display());

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(tsv_path)?;

    let mut all_min_sims = Vec::new();
    let mut all_avg_sims = Vec::new();
    let mut doc_count = 0;

    for result in rdr.records() {
        let record = result?;
        if record.len() < 2 {
            continue;
        }

        let text = &record[1];

        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?;
        let ids = encoding.get_ids();

        if ids.is_empty() || ids.len() > 512 {
            continue;
        }

        let input = Tensor::new(ids, &device)?.unsqueeze(0)?;

        let emb_vanilla = vanilla_model.forward(&input)?;
        let emb_fbgemm = fbgemm_model.forward(&input)?;

        let comparison = compare_embeddings(&emb_vanilla, &emb_fbgemm)?;

        all_min_sims.push(comparison.min_similarity);
        all_avg_sims.push(comparison.avg_similarity);

        doc_count += 1;

        if doc_count % 10 == 0 {
            eprintln!(
                "  Doc {}: {} tokens, min_sim={:.6}, avg_sim={:.6}",
                doc_count,
                comparison.total_vectors,
                comparison.min_similarity,
                comparison.avg_similarity
            );
        }

        if doc_count >= 50 {
            break;
        }
    }

    let overall_min = all_min_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let overall_avg_min = all_min_sims.iter().sum::<f32>() / all_min_sims.len() as f32;
    let overall_avg = all_avg_sims.iter().sum::<f32>() / all_avg_sims.len() as f32;

    eprintln!("\n=== Results (Vanilla QMatMul vs fbgemm-rs hybrid-dequant) ===");
    eprintln!("Documents processed: {}", doc_count);
    eprintln!("Minimum similarity across all vectors: {:.6}", overall_min);
    eprintln!("Average of minimum similarities per doc: {:.6}", overall_avg_min);
    eprintln!("Average of average similarities per doc: {:.6}", overall_avg);

    if overall_min < 0.99 {
        eprintln!("\n  WARNING: Low similarity (<0.99) — fbgemm-rs path may diverge");
    } else {
        eprintln!("\n  Excellent similarity (>0.99) — fbgemm-rs matches vanilla");
    }

    Ok(())
}

fn main() -> Result<()> {
    let assets = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "assets".into()));
    let tsv_path = PathBuf::from(
        std::env::args()
            .nth(2)
            .unwrap_or_else(|| "datasets/nfcorpus.tsv".into()),
    );

    eprintln!("Assets directory: {}", assets.display());
    eprintln!("Dataset: {}", tsv_path.display());

    if !tsv_path.exists() {
        eprintln!("Error: Dataset not found at {}", tsv_path.display());
        eprintln!("Usage: compare-backends [ASSETS_DIR] [TSV_FILE]");
        return Ok(());
    }

    let tokenizer = load_tokenizer(&assets)?;

    // Always run self-check first
    compare_quantized_backends(&assets, &tokenizer, &tsv_path)?;

    // Compare with OpenVINO if available
    #[cfg(feature = "ov")]
    {
        let xml_path = assets.join("xtr-ov-int4.xml");
        if xml_path.exists() {
            compare_quantized_vs_openvino(&assets, &tokenizer, &tsv_path)?;
        } else {
            eprintln!("\nSkipping OpenVINO comparison (model files not found)");
            eprintln!("Run: python quantize_int4.py");
        }
    }

    #[cfg(not(feature = "ov"))]
    eprintln!("\nOpenVINO comparison not available (build with --features ov)");

    #[cfg(feature = "fbgemm")]
    compare_vanilla_vs_fbgemm(&assets, &tokenizer, &tsv_path)?;

    #[cfg(not(feature = "fbgemm"))]
    eprintln!("\nfbgemm-rs comparison not available (build with --features fbgemm)");

    Ok(())
}
