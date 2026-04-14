use super::t5_encoder;
use anyhow::Result;
use candle_core::{Device, Tensor};
use log::debug;
use tokenizers::Tokenizer;

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
pub struct Embedder {
    tokenizer: Tokenizer,
    model: t5_encoder::T5EncoderModel,
}

impl Embedder {
    pub fn new(device: &Device, assets: &std::path::Path) -> Result<Self> {
        let (builder, tokenizer) = t5_encoder::T5ModelBuilder::load(assets)?;
        let model = builder.build_encoder(device, assets)?;
        Ok(Self { tokenizer, model })
    }

    pub fn embed(&self, text: &str) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let now = std::time::Instant::now();
        let model = &self.model;
        let device = model.device();

        let max_len: usize = 2048;
        let stride: usize = 256;

        let encoding = self.tokenizer.encode(text, true).unwrap();
        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets().to_vec();

        let n_tokens = ids.len();
        let mut accum: Vec<Option<Tensor>> = vec![None; n_tokens];

        let mut start = 0;
        loop {
            let end = (start + max_len).min(n_tokens);
            let input = Tensor::new(&ids[start..end], device)?.unsqueeze(0)?;
            let fwd_t = std::time::Instant::now();
            let chunk = model.forward(&input)?.squeeze(0)?.to_device(&Device::Cpu)?;
            log::info!("forward: {} tokens in {}ms", end - start, fwd_t.elapsed().as_millis());

            let (m, _n) = chunk.dims2()?;
            for i in 0..m {
                let global_idx = start + i;
                if global_idx >= n_tokens {
                    break;
                }
                let emb = chunk.get(i)?;
                match &accum[global_idx] {
                    None => accum[global_idx] = Some(emb.clone()),
                    Some(prev) => {
                        let sum = (prev + &emb)?;
                        accum[global_idx] = Some(sum);
                    }
                }
            }

            if end == n_tokens {
                break;
            }
            start = end.saturating_sub(stride); // overlap window
        }

        let token_embs: Vec<Tensor> = accum
            .into_iter()
            .enumerate()
            .map(|(i, maybe_t)| {
                maybe_t.unwrap_or_else(|| {
                    panic!("Missing embedding for token {} — check stride settings", i)
                })
            })
            .collect();

        // Filter out low-signal tokens (e.g. </s>, padding) whose pre-normalization
        // norm is near zero. These carry no directional information and become
        // amplified noise after L2 normalization. Content tokens have norm ~5.0+;
        // </s> tokens have norm ~0.27.
        const MIN_NORM: f32 = 1.0;
        let mut filtered_embs = Vec::with_capacity(token_embs.len());
        let mut filtered_offsets = Vec::with_capacity(offsets.len());
        let mut norms_debug = Vec::new();
        for (emb, offset) in token_embs.into_iter().zip(offsets.into_iter()) {
            let norm = emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            norms_debug.push(norm);
            if norm >= MIN_NORM {
                filtered_embs.push(emb);
                filtered_offsets.push(offset);
            }
        }
        if filtered_embs.is_empty() {
            let first_few: Vec<String> = norms_debug.iter().take(5).map(|n| format!("{n:.4}")).collect();
            log::warn!("all norms below threshold: n_tokens={} norms={:?} text_len={} text={:.120}", norms_debug.len(), first_few, text.len(), text);
            anyhow::bail!("all token embeddings below minimum norm threshold");
        }

        let matrix = Tensor::stack(&filtered_embs, 0)?.unsqueeze(0)?;
        let normalized = normalize_l2(&matrix)?;
        debug!(
            "embedder took {} ms, kept {}/{} tokens.",
            now.elapsed().as_millis(),
            filtered_embs.len(),
            n_tokens,
        );
        Ok((normalized, filtered_offsets))
    }

    /*
    pub fn embed(self: &Self, text: &str) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let now = std::time::Instant::now();
        let enc = self.tokenizer.encode(text, true).map_err(E::msg).unwrap();
        let offsets = enc.get_offsets().to_vec();
        let tokens = enc.get_ids().to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        debug!("embedder took {} ms.", now.elapsed().as_millis());
        let normalized = normalize_l2(&embeddings)?;
        Ok((normalized, offsets))
    }
    */
}
