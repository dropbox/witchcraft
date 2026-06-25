use super::t5_encoder;
use anyhow::Result;
use candle_core::{Device, Tensor};
use log::debug;
use tokenizers::Tokenizer;

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}

fn model_forward_with_gate(
    model: &t5_encoder::T5EncoderModel,
    input: &Tensor,
) -> Result<(Tensor, Option<Tensor>)> {
    #[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
    {
        Ok(model.forward_with_gate(input)?)
    }
    #[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
    {
        Ok((model.forward(input)?, None))
    }
}

pub struct Embedder {
    tokenizer: Tokenizer,
    model: t5_encoder::T5EncoderModel,
}

pub struct EmbeddingOutput {
    pub embeddings: Tensor,
    pub offsets: Vec<(usize, usize)>,
    pub gate_scores: Option<Vec<f32>>,
    pub tokens: Vec<String>,
}

impl Embedder {
    pub fn new(device: &Device, assets: &std::path::Path) -> Result<Self> {
        let (builder, tokenizer) = t5_encoder::T5ModelBuilder::load(assets)?;
        let model = builder.build_encoder(device, assets)?;
        Ok(Self { tokenizer, model })
    }

    pub fn embed(&self, text: &str) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let output = self.embed_inner(text, false)?;
        Ok((output.embeddings, output.offsets))
    }

    pub fn embed_with_gate_scores(
        &self,
        text: &str,
    ) -> Result<(Tensor, Vec<(usize, usize)>, Option<Vec<f32>>)> {
        let output = self.embed_inner(text, true)?;
        Ok((output.embeddings, output.offsets, output.gate_scores))
    }

    pub fn embed_with_gate_scores_and_tokens(&self, text: &str) -> Result<EmbeddingOutput> {
        self.embed_inner(text, true)
    }

    fn embed_inner(
        &self,
        text: &str,
        collect_gate_scores: bool,
    ) -> Result<EmbeddingOutput> {
        let now = std::time::Instant::now();
        let model = &self.model;
        let device = model.device();

        let max_len: usize = 2048;
        let stride: usize = 256;

        let encoding = self.tokenizer.encode(text, true).unwrap();
        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets().to_vec();
        let tokens = encoding.get_tokens().to_vec();

        let n_tokens = ids.len();
        let mut accum: Vec<Option<Tensor>> = vec![None; n_tokens];
        let mut gate_scores: Option<Vec<f32>> = None;
        let mut gate_counts: Option<Vec<u32>> = None;

        let mut start = 0;
        loop {
            let end = (start + max_len).min(n_tokens);
            let input = Tensor::new(&ids[start..end], device)?.unsqueeze(0)?;
            let (chunk, gates) = if collect_gate_scores {
                model_forward_with_gate(model, &input)?
            } else {
                (model.forward(&input)?, None)
            };
            let chunk = chunk.squeeze(0)?.to_device(&Device::Cpu)?;
            let gates = gates
                .map(|g| g.squeeze(0)?.to_device(&Device::Cpu)?.to_vec1::<f32>())
                .transpose()?;

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
                if let Some(gates) = gates.as_ref() {
                    if gate_scores.is_none() {
                        gate_scores = Some(vec![0.0; n_tokens]);
                        gate_counts = Some(vec![0; n_tokens]);
                    }
                    if let (Some(scores), Some(counts)) =
                        (gate_scores.as_mut(), gate_counts.as_mut())
                    {
                        scores[global_idx] += gates[i];
                        counts[global_idx] += 1;
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
        let mut filtered_tokens = Vec::with_capacity(tokens.len());
        let mut filtered_scores = gate_scores.as_ref().map(|_| Vec::with_capacity(token_embs.len()));
        for (idx, (emb, offset)) in token_embs.into_iter().zip(offsets.into_iter()).enumerate() {
            let norm = emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm >= MIN_NORM {
                filtered_embs.push(emb);
                filtered_offsets.push(offset);
                filtered_tokens.push(tokens[idx].clone());
                if let Some(out) = filtered_scores.as_mut() {
                    let scores = gate_scores.as_ref().expect("gate scores missing");
                    let counts = gate_counts.as_ref().expect("gate counts missing");
                    let count = counts[idx].max(1) as f32;
                    out.push(scores[idx] / count);
                }
            }
        }
        if filtered_embs.is_empty() {
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
        Ok(EmbeddingOutput {
            embeddings: normalized,
            offsets: filtered_offsets,
            gate_scores: filtered_scores,
            tokens: filtered_tokens,
        })
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
