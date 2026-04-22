use super::t5_encoder;
use anyhow::Result;
use candle_core::{Device, Tensor};
use log::debug;
use tokenizers::Tokenizer;

const MAX_TOKENS: usize = 2048;

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
        self.embed_batch(&[text])?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embed_batch returned empty"))
    }

    /// Embed multiple texts. Short texts are padded and batched together for
    /// throughput; texts longer than 2048 tokens use a windowed stride path.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<(Tensor, Vec<(usize, usize)>)>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let now = std::time::Instant::now();
        let max_len = MAX_TOKENS;
        let device = self.model.device();

        let encodings: Vec<_> = texts
            .iter()
            .map(|t| self.tokenizer.encode(*t, true).unwrap())
            .collect();

        let mut long_indices = vec![];
        let mut short_indices = vec![];
        for (i, enc) in encodings.iter().enumerate() {
            if enc.get_ids().len() > max_len {
                long_indices.push(i);
            } else {
                short_indices.push(i);
            }
        }

        let mut results: Vec<(usize, (Tensor, Vec<(usize, usize)>))> =
            Vec::with_capacity(texts.len());

        // Long texts: windowed stride path
        for &i in &long_indices {
            let enc = &encodings[i];
            results.push((i, self.embed_windowed(enc.get_ids(), enc.get_offsets())?));
        }

        // Short texts: batch with padding
        if !short_indices.is_empty() {
            let mut sorted_short: Vec<usize> = short_indices.clone();
            sorted_short.sort_by_key(|&i| std::cmp::Reverse(encodings[i].get_ids().len()));

            let batch_size = 32;
            for chunk in sorted_short.chunks(batch_size) {
                let chunk_max = encodings[chunk[0]].get_ids().len();

                let mut batch_ids = Vec::with_capacity(chunk.len());
                for &i in chunk {
                    let ids = encodings[i].get_ids();
                    let mut padded = ids.to_vec();
                    padded.resize(chunk_max, 0); // T5 pad token = 0
                    batch_ids.push(Tensor::new(&padded[..], device)?);
                }
                let input = Tensor::stack(&batch_ids, 0)?;
                let output = self.model.forward(&input)?;

                for (batch_idx, &orig_idx) in chunk.iter().enumerate() {
                    let seq_len = encodings[orig_idx].get_ids().len();
                    let offsets = encodings[orig_idx].get_offsets().to_vec();
                    let seq_output = output.get(batch_idx)?.narrow(0, 0, seq_len)?;

                    let filtered = self.filter_and_normalize(&seq_output, &offsets)?;
                    results.push((orig_idx, filtered));
                }
            }
        }

        debug!(
            "embed_batch: {} texts ({} batched, {} windowed) in {} ms",
            texts.len(),
            short_indices.len(),
            long_indices.len(),
            now.elapsed().as_millis(),
        );

        results.sort_by_key(|(i, _)| *i);
        Ok(results.into_iter().map(|(_, r)| r).collect())
    }

    /// Windowed stride embedding for texts longer than 2048 tokens.
    /// Overlapping windows are averaged at shared positions.
    fn embed_windowed(
        &self,
        ids: &[u32],
        offsets: &[(usize, usize)],
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let max_len = MAX_TOKENS;
        let stride: usize = 256;
        let device = self.model.device();
        let n_tokens = ids.len();

        let mut accum: Vec<Option<Tensor>> = vec![None; n_tokens];

        let mut start = 0;
        loop {
            let end = (start + max_len).min(n_tokens);
            let input = Tensor::new(&ids[start..end], device)?.unsqueeze(0)?;
            let chunk = self.model.forward(&input)?.squeeze(0)?.to_device(&Device::Cpu)?;

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
            start = end.saturating_sub(stride);
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

        let stacked = Tensor::stack(&token_embs, 0)?;
        self.filter_and_normalize(&stacked, offsets)
    }

    /// Filter out low-signal tokens and L2-normalize.
    fn filter_and_normalize(
        &self,
        token_embeddings: &Tensor,
        offsets: &[(usize, usize)],
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let seq_len = token_embeddings.dim(0)?;
        const MIN_NORM: f32 = 1.0;
        let mut filtered_embs = Vec::with_capacity(seq_len);
        let mut filtered_offsets = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let emb = token_embeddings.get(i)?;
            let norm = emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if norm >= MIN_NORM {
                filtered_embs.push(emb);
                if i < offsets.len() {
                    filtered_offsets.push(offsets[i]);
                }
            }
        }
        if filtered_embs.is_empty() {
            anyhow::bail!("all token embeddings below minimum norm threshold");
        }
        let matrix = Tensor::stack(&filtered_embs, 0)?.unsqueeze(0)?;
        let normalized = normalize_l2(&matrix)?;
        Ok((normalized, filtered_offsets))
    }
}
