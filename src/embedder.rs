use super::t5_encoder;
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use log::debug;
use tokenizers::Tokenizer;

const MAX_LEN: usize = 2048;
const STRIDE: usize = 256;
const MIN_NORM: f32 = 1.0;
const DEFAULT_BATCH_SIZE: usize = 32;
const PAD_BUCKET_WIDTH: usize = 64;

pub(crate) fn embedding_batch_size() -> usize {
    std::env::var("WITCHCRAFT_EMBED_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_BATCH_SIZE)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}

fn model_forward_with_gate(model: &t5_encoder::T5EncoderModel, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
    #[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
    {
        Ok(model.forward_with_gate(input)?)
    }
    #[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
    {
        Ok((model.forward(input)?, None))
    }
}

fn model_forward_with_gate_for_lengths(
    model: &t5_encoder::T5EncoderModel,
    input: &Tensor,
    lengths: &[usize],
) -> Result<(Tensor, Option<Tensor>)> {
    #[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
    {
        Ok(model.forward_with_gate_for_lengths(input, lengths)?)
    }
    #[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
    {
        let _ = lengths;
        model_forward_with_gate(model, input)
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

    pub fn embed_with_gate_scores(&self, text: &str) -> Result<(Tensor, Vec<(usize, usize)>, Option<Vec<f32>>)> {
        let output = self.embed_inner(text, true)?;
        Ok((output.embeddings, output.offsets, output.gate_scores))
    }

    pub fn embed_with_gate_scores_and_tokens(&self, text: &str) -> Result<EmbeddingOutput> {
        self.embed_inner(text, true)
    }

    pub fn embed_batch_with_gate_scores_and_tokens(&self, texts: &[String]) -> Result<Vec<EmbeddingOutput>> {
        let now = std::time::Instant::now();
        let batch_size = embedding_batch_size();
        let encodings = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|err| anyhow!("tokenize text: {err}"))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut outputs: Vec<Option<EmbeddingOutput>> = (0..texts.len()).map(|_| None).collect();
        let mut batched_indices = encodings
            .iter()
            .enumerate()
            .filter_map(|(idx, encoding)| {
                let len = encoding.get_ids().len();
                (len <= MAX_LEN).then_some((idx, len))
            })
            .collect::<Vec<_>>();
        batched_indices.sort_by_key(|&(_, len)| len);

        let mut start = 0usize;
        while start < batched_indices.len() {
            let min_len = batched_indices[start].1;
            let mut end = start + 1;
            while end < batched_indices.len()
                && end - start < batch_size
                && batched_indices[end].1.saturating_sub(min_len) <= PAD_BUCKET_WIDTH
            {
                end += 1;
            }
            self.embed_tokenized_batch(&encodings, &batched_indices[start..end], &mut outputs)?;
            start = end;
        }

        for (idx, encoding) in encodings.iter().enumerate() {
            if outputs[idx].is_none() {
                outputs[idx] = Some(self.embed_inner(texts[idx].as_str(), true)?);
            }
            debug!(
                "batched embedder doc {} tokenized to {} tokens.",
                idx,
                encoding.get_ids().len()
            );
        }

        debug!(
            "batched embedder took {} ms for {} docs.",
            now.elapsed().as_millis(),
            texts.len(),
        );
        outputs
            .into_iter()
            .map(|output| output.ok_or_else(|| anyhow!("missing batched embedding output")))
            .collect()
    }

    fn embed_tokenized_batch(
        &self,
        encodings: &[tokenizers::Encoding],
        batch: &[(usize, usize)],
        outputs: &mut [Option<EmbeddingOutput>],
    ) -> Result<()> {
        let model = &self.model;
        let device = model.device();
        let max_len = batch.iter().map(|&(_, len)| len).max().unwrap_or(0);
        let mut input_ids = Vec::with_capacity(batch.len() * max_len);
        let mut lengths = Vec::with_capacity(batch.len());
        for &(idx, len) in batch {
            let ids = encodings[idx].get_ids();
            lengths.push(len);
            input_ids.extend_from_slice(ids);
            input_ids.resize(input_ids.len() + (max_len - len), 0);
        }

        let input = Tensor::from_vec(input_ids, (batch.len(), max_len), device)?;
        let (embeddings, gates) = model_forward_with_gate_for_lengths(model, &input, &lengths)?;
        let embeddings = embeddings.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let embedding_rows = embeddings.to_vec3::<f32>()?;
        let gates = gates
            .map(|g| g.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>())
            .transpose()?;

        for (batch_idx, &(doc_idx, len)) in batch.iter().enumerate() {
            let rows = embedding_rows
                .get(batch_idx)
                .ok_or_else(|| anyhow!("missing embedding batch row {batch_idx}"))?;
            let gate_scores = gates
                .as_ref()
                .map(|gates| {
                    gates
                        .get(batch_idx)
                        .ok_or_else(|| anyhow!("missing gate batch row {batch_idx}"))
                        .map(|scores| scores[..len].to_vec())
                })
                .transpose()?;
            outputs[doc_idx] = Some(finish_embedding_from_rows(
                &rows[..len],
                encodings[doc_idx].get_offsets().to_vec(),
                encodings[doc_idx].get_tokens().to_vec(),
                gate_scores,
                len,
            )?);
        }
        Ok(())
    }

    fn embed_inner(&self, text: &str, collect_gate_scores: bool) -> Result<EmbeddingOutput> {
        let now = std::time::Instant::now();
        let model = &self.model;
        let device = model.device();

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
            let end = (start + MAX_LEN).min(n_tokens);
            let input = Tensor::new(&ids[start..end], device)?.unsqueeze(0)?;
            let (chunk, gates) = if collect_gate_scores {
                model_forward_with_gate(model, &input)?
            } else {
                (model.forward(&input)?, None)
            };
            let chunk = chunk.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let gates = gates
                .map(|g| {
                    g.squeeze(0)?
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()
                })
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
                    },
                }
                if let Some(gates) = gates.as_ref() {
                    if gate_scores.is_none() {
                        gate_scores = Some(vec![0.0; n_tokens]);
                        gate_counts = Some(vec![0; n_tokens]);
                    }
                    if let (Some(scores), Some(counts)) = (gate_scores.as_mut(), gate_counts.as_mut()) {
                        scores[global_idx] += gates[i];
                        counts[global_idx] += 1;
                    }
                }
            }

            if end == n_tokens {
                break;
            }
            start = end.saturating_sub(STRIDE); // overlap window
        }

        let token_embs: Vec<Tensor> = accum
            .into_iter()
            .enumerate()
            .map(|(i, maybe_t)| {
                maybe_t.unwrap_or_else(|| panic!("Missing embedding for token {} — check stride settings", i))
            })
            .collect();

        let output = finish_embedding(token_embs, offsets, tokens, gate_scores, gate_counts, n_tokens)?;
        debug!(
            "embedder took {} ms, kept {}/{} tokens.",
            now.elapsed().as_millis(),
            output.offsets.len(),
            n_tokens,
        );
        Ok(output)
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

fn finish_embedding(
    token_embs: Vec<Tensor>,
    offsets: Vec<(usize, usize)>,
    tokens: Vec<String>,
    gate_scores: Option<Vec<f32>>,
    gate_counts: Option<Vec<u32>>,
    n_tokens: usize,
) -> Result<EmbeddingOutput> {
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
                let count = gate_counts
                    .as_ref()
                    .map(|counts| counts[idx].max(1) as f32)
                    .unwrap_or(1.0);
                out.push(scores[idx] / count);
            }
        }
    }
    if filtered_embs.is_empty() {
        anyhow::bail!("all token embeddings below minimum norm threshold");
    }

    let matrix = Tensor::stack(&filtered_embs, 0)?.unsqueeze(0)?;
    let normalized = normalize_l2(&matrix)?;
    debug!("kept {}/{} tokens after norm filtering.", filtered_embs.len(), n_tokens,);
    Ok(EmbeddingOutput {
        embeddings: normalized,
        offsets: filtered_offsets,
        gate_scores: filtered_scores,
        tokens: filtered_tokens,
    })
}

fn finish_embedding_from_rows(
    rows: &[Vec<f32>],
    offsets: Vec<(usize, usize)>,
    tokens: Vec<String>,
    gate_scores: Option<Vec<f32>>,
    n_tokens: usize,
) -> Result<EmbeddingOutput> {
    let dim = rows
        .first()
        .map(|row| row.len())
        .ok_or_else(|| anyhow!("empty embedding rows"))?;
    let mut filtered_values = Vec::with_capacity(rows.len() * dim);
    let mut filtered_offsets = Vec::with_capacity(offsets.len());
    let mut filtered_tokens = Vec::with_capacity(tokens.len());
    let mut filtered_scores = gate_scores.as_ref().map(|_| Vec::with_capacity(rows.len()));

    for (idx, (row, offset)) in rows.iter().zip(offsets.into_iter()).enumerate() {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm >= MIN_NORM {
            let inv_norm = norm.recip();
            filtered_values.extend(row.iter().map(|value| value * inv_norm));
            filtered_offsets.push(offset);
            filtered_tokens.push(tokens[idx].clone());
            if let Some(out) = filtered_scores.as_mut() {
                let scores = gate_scores.as_ref().expect("gate scores missing");
                out.push(scores[idx]);
            }
        }
    }
    if filtered_offsets.is_empty() {
        anyhow::bail!("all token embeddings below minimum norm threshold");
    }

    debug!(
        "kept {}/{} tokens after norm filtering.",
        filtered_offsets.len(),
        n_tokens,
    );
    let embedding_count = filtered_offsets.len();
    Ok(EmbeddingOutput {
        embeddings: Tensor::from_vec(filtered_values, (1, embedding_count, dim), &Device::Cpu)?,
        offsets: filtered_offsets,
        gate_scores: filtered_scores,
        tokens: filtered_tokens,
    })
}
