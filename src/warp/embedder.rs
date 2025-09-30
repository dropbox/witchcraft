use super::t5_encoder;
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
pub struct Embedder {
    tokenizer: Tokenizer,
    model: t5_encoder::T5EncoderModel,
}

impl Embedder {
    pub fn new(device: &Device, assets: &std::path::PathBuf) -> Result<Self> {
        let (builder, tokenizer) = t5_encoder::T5ModelBuilder::load(assets)?;
        let model = builder.build_encoder(&device, assets)?;
        Ok(Self { tokenizer, model })
    }
    pub fn embed(self: &Self, text: &str) -> Result<Tensor> {
        //let now = std::time::Instant::now();
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        //println!("embedder took {} ms.", now.elapsed().as_millis());
        normalize_l2(&embeddings)
    }
}
