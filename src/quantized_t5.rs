//! T5 model implementation with quantization support.
//!
//! T5 is an encoder-decoder model pre-trained on a multi-task mixture of supervised
//! and unsupervised tasks. This implementation provides quantization for reduced
//! memory and compute requirements.
//!
//! Key characteristics:
//! - Encoder-decoder architecture
//! - Layer normalization
//! - Relative positional encodings
//! - Support for 8-bit quantization
//!
//! References:
//! - 📝 [T5 Paper](https://arxiv.org/abs/1910.10683)
//! - 🤗 [Model Card](https://huggingface.co/t5-base)
//! - 🤗 Original model from [T5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)

use crate::embed_zst_asset;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Activation;
use candle_transformers::models::t5::{
    deserialize_feed_forward_proj_activation, ActivationWithOptionalGating,
};
#[cfg(not(feature = "hybrid-dequant"))]
use candle_core::quantized::QMatMul;
#[cfg(feature = "hybrid-dequant")]
use crate::fused_matmul::MatMul as QMatMul;
use candle_transformers::quantized_nn::Embedding;
use candle_transformers::quantized_var_builder::VarBuilder;
use serde::Deserialize;
use std::io::{Error, ErrorKind};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[cfg(not(feature = "hybrid-dequant"))]
fn new_qmm(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let device = vb.device();
    let ws = vb.get((out_d, in_d), "weight")?;
    if matches!(device, Device::Cpu) {
        let tensor = ws.dequantize(device)?;
        Ok(QMatMul::Tensor(tensor))
    } else {
        QMatMul::from_arc(ws)
    }
}

#[cfg(feature = "hybrid-dequant")]
fn new_qmm(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let ws = vb.get((out_d, in_d), "weight")?;
    Ok(QMatMul::from_qtensor(ws))
}

#[cfg(feature = "hybrid-dequant")]
fn new_qmm_dequant(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let ws = vb.get((out_d, in_d), "weight")?;
    let tensor = ws.dequantize(vb.device())?;
    Ok(QMatMul::from_tensor(tensor))
}

fn default_relative_attention_max_distance() -> usize {
    128
}

fn default_is_decoder() -> bool {
    false
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    num_layers: usize,
    num_decoder_layers: Option<usize>,
    num_heads: usize,
    relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    relative_attention_max_distance: usize,
    dropout_rate: f64,
    layer_norm_epsilon: f64,
    initializer_factor: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
    #[serde(default = "default_is_decoder")]
    is_decoder: bool,
    is_encoder_decoder: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub decoder_start_token_id: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            is_decoder: false,
            is_encoder_decoder: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?.dequantize(vb.device())?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseActDense {
    wi: QMatMul,
    wo: QMatMul,
    act: Activation,
}

impl T5DenseActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi = new_qmm(cfg.d_model, cfg.d_ff, vb.pp("wi"))?;
        #[cfg(feature = "hybrid-dequant")]
        let wo = new_qmm_dequant(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        #[cfg(not(feature = "hybrid-dequant"))]
        let wo = new_qmm(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi,
            wo,
            act: Activation::Relu,
        })
    }
}

impl Module for T5DenseActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.wi.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseGatedActDense {
    wi_0: QMatMul,
    wi_1: QMatMul,
    wo: QMatMul,
    act: Activation,
}

impl T5DenseGatedActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi_0 = new_qmm(cfg.d_model, cfg.d_ff, vb.pp("wi_0"))?;
        let wi_1 = new_qmm(cfg.d_model, cfg.d_ff, vb.pp("wi_1"))?;
        #[cfg(feature = "hybrid-dequant")]
        let wo = new_qmm_dequant(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        #[cfg(not(feature = "hybrid-dequant"))]
        let wo = new_qmm(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            act: cfg.feed_forward_proj.activation,
        })
    }
}

impl Module for T5DenseGatedActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "hybrid-dequant")]
        let hidden = match self.act {
            Activation::NewGelu | Activation::GeluPytorchTanh => {
                crate::fused_matmul::forward_gated_gelu(&self.wi_0, &self.wi_1, xs)?
            }
            _ => {
                let hidden_act = self.act.forward(&self.wi_0.forward(xs)?)?;
                let hidden_linear = self.wi_1.forward(xs)?;
                hidden_act.broadcast_mul(&hidden_linear)?
            }
        };
        #[cfg(not(feature = "hybrid-dequant"))]
        let hidden = {
            let hidden_act = self.act.forward(&self.wi_0.forward(xs)?)?;
            let hidden_linear = self.wi_1.forward(xs)?;
            hidden_act.broadcast_mul(&hidden_linear)?
        };
        self.wo.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
struct T5LayerFF {
    dense_act: Option<T5DenseActDense>,
    gated_dense_act: Option<T5DenseGatedActDense>,
    layer_norm: T5LayerNorm,
}

impl T5LayerFF {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let (dense_act, gated_dense_act) = if cfg.feed_forward_proj.gated {
            (
                None,
                Some(T5DenseGatedActDense::load(vb.pp("DenseReluDense"), cfg)?),
            )
        } else {
            (
                Some(T5DenseActDense::load(vb.pp("DenseReluDense"), cfg)?),
                None,
            )
        };
        Ok(Self {
            dense_act,
            gated_dense_act,
            layer_norm,
        })
    }
}

impl Module for T5LayerFF {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.layer_norm.forward(xs)?;
        let ys = match &self.dense_act {
            Some(dense_act) => dense_act.forward(&ys)?,
            None => self.gated_dense_act.as_ref().unwrap().forward(&ys)?,
        };
        let xs = crate::fast_ops::fast_add(xs, &ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5Attention {
    #[cfg(feature = "hybrid-dequant")]
    qkv: QMatMul,
    #[cfg(not(feature = "hybrid-dequant"))]
    q: QMatMul,
    #[cfg(not(feature = "hybrid-dequant"))]
    k: QMatMul,
    #[cfg(not(feature = "hybrid-dequant"))]
    v: QMatMul,
    o: QMatMul,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
}

impl T5Attention {
    fn load(has_relative_attention_bias: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        #[cfg(feature = "hybrid-dequant")]
        let (qkv, o) = {
            let q_w = vb.pp("q").get((inner_dim, cfg.d_model), "weight")?.dequantize(vb.device())?;
            let k_w = vb.pp("k").get((inner_dim, cfg.d_model), "weight")?.dequantize(vb.device())?;
            let v_w = vb.pp("v").get((inner_dim, cfg.d_model), "weight")?.dequantize(vb.device())?;
            let qkv = QMatMul::from_tensor(Tensor::cat(&[&q_w, &k_w, &v_w], 0)?);
            let o = new_qmm_dequant(inner_dim, cfg.d_model, vb.pp("o"))?;
            (qkv, o)
        };
        #[cfg(not(feature = "hybrid-dequant"))]
        let (q, k, v, o) = {
            let q = new_qmm(cfg.d_model, inner_dim, vb.pp("q"))?;
            let k = new_qmm(cfg.d_model, inner_dim, vb.pp("k"))?;
            let v = new_qmm(cfg.d_model, inner_dim, vb.pp("v"))?;
            let o = new_qmm(inner_dim, cfg.d_model, vb.pp("o"))?;
            (q, k, v, o)
        };
        let relative_attention_bias = if has_relative_attention_bias {
            let emb = Embedding::new(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            #[cfg(feature = "hybrid-dequant")]
            qkv,
            #[cfg(not(feature = "hybrid-dequant"))]
            q,
            #[cfg(not(feature = "hybrid-dequant"))]
            k,
            #[cfg(not(feature = "hybrid-dequant"))]
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);

        #[cfg(feature = "hybrid-dequant")]
        let (q, k, v) = {
            let _ = key_value_states;
            let qkv = self.qkv.forward(xs)?;
            let qkv = qkv
                .reshape((b_sz, q_len, 3, self.n_heads, self.d_kv))?
                .permute((2, 0, 3, 1, 4))?
                .contiguous()?;
            (
                qkv.narrow(0, 0, 1)?.squeeze(0)?,
                qkv.narrow(0, 1, 1)?.squeeze(0)?,
                qkv.narrow(0, 2, 1)?.squeeze(0)?,
            )
        };
        #[cfg(not(feature = "hybrid-dequant"))]
        let (q, k, v) = {
            let kv_input = match key_value_states {
                None => xs,
                Some(key_value_states) => key_value_states,
            };
            let kv_len = kv_input.dim(1)?;
            let q = self.q.forward(xs)?;
            let k = self.k.forward(kv_input)?;
            let v = self.v.forward(kv_input)?;
            let q = q
                .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
                .transpose(1, 2)?
                .contiguous()?;
            (q, k, v)
        };

        let scores = q.matmul(&k.t()?)?;
        let scores = match mask {
            None => scores,
            Some(mask) => masked_fill(
                &scores,
                &mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .repeat((b_sz, self.n_heads))?,
                f32::NEG_INFINITY,
            )?,
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => {
                let scores = crate::fast_ops::fast_add(&scores, position_bias)?;
                (scores, Some(position_bias.clone()))
            }
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    // This only handles the bidirectional case.
                    let kv_len = k.dim(2)?;
                    let (q_start, q_end) = (0_u32, kv_len as u32);
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        u32::min(max_exact + b as u32, num_buckets - 1)
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?
                        .contiguous()?;
                    let scores = crate::fast_ops::fast_add(&scores, &position_bias)?;
                    (scores, Some(position_bias))
                }
            },
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerSelfAttention {
    fn load(h: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, vb.pp("SelfAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            self_attention,
            layer_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attention
                .forward(&normed_xs, position_bias, None, mask)?;
        let ys = crate::fast_ops::fast_add(xs, &ys)?;
        Ok((ys, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5LayerCrossAttention {
    cross_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerCrossAttention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let cross_attention = T5Attention::load(false, vb.pp("EncDecAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            cross_attention,
            layer_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_hidden_states = self.layer_norm.forward(hidden_states)?;
        let (ys, position_bias) = self.cross_attention.forward(
            &normed_hidden_states,
            position_bias,
            Some(key_value_states),
            None,
        )?;
        let ys = crate::fast_ops::fast_add(hidden_states, &ys)?;
        Ok((ys, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(has_relative_attention_bias: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn = T5LayerSelfAttention::load(has_relative_attention_bias, vb.pp("0"), cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(ff_i), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // TODO: Cache masks
        let mask = match self.cross_attn.is_some() {
            true => {
                let mask_len = xs.dim(1)?;
                // If the input seq length is 1, no need for a mask, this is also helpful to avoid shape
                // issues when using the KV cache in the decoder.
                if mask_len <= 1 {
                    None
                } else {
                    Some(get_mask(mask_len, xs.device())?)
                }
            }
            false => None,
        };
        let (mut xs, position_bias) = self.self_attn.forward(xs, position_bias, mask.as_ref())?;
        // TODO: clamp for f16?
        if let Some(cross_attn) = &self.cross_attn {
            (xs, _) = cross_attn.forward(&xs, None, encoder_hidden_states.unwrap())?;
            // TODO: clamp for f16?
        }
        let xs = self.ff.forward(&xs)?;
        // TODO: clamp for f16?
        Ok((xs, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
}

impl T5Stack {
    fn load(vb: VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, vb.pp(format!("block.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter() {
            (hidden_states, position_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
                encoder_hidden_states,
            )?
        }
        self.final_layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    final_projection: QMatMul,
    device: Device,
}

impl T5EncoderModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let shared_vb = if vb.contains_key("shared.weight") {
            vb.pp("shared")
        } else {
            vb.pp("encoder").pp("embed_tokens")
        };
        let shared = Embedding::new(cfg.vocab_size, cfg.d_model, shared_vb)?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(vb.pp("encoder"), &shared, cfg)?;
        let final_projection = new_qmm(768, 128, vb.pp("linear"))?;
        Ok(Self {
            encoder,
            final_projection,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let encoder_output = self.encoder.forward(input_ids, None)?;
        self.final_projection.forward(&encoder_output)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

embed_zst_asset!(pub CONFIG,    "config.json.zst");
embed_zst_asset!(pub TOKENIZER, "tokenizer.json.zst");
embed_zst_asset!(pub MODEL,     "xtr.gguf.zst");

pub struct T5ModelBuilder {
    config: Config,
}

impl T5ModelBuilder {
    pub fn load(assets: &std::path::PathBuf) -> candle_core::Result<(Self, Tokenizer)> {
        // CONFIG: bytes -> JSON
        let cfg_bytes = CONFIG.bytes(assets).map_err(|_| {
            Error::new(
                ErrorKind::Other,
                "failed to get decompressed bytes for CONFIG",
            )
        })?;
        let config: Config = serde_json::from_slice(cfg_bytes).map_err(|e| {
            Error::new(
                ErrorKind::Other,
                format!("failed to parse CONFIG as JSON: {e}"),
            )
        })?;

        // TOKENIZER: bytes -> Tokenizer
        let tok_bytes = TOKENIZER.bytes(assets).map_err(|_| {
            Error::new(
                ErrorKind::Other,
                "failed to get decompressed bytes for TOKENIZER",
            )
        })?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("failed to parse TOKENIZER: {e}")))?;

        Ok((Self { config }, tokenizer))
    }

    pub fn build_encoder(
        &self,
        device: &Device,
        assets: &std::path::PathBuf,
    ) -> candle_core::Result<T5EncoderModel> {
        // MODEL: bytes -> gguf VarBuilder
        let model_bytes = MODEL.bytes(assets).map_err(|_| {
            Error::new(
                ErrorKind::Other,
                "failed to get decompressed bytes for MODEL",
            )
        })?;

        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            model_bytes,
            device,
        )?;

        let enc = T5EncoderModel::load(vb, &self.config)
            .map_err(|e| Error::new(ErrorKind::Other, format!("failed to load T5 encoder: {e}")))?;

        Ok(enc)
    }
}
