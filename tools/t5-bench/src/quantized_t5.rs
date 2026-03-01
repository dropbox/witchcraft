// Standalone copy of quantized_t5.rs for benchmarking (no embed_zst_asset dependency).

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::Activation;
use candle_transformers::models::t5::{
    deserialize_feed_forward_proj_activation, ActivationWithOptionalGating,
};
#[cfg(not(feature = "fused-gelu"))]
use candle_core::quantized::QMatMul;
#[cfg(feature = "fused-gelu")]
use crate::fused_matmul::MatMul as QMatMul;
use candle_transformers::quantized_nn::Embedding;
use candle_transformers::quantized_var_builder::VarBuilder;
use serde::Deserialize;
use std::sync::Arc;

#[cfg(not(feature = "fused-gelu"))]
fn new_qmm(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let ws = vb.get((out_d, in_d), "weight")?;
    QMatMul::from_arc(ws)
}

#[cfg(feature = "fused-gelu")]
fn new_qmm(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let ws = vb.get((out_d, in_d), "weight")?;
    Ok(QMatMul::from_qtensor(ws))
}

#[cfg(feature = "fused-gelu")]
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
        #[cfg(feature = "fused-gelu")]
        let wo = new_qmm_dequant(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        #[cfg(not(feature = "fused-gelu"))]
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
        #[cfg(feature = "fused-gelu")]
        let wo = new_qmm_dequant(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        #[cfg(not(feature = "fused-gelu"))]
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
        #[cfg(feature = "fused-gelu")]
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
        #[cfg(not(feature = "fused-gelu"))]
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
        let xs = (xs + ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5Attention {
    q: QMatMul,
    k: QMatMul,
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
        #[cfg(feature = "fused-gelu")]
        let (q, k, v, o) = {
            let q = new_qmm_dequant(cfg.d_model, inner_dim, vb.pp("q"))?;
            let k = new_qmm_dequant(cfg.d_model, inner_dim, vb.pp("k"))?;
            let v = new_qmm_dequant(cfg.d_model, inner_dim, vb.pp("v"))?;
            let o = new_qmm_dequant(inner_dim, cfg.d_model, vb.pp("o"))?;
            (q, k, v, o)
        };
        #[cfg(not(feature = "fused-gelu"))]
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
            q,
            k,
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
        _mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let kv_input = match key_value_states {
            None => xs,
            Some(key_value_states) => key_value_states,
        };
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
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

        let scores = q.matmul(&k.t()?)?;

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
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
                        .unsqueeze(0)?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
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
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attention
                .forward(&normed_xs, position_bias, None, None)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(has_relative_attention_bias: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn = T5LayerSelfAttention::load(has_relative_attention_bias, vb.pp("0"), cfg)?;
        let ff = T5LayerFF::load(vb.pp("1"), cfg)?;
        Ok(Self { self_attn, ff })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (xs, position_bias) = self.self_attn.forward(xs, position_bias)?;
        let xs = self.ff.forward(&xs)?;
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

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter() {
            (hidden_states, position_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
            )?
        }
        self.final_layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    final_projection: QMatMul,
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
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let encoder_output = self.encoder.forward(input_ids)?;
        self.final_projection.forward(&encoder_output)
    }
}
