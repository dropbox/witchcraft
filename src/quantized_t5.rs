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

#[cfg(feature = "hybrid-dequant")]
use crate::fused_matmul::MatMul as QMatMul;
#[cfg(not(feature = "hybrid-dequant"))]
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Activation;
use candle_transformers::models::t5::{
    deserialize_feed_forward_proj_activation, ActivationWithOptionalGating,
};
use candle_transformers::quantized_nn::Embedding;
use candle_transformers::quantized_var_builder::VarBuilder;
use serde::Deserialize;
use std::io::Error;
use std::sync::Arc;

#[cfg(feature = "flash-attention")]
use std::sync::OnceLock;

#[cfg(feature = "flash-attention")]
static FA2_KERNELS: OnceLock<crate::triton_kernels::T5Kernels> = OnceLock::new();
use tokenizers::Tokenizer;

use crate::embed_asset;
embed_asset!(pub CONFIG,    "config.json");
embed_asset!(pub TOKENIZER, "tokenizer.json");
embed_asset!(pub MODEL,     "xtr.gguf");

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

/// Pre-dequantize to F16 on Metal for FA2 attention path (faster matmul, no per-call cast).
#[cfg(feature = "flash-attention")]
fn new_qmm_attn(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let ws = vb.get((out_d, in_d), "weight")?;
    let device = vb.device();
    let tensor = ws.dequantize(device)?;
    if matches!(device, Device::Metal(_)) {
        Ok(QMatMul::Tensor(tensor.to_dtype(DType::F16)?))
    } else {
        Ok(QMatMul::Tensor(tensor))
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
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f64,
    pub layer_norm_epsilon: f64,
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
        #[cfg(feature = "flash-attention")]
        if let Device::Metal(metal_device) = xs.device() {
            if let Some(kernels) = FA2_KERNELS.get() {
                let shape = xs.shape();
                let dim = shape.dims().last().copied().unwrap();
                let n_rows = xs.elem_count() / dim;
                let xs_2d = xs.reshape((n_rows, dim))?.contiguous()?;
                let out = crate::triton_kernels::triton_rms_norm(
                    metal_device,
                    &kernels.rms_norm,
                    &xs_2d,
                    &self.weight,
                    self.variance_epsilon as f32,
                )
                .map_err(|e| candle_core::Error::Msg(format!("RMS norm: {e}")))?;
                return Ok(out.reshape(shape)?);
            }
        }

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
        {
            let hidden_act = self.act.forward(&self.wi_0.forward(xs)?)?;
            let hidden_linear = self.wi_1.forward(xs)?;
            self.wo.forward(&hidden_act.broadcast_mul(&hidden_linear)?)
        }
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
#[allow(dead_code)] // Variants are conditionally used based on hybrid-dequant feature
enum AttentionWeights {
    /// Fused QKV matrix (hybrid-dequant: concatenated q, k, v for single matmul)
    Fused(QMatMul),
    /// Separate Q, K, V matrices
    Separate { q: QMatMul, k: QMatMul, v: QMatMul },
}

#[derive(Debug, Clone)]
struct T5Attention {
    qkv: AttentionWeights,
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
            let q_w = vb
                .pp("q")
                .get((inner_dim, cfg.d_model), "weight")?
                .dequantize(vb.device())?;
            let k_w = vb
                .pp("k")
                .get((inner_dim, cfg.d_model), "weight")?
                .dequantize(vb.device())?;
            let v_w = vb
                .pp("v")
                .get((inner_dim, cfg.d_model), "weight")?
                .dequantize(vb.device())?;
            let qkv =
                AttentionWeights::Fused(QMatMul::from_tensor(Tensor::cat(&[&q_w, &k_w, &v_w], 0)?));
            let o = new_qmm_dequant(inner_dim, cfg.d_model, vb.pp("o"))?;
            (qkv, o)
        };

        #[cfg(all(not(feature = "hybrid-dequant"), feature = "flash-attention"))]
        let (qkv, o) = {
            let q = new_qmm_attn(cfg.d_model, inner_dim, vb.pp("q"))?;
            let k = new_qmm_attn(cfg.d_model, inner_dim, vb.pp("k"))?;
            let v = new_qmm_attn(cfg.d_model, inner_dim, vb.pp("v"))?;
            let qkv = AttentionWeights::Separate { q, k, v };
            let o = new_qmm_attn(inner_dim, cfg.d_model, vb.pp("o"))?;
            (qkv, o)
        };

        #[cfg(all(not(feature = "hybrid-dequant"), not(feature = "flash-attention")))]
        let (qkv, o) = {
            let q = new_qmm(cfg.d_model, inner_dim, vb.pp("q"))?;
            let k = new_qmm(cfg.d_model, inner_dim, vb.pp("k"))?;
            let v = new_qmm(cfg.d_model, inner_dim, vb.pp("v"))?;
            let qkv = AttentionWeights::Separate { q, k, v };
            let o = new_qmm(inner_dim, cfg.d_model, vb.pp("o"))?;
            (qkv, o)
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
            qkv,
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

        let (q, k, v) = match &self.qkv {
            AttentionWeights::Fused(qkv_mm) => {
                let _ = key_value_states;
                let qkv = qkv_mm.forward(xs)?;
                let qkv = qkv
                    .reshape((b_sz, q_len, 3, self.n_heads, self.d_kv))?
                    .permute((2, 0, 3, 1, 4))?
                    .contiguous()?;
                (
                    qkv.narrow(0, 0, 1)?.squeeze(0)?,
                    qkv.narrow(0, 1, 1)?.squeeze(0)?,
                    qkv.narrow(0, 2, 1)?.squeeze(0)?,
                )
            }
            AttentionWeights::Separate {
                q: q_mm,
                k: k_mm,
                v: v_mm,
            } => {
                let kv_input = match key_value_states {
                    None => xs,
                    Some(key_value_states) => key_value_states,
                };
                let kv_len = kv_input.dim(1)?;

                // FA2 on Metal: weights are F16, cast inputs to match
                #[cfg(feature = "flash-attention")]
                let (xs_proj, kv_proj);
                #[cfg(feature = "flash-attention")]
                if matches!(xs.device(), Device::Metal(_)) {
                    xs_proj = xs.to_dtype(DType::F16)?;
                    kv_proj = if key_value_states.is_none() {
                        xs_proj.clone()
                    } else {
                        kv_input.to_dtype(DType::F16)?
                    };
                } else {
                    xs_proj = xs.clone();
                    kv_proj = kv_input.clone();
                }
                #[cfg(not(feature = "flash-attention"))]
                let (xs_proj, kv_proj) = (xs.clone(), kv_input.clone());

                let q = q_mm.forward(&xs_proj)?;
                let k = k_mm.forward(&kv_proj)?;
                let v = v_mm.forward(&kv_proj)?;
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
            }
        };

        // Compute or propagate position bias (needed before attention for FA2 path)
        let kv_len = k.dim(2)?;
        let position_bias = match position_bias {
            Some(position_bias) => Some(position_bias.clone()),
            None => match &self.relative_attention_bias {
                None => None,
                Some(relative_attention_bias) => {
                    // Bidirectional relative position bias
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
                    Some(
                        relative_attention_bias
                            .forward(&relative_buckets)?
                            .permute((2, 0, 1))?
                            .unsqueeze(0)?
                            .contiguous()?,
                    )
                }
            },
        };

        // Flash Attention 2 path: fused QK^T + bias + softmax + @V
        // Only for encoder self-attention on Metal (no mask, batch=1)
        #[cfg(feature = "flash-attention")]
        if mask.is_none() && b_sz == 1 {
            if let Device::Metal(metal_device) = q.device() {
                if let Some(ref bias) = position_bias {
                    let kernels = FA2_KERNELS.get_or_init(|| {
                        crate::triton_kernels::T5Kernels::load(metal_device)
                            .expect("Failed to load FA2 Metal kernels")
                    });

                    // Q/K/V are already F16 (from F16 weight projections).
                    // Pad to BM=32-aligned rows so kernel writes don't overflow heads.
                    const BM: usize = 32;
                    let padded_seq = (q_len + BM - 1) / BM * BM;
                    let dev = q.device();

                    let q16 = q.squeeze(0)?.contiguous()?;
                    let k16 = k.squeeze(0)?.contiguous()?;
                    let v16 = v.squeeze(0)?.contiguous()?;

                    let (q16, k16, v16) = if q_len < padded_seq {
                        let pad = |t: Tensor| -> Result<Tensor> {
                            let pad_rows = padded_seq - q_len;
                            let zeros = Tensor::zeros(
                                (self.n_heads, pad_rows, self.d_kv),
                                DType::F16,
                                dev,
                            )?;
                            Tensor::cat(&[&t, &zeros], 1)?.contiguous()
                        };
                        (pad(q16)?, pad(k16)?, pad(v16)?)
                    } else {
                        (q16, k16, v16)
                    };

                    let out = q16.zeros_like()?;

                    // Bias: [1, n_heads, q_len, kv_len] → [n_heads, q_len, kv_len] F32
                    let bias_3d = bias.squeeze(0)?.contiguous()?;

                    crate::triton_kernels::triton_flash_attention_bias(
                        metal_device,
                        &kernels.flash_attention_bias,
                        &q16,
                        &k16,
                        &v16,
                        &out,
                        &bias_3d,
                        self.n_heads,
                        q_len,
                        self.d_kv,
                        1.0,
                    )
                    .map_err(|e| candle_core::Error::Msg(format!("FA2 dispatch: {e}")))?;

                    // [n_heads, q_len, d_kv] F16 → [1, q_len, inner_dim] F16
                    let fa2_out = out.narrow(1, 0, q_len)?;
                    let attn_output = fa2_out
                        .transpose(0, 1)?
                        .contiguous()?
                        .reshape((1, q_len, self.inner_dim))?;
                    // O projection in F16, then back to F32 for residual add
                    let attn_output = self.o.forward(&attn_output)?
                        .to_dtype(DType::F32)?;
                    return Ok((attn_output, position_bias));
                }
            }
        }

        // Standard attention path (CPU or fallback)
        // When flash-attention is enabled, Q/K/V may be F16 — cast to F32 for softmax precision.
        #[cfg(feature = "flash-attention")]
        let (q, k, v) = (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?);

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

        let scores = match &position_bias {
            Some(bias) => crate::fast_ops::fast_add(&scores, bias)?,
            None => scores,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        // When flash-attention is enabled, O has F16 weights — project then cast back to F32.
        #[cfg(feature = "flash-attention")]
        let attn_output = {
            let attn_output = attn_output.to_dtype(DType::F16)?;
            self.o.forward(&attn_output)?.to_dtype(DType::F32)?
        };
        #[cfg(not(feature = "flash-attention"))]
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

pub struct T5EncoderModel {
    encoder: T5Stack,
    final_projection: QMatMul,
    config: Config,
    device: Device,
    #[cfg(feature = "flash-attention")]
    gpu_encoder: Option<crate::gpu_t5_encoder::GpuT5Encoder>,
    #[cfg(feature = "triton-d3d12")]
    gpu_encoder_d3d12: Option<crate::gpu_t5_encoder_d3d12::GpuT5EncoderD3D12>,
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
            config: cfg.clone(),
            device: vb.device().clone(),
            #[cfg(feature = "flash-attention")]
            gpu_encoder: None,
            #[cfg(feature = "triton-d3d12")]
            gpu_encoder_d3d12: None,
        })
    }

    /// Try to create a GPU encoder for Metal acceleration.
    /// Loads weights from GGUF directly (separate from candle model weights).
    #[cfg(feature = "flash-attention")]
    pub fn try_init_gpu_encoder(
        &mut self,
        cfg: &Config,
        assets: &std::path::Path,
    ) -> std::result::Result<(), anyhow::Error> {
        use log::{info, warn};

        // Create own Metal device (separate from candle's, which may be CPU)
        let metal_device = match Device::new_metal(0) {
            Ok(Device::Metal(md)) => md,
            _ => {
                warn!("GPU encoder: no Metal device available");
                return Ok(());
            }
        };

        info!("GPU encoder: loading weights to Metal device...");
        let now = std::time::Instant::now();

        // Load GGUF weights directly for the GPU encoder
        let model_bytes = MODEL
            .bytes(assets)
            .map_err(|e| anyhow::anyhow!("failed to get GGUF bytes: {e:?}"))?;
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            model_bytes,
            &Device::Cpu,
        )
        .map_err(|e| anyhow::anyhow!("GGUF VarBuilder: {e}"))?;

        let gpu_enc = crate::gpu_t5_encoder::GpuT5Encoder::new(
            &metal_device,
            cfg,
            vb,
            512, // max seq len
        )?;

        info!(
            "GPU encoder: loaded in {}ms",
            now.elapsed().as_millis()
        );
        self.gpu_encoder = Some(gpu_enc);
        Ok(())
    }

    /// Try to create a D3D12 GPU encoder for Windows GPU acceleration.
    #[cfg(feature = "triton-d3d12")]
    pub fn try_init_gpu_encoder_d3d12(
        &mut self,
        cfg: &Config,
        assets: &std::path::Path,
    ) -> std::result::Result<(), anyhow::Error> {
        use log::info;

        info!("D3D12 GPU encoder: loading weights...");
        let now = std::time::Instant::now();

        let model_bytes = MODEL
            .bytes(assets)
            .map_err(|e| anyhow::anyhow!("failed to get GGUF bytes: {e:?}"))?;
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            model_bytes,
            &Device::Cpu,
        )
        .map_err(|e| anyhow::anyhow!("GGUF VarBuilder: {e}"))?;

        // Max 1024 tokens: keeps the full batched dispatch under the Windows TDR
        // timeout (~2s) on integrated GPUs. Larger sequences fall back to CPU.
        let gpu_enc = crate::gpu_t5_encoder_d3d12::GpuT5EncoderD3D12::new(cfg, vb, 1024)?;

        info!("D3D12 GPU encoder: loaded in {}ms", now.elapsed().as_millis());
        self.gpu_encoder_d3d12 = Some(gpu_enc);

        // Drop CPU encoder blocks — the GPU handles nearly all inputs.
        // Keeps shared embedding table and final_layer_norm for the rare CPU fallback.
        let n_blocks = self.encoder.block.len();
        self.encoder.block.clear();
        info!("Dropped {n_blocks} CPU encoder blocks to save memory");

        Ok(())
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // GPU encoder path: embedding on CPU → full encoder on GPU → final projection on CPU
        #[cfg(feature = "flash-attention")]
        if let Some(ref gpu_enc) = self.gpu_encoder {
            let input_embeds = self.encoder.shared.as_ref().forward(input_ids)?;
            let encoder_output = gpu_enc
                .forward(&input_embeds)
                .map_err(|e| candle_core::Error::Msg(format!("GPU encoder: {e}")))?;
            let encoder_output = encoder_output.to_device(&self.device)?;
            return self.final_projection.forward(&encoder_output);
        }

        // D3D12 GPU encoder path (Windows) — falls back to CPU for oversized inputs
        #[cfg(feature = "triton-d3d12")]
        if let Some(ref gpu_enc) = self.gpu_encoder_d3d12 {
            let input_embeds = self.encoder.shared.as_ref().forward(input_ids)?;
            match gpu_enc.forward(&input_embeds) {
                Ok(encoder_output) => {
                    return self.final_projection.forward(&encoder_output);
                }
                Err(e) => {
                    log::info!("D3D12 GPU fallback to CPU: {e}");
                    // GPU encoder blocks were dropped to save memory — run the
                    // full candle CPU encoder for this oversized input.
                    return self.cpu_fallback_forward(input_ids);
                }
            }
        }

        let encoder_output = self.encoder.forward(input_ids, None)?;
        self.final_projection.forward(&encoder_output)
    }

    /// One-shot CPU forward for oversized sequences that don't fit the GPU.
    /// Loads encoder blocks from GGUF on demand, runs forward, drops them.
    #[cfg(feature = "triton-d3d12")]
    fn cpu_fallback_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let model_bytes = MODEL
            .bytes(std::path::Path::new("assets"))
            .or_else(|_| MODEL.bytes(std::path::Path::new(".")))
            .map_err(|_| candle_core::Error::Msg("CPU fallback: failed to load GGUF".into()))?;
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            model_bytes,
            &Device::Cpu,
        )?;
        let tmp_enc = T5Stack::load(vb.pp("encoder"), &self.encoder.shared, &self.config)?;
        log::info!("CPU fallback: loaded encoder in {}ms", t0.elapsed().as_millis());
        let encoder_output = tmp_enc.forward(input_ids, None)?;
        self.final_projection.forward(&encoder_output)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

pub struct T5ModelBuilder {
    config: Config,
}

impl T5ModelBuilder {
    pub fn load(assets: &std::path::Path) -> candle_core::Result<(Self, Tokenizer)> {
        // CONFIG: bytes -> JSON
        let cfg_bytes = CONFIG
            .bytes(assets)
            .map_err(|_| Error::other("failed to get bytes for CONFIG"))?;
        let config: Config = serde_json::from_slice(cfg_bytes)
            .map_err(|e| Error::other(format!("failed to parse CONFIG as JSON: {e}")))?;

        // TOKENIZER: bytes -> Tokenizer
        let tok_bytes = TOKENIZER
            .bytes(assets)
            .map_err(|_| Error::other("failed to get bytes for TOKENIZER"))?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| Error::other(format!("failed to parse TOKENIZER: {e}")))?;

        Ok((Self { config }, tokenizer))
    }

    pub fn build_encoder(
        &self,
        device: &Device,
        assets: &std::path::Path,
    ) -> candle_core::Result<T5EncoderModel> {
        let model_bytes = MODEL
            .bytes(assets)
            .map_err(|_| Error::other("failed to get bytes for MODEL"))?;

        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
            model_bytes,
            device,
        )?;

        let mut enc = T5EncoderModel::load(vb, &self.config)
            .map_err(|e| Error::other(format!("failed to load T5 encoder: {e}")))?;

        // On Intel Macs, use full Triton GPU encoder (candle Metal crashes on Intel GPUs).
        #[cfg(all(feature = "flash-attention", target_arch = "x86_64"))]
        if let Err(e) = enc.try_init_gpu_encoder(&self.config, assets) {
            log::warn!("GPU encoder init failed (falling back to CPU): {e}");
        }

        // On Windows, use D3D12 GPU encoder with Triton-compiled DXIL shaders.
        #[cfg(feature = "triton-d3d12")]
        if let Err(e) = enc.try_init_gpu_encoder_d3d12(&self.config, assets) {
            log::warn!("D3D12 GPU encoder init failed (falling back to CPU): {e}");
        }

        Ok(enc)
    }
}
