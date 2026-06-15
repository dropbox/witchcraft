//! Quantized ModernBERT encoder — GGUF Q4K weights via candle quantized backend.
//!
//! Same architecture as modernbert.rs (RoPE, GeGLU MLP, RMSNorm, sliding window)
//! but loads from GGUF with quantized weight matrices for ~7x size reduction.

#[cfg(feature = "hybrid-dequant")]
use crate::fused_matmul::MatMul as QMatMul;
#[cfg(not(feature = "hybrid-dequant"))]
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Activation;
use candle_transformers::quantized_var_builder::VarBuilder;
use serde::Deserialize;
use std::io::Error;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

use crate::embed_asset;

embed_asset!(pub CONFIG,    "modernbert-config.json");
embed_asset!(pub TOKENIZER, "modernbert-tokenizer.json");
embed_asset!(pub MODEL,     "modernbert.gguf");

#[derive(Debug, Deserialize)]
struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rope_theta: f64,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    norm_eps: f64,
    #[serde(default)]
    projection_mlp: Option<usize>,
    #[serde(default = "default_projection_dim")]
    projection_dim: usize,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
    #[serde(default = "default_global_attn_every_n")]
    global_attn_every_n_layers: usize,
    #[serde(default = "default_activation")]
    hidden_activation: String,
}

fn default_activation() -> String {
    "gelu".to_string()
}

fn default_projection_dim() -> usize {
    128
}

fn default_local_attention() -> usize {
    128
}

fn default_global_attn_every_n() -> usize {
    3
}

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
    #[cfg(feature = "fbgemm")]
    {
        new_qmm_dequant(in_d, out_d, vb)
    }
    #[cfg(not(feature = "fbgemm"))]
    {
        let ws = vb.get((out_d, in_d), "weight")?;
        Ok(QMatMul::from_qtensor(ws))
    }
}

#[cfg(feature = "hybrid-dequant")]
fn new_qmm_dequant(in_d: usize, out_d: usize, vb: VarBuilder) -> Result<QMatMul> {
    let ws = vb.get((out_d, in_d), "weight")?;
    let tensor = ws.dequantize(vb.device())?;
    Ok(QMatMul::from_tensor(tensor))
}

#[derive(Debug, Clone)]
struct LayerNormNoBias {
    weight: Tensor,
    eps: f64,
}

impl LayerNormNoBias {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self { weight, eps })
    }
}

impl candle_core::Module for LayerNormNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let mean = xs_f32.mean_keepdim(D::Minus1)?;
        let centered = xs_f32.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(dtype)?;
        normed.broadcast_mul(&self.weight)
    }
}

fn build_rope_cache(
    max_len: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
    let positions: Vec<f32> = (0..max_len).map(|i| i as f32).collect();
    let positions = Tensor::new(positions.as_slice(), device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half)?.contiguous()?;
    let x2 = x.narrow(3, half, half)?.contiguous()?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let r1 = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let r2 = x2
        .broadcast_mul(&cos)?
        .broadcast_add(&x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&r1, &r2], 3)?.contiguous()
}

static PROFILE_REMAINING: OnceLock<AtomicUsize> = OnceLock::new();

#[derive(Default)]
struct ForwardProfile {
    seq_len: usize,
    started: Option<Instant>,
    embedding: Duration,
    final_norm: Duration,
    projection: Duration,
    layers: Vec<LayerProfile>,
}

#[derive(Default)]
struct LayerProfile {
    index: usize,
    attention_kind: &'static str,
    attn_norm: Duration,
    qkv: Duration,
    rope: Duration,
    full_scores: Duration,
    mask: Duration,
    softmax: Duration,
    value: Duration,
    fused_local: Duration,
    output_projection: Duration,
    attention_total: Duration,
    mlp_norm: Duration,
    mlp_wi: Duration,
    mlp_activation: Duration,
    mlp_wo: Duration,
    mlp_total: Duration,
    total: Duration,
}

fn profile_enabled() -> bool {
    match std::env::var("WARP_MODERNBERT_PROFILE") {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}

fn profile_target_tokens() -> Option<usize> {
    std::env::var("WARP_MODERNBERT_PROFILE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
}

fn profile_take(seq_len: usize) -> bool {
    if !profile_enabled() {
        return false;
    }
    if let Some(target) = profile_target_tokens() {
        if seq_len != target {
            return false;
        }
    }
    let remaining = PROFILE_REMAINING.get_or_init(|| {
        let limit = std::env::var("WARP_MODERNBERT_PROFILE_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        AtomicUsize::new(limit)
    });
    remaining
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_sub(1)
        })
        .is_ok()
}

impl ForwardProfile {
    fn new(seq_len: usize) -> Self {
        Self {
            seq_len,
            started: Some(Instant::now()),
            ..Self::default()
        }
    }

    fn print(&self) {
        let total = self
            .started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        let mut attn_norm = Duration::default();
        let mut qkv = Duration::default();
        let mut rope = Duration::default();
        let mut full_scores = Duration::default();
        let mut mask = Duration::default();
        let mut softmax = Duration::default();
        let mut value = Duration::default();
        let mut fused_local = Duration::default();
        let mut output_projection = Duration::default();
        let mut attention_total = Duration::default();
        let mut mlp_norm = Duration::default();
        let mut mlp_wi = Duration::default();
        let mut mlp_activation = Duration::default();
        let mut mlp_wo = Duration::default();
        let mut mlp_total = Duration::default();
        let mut layers_total = Duration::default();

        for layer in &self.layers {
            attn_norm += layer.attn_norm;
            qkv += layer.qkv;
            rope += layer.rope;
            full_scores += layer.full_scores;
            mask += layer.mask;
            softmax += layer.softmax;
            value += layer.value;
            fused_local += layer.fused_local;
            output_projection += layer.output_projection;
            attention_total += layer.attention_total;
            mlp_norm += layer.mlp_norm;
            mlp_wi += layer.mlp_wi;
            mlp_activation += layer.mlp_activation;
            mlp_wo += layer.mlp_wo;
            mlp_total += layer.mlp_total;
            layers_total += layer.total;
        }

        eprintln!(
            "modernbert-profile: seq={} total={:.2}ms embedding={:.2}ms layers={:.2}ms final_norm={:.2}ms projection={:.2}ms",
            self.seq_len,
            ms(total),
            ms(self.embedding),
            ms(layers_total),
            ms(self.final_norm),
            ms(self.projection),
        );
        eprintln!(
            "modernbert-profile: attention total={:.2}ms norm={:.2}ms qkv={:.2}ms rope={:.2}ms full_scores={:.2}ms mask={:.2}ms softmax={:.2}ms value={:.2}ms fused_local={:.2}ms out={:.2}ms",
            ms(attention_total),
            ms(attn_norm),
            ms(qkv),
            ms(rope),
            ms(full_scores),
            ms(mask),
            ms(softmax),
            ms(value),
            ms(fused_local),
            ms(output_projection),
        );
        eprintln!(
            "modernbert-profile: mlp total={:.2}ms norm={:.2}ms wi={:.2}ms act_mul={:.2}ms wo={:.2}ms",
            ms(mlp_total),
            ms(mlp_norm),
            ms(mlp_wi),
            ms(mlp_activation),
            ms(mlp_wo),
        );
        for layer in &self.layers {
            eprintln!(
                "modernbert-profile: layer {:02} {:>11} total={:.2}ms attn={:.2}ms mlp={:.2}ms qkv={:.2}ms attn_core={:.2}ms mlp_wi={:.2}ms mlp_wo={:.2}ms",
                layer.index,
                layer.attention_kind,
                ms(layer.total),
                ms(layer.attention_total),
                ms(layer.mlp_total),
                ms(layer.qkv),
                ms(layer.full_scores + layer.softmax + layer.value + layer.fused_local),
                ms(layer.mlp_wi),
                ms(layer.mlp_wo),
            );
        }
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn attention_kind(seq_len: usize, local_window: Option<usize>) -> &'static str {
    match local_window {
        None => "global",
        Some(window) if crate::fast_ops::should_use_modernbert_local_attention(seq_len, window) => {
            "local-fused"
        }
        Some(_) => "local-full",
    }
}

#[derive(Debug, Clone)]
struct Attention {
    wqkv: QMatMul,
    wo: QMatMul,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let wqkv = new_qmm(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("Wqkv"))?;
        #[cfg(feature = "hybrid-dequant")]
        let wo = new_qmm_dequant(cfg.hidden_size, cfg.hidden_size, vb.pp("Wo"))?;
        #[cfg(not(feature = "hybrid-dequant"))]
        let wo = new_qmm(cfg.hidden_size, cfg.hidden_size, vb.pp("Wo"))?;
        Ok(Self {
            wqkv,
            wo,
            n_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        local_window: Option<usize>,
        mut profile: Option<&mut LayerProfile>,
    ) -> Result<Tensor> {
        let attention_started = Instant::now();
        let (b, s, _) = xs.dims3()?;
        let started = Instant::now();
        let qkv = self.wqkv.forward(xs)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.qkv = started.elapsed();
        }

        let started = Instant::now();
        let qkv = qkv
            .reshape((b, s, 3, self.n_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let (q, k, v) = if matches!(qkv.device(), Device::Cpu) {
            let qkv = crate::fast_ops::modernbert_rope_qkv(&qkv, cos, sin)?;
            (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?)
        } else {
            (
                apply_rope(&qkv.get(0)?, cos, sin)?,
                apply_rope(&qkv.get(1)?, cos, sin)?,
                qkv.get(2)?,
            )
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.rope = started.elapsed();
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        if let Some(w) = local_window {
            if crate::fast_ops::should_use_modernbert_local_attention(s, w)
                && matches!(q.device(), Device::Cpu)
            {
                let started = Instant::now();
                let out = crate::fast_ops::modernbert_local_attention(&q, &k, &v, scale as f32, w)?;
                if let Some(profile) = profile.as_deref_mut() {
                    profile.fused_local = started.elapsed();
                }

                let started = Instant::now();
                let out = out.transpose(1, 2)?.contiguous()?.reshape((
                    b,
                    s,
                    self.n_heads * self.head_dim,
                ))?;
                let out = self.wo.forward(&out)?;
                if let Some(profile) = profile {
                    profile.output_projection = started.elapsed();
                    profile.attention_total = attention_started.elapsed();
                }
                return Ok(out);
            }
        }

        let started = Instant::now();
        let mut attn = (q.matmul(&k.t()?)? * scale)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.full_scores = started.elapsed();
        }

        if let Some(w) = local_window {
            if s > w {
                let started = Instant::now();
                let half_w = w / 2;
                let mask: Vec<f32> = (0..s)
                    .flat_map(|i| {
                        (0..s).map(move |j| {
                            if (i as isize - j as isize).unsigned_abs() <= half_w {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect();
                let mask = Tensor::new(mask.as_slice(), attn.device())?.reshape((1, 1, s, s))?;
                attn = attn.broadcast_add(&mask)?;
                if let Some(profile) = profile.as_deref_mut() {
                    profile.mask = started.elapsed();
                }
            }
        }

        let started = Instant::now();
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.softmax = started.elapsed();
        }

        let started = Instant::now();
        let out = attn.matmul(&v)?;
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.n_heads * self.head_dim))?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.value = started.elapsed();
        }

        let started = Instant::now();
        let out = self.wo.forward(&out)?;
        if let Some(profile) = profile {
            profile.output_projection = started.elapsed();
            profile.attention_total = attention_started.elapsed();
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    wi: QMatMul,
    wo: QMatMul,
    intermediate_size: usize,
    activation: Activation,
    fast_activation: crate::fast_ops::ModernBertActivation,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi = new_qmm(cfg.hidden_size, 2 * cfg.intermediate_size, vb.pp("Wi"))?;
        #[cfg(feature = "hybrid-dequant")]
        let wo = new_qmm_dequant(cfg.intermediate_size, cfg.hidden_size, vb.pp("Wo"))?;
        #[cfg(not(feature = "hybrid-dequant"))]
        let wo = new_qmm(cfg.intermediate_size, cfg.hidden_size, vb.pp("Wo"))?;
        let (activation, fast_activation) = match cfg.hidden_activation.as_str() {
            "silu" | "swish" => (
                Activation::Silu,
                crate::fast_ops::ModernBertActivation::Silu,
            ),
            _ => (
                Activation::Gelu,
                crate::fast_ops::ModernBertActivation::Gelu,
            ),
        };
        Ok(Self {
            wi,
            wo,
            intermediate_size: cfg.intermediate_size,
            activation,
            fast_activation,
        })
    }

    fn forward(&self, xs: &Tensor, mut profile: Option<&mut LayerProfile>) -> Result<Tensor> {
        let mlp_started = Instant::now();
        let started = Instant::now();
        let h = self.wi.forward(xs)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.mlp_wi = started.elapsed();
        }

        let started = Instant::now();
        let h = if matches!(h.device(), Device::Cpu) {
            crate::fast_ops::modernbert_gated_activation(
                &h,
                self.intermediate_size,
                self.fast_activation,
            )?
        } else {
            let gate = h.narrow(D::Minus1, 0, self.intermediate_size)?;
            let up = h.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
            self.activation.forward(&gate)?.broadcast_mul(&up)?
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.mlp_activation = started.elapsed();
        }

        let started = Instant::now();
        let out = self.wo.forward(&h)?;
        if let Some(profile) = profile {
            profile.mlp_wo = started.elapsed();
            profile.mlp_total = mlp_started.elapsed();
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attn_norm: Option<LayerNormNoBias>,
    attn: Attention,
    mlp_norm: LayerNormNoBias,
    mlp: Mlp,
}

impl Layer {
    fn load(i: usize, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn_norm = if i > 0 {
            Some(LayerNormNoBias::load(
                cfg.hidden_size,
                cfg.norm_eps,
                vb.pp("attn_norm"),
            )?)
        } else {
            None
        };
        let attn = Attention::load(vb.pp("attn"), cfg)?;
        let mlp_norm = LayerNormNoBias::load(cfg.hidden_size, cfg.norm_eps, vb.pp("mlp_norm"))?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        Ok(Self {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        local_window: Option<usize>,
        profile: Option<&mut LayerProfile>,
    ) -> Result<Tensor> {
        let layer_started = Instant::now();
        let mut profile = profile;
        let started = Instant::now();
        let normed = match &self.attn_norm {
            Some(norm) => candle_core::Module::forward(norm, xs)?,
            None => xs.clone(),
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.attn_norm = started.elapsed();
        }

        let xs = (xs
            + self
                .attn
                .forward(&normed, cos, sin, local_window, profile.as_deref_mut())?)?;

        let started = Instant::now();
        let normed = candle_core::Module::forward(&self.mlp_norm, &xs)?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.mlp_norm = started.elapsed();
        }

        let xs = (xs + self.mlp.forward(&normed, profile.as_deref_mut())?)?;
        if let Some(profile) = profile {
            profile.total = layer_started.elapsed();
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    embedding: candle_transformers::quantized_nn::Embedding,
    embedding_norm: LayerNormNoBias,
    layers: Vec<Layer>,
    final_norm: LayerNormNoBias,
    local_rope_cos: Tensor,
    local_rope_sin: Tensor,
    global_rope_cos: Tensor,
    global_rope_sin: Tensor,
    local_attention: usize,
    global_attn_every_n: usize,
}

impl Encoder {
    fn load(vb: VarBuilder, cfg: &Config, device: &Device) -> Result<Self> {
        let vb_enc = vb.pp("encoder");
        let embedding = candle_transformers::quantized_nn::Embedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_enc.pp("embeddings").pp("tok_embeddings"),
        )?;
        let embedding_norm = LayerNormNoBias::load(
            cfg.hidden_size,
            cfg.norm_eps,
            vb_enc.pp("embeddings").pp("norm"),
        )?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(i, vb_enc.pp("layers").pp(i.to_string()), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_norm =
            LayerNormNoBias::load(cfg.hidden_size, cfg.norm_eps, vb_enc.pp("final_norm"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let local_theta = cfg.rope_theta;
        let global_theta = cfg.global_rope_theta.unwrap_or(local_theta);
        let (local_rope_cos, local_rope_sin) =
            build_rope_cache(8192, head_dim, local_theta, device)?;
        let (global_rope_cos, global_rope_sin) = if global_theta == local_theta {
            (local_rope_cos.clone(), local_rope_sin.clone())
        } else {
            build_rope_cache(8192, head_dim, global_theta, device)?
        };
        Ok(Self {
            embedding,
            embedding_norm,
            layers,
            final_norm,
            local_rope_cos,
            local_rope_sin,
            global_rope_cos,
            global_rope_sin,
            local_attention: cfg.local_attention,
            global_attn_every_n: cfg.global_attn_every_n_layers,
        })
    }

    fn forward(&self, input_ids: &Tensor, profile: Option<&mut ForwardProfile>) -> Result<Tensor> {
        let mut profile = profile;
        let seq_len = input_ids.dim(D::Minus1)?;
        let started = Instant::now();
        let mut xs = candle_core::Module::forward(
            &self.embedding_norm,
            &self.embedding.forward(input_ids)?,
        )?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.embedding = started.elapsed();
        }

        let local_cos = self.local_rope_cos.narrow(0, 0, seq_len)?;
        let local_sin = self.local_rope_sin.narrow(0, 0, seq_len)?;
        let global_cos = self.global_rope_cos.narrow(0, 0, seq_len)?;
        let global_sin = self.global_rope_sin.narrow(0, 0, seq_len)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let is_global = self.global_attn_every_n > 0 && i % self.global_attn_every_n == 0;
            let (cos, sin, local_window) = if is_global {
                (&global_cos, &global_sin, None)
            } else {
                (&local_cos, &local_sin, Some(self.local_attention))
            };
            if let Some(profile) = profile.as_deref_mut() {
                let mut layer_profile = LayerProfile {
                    index: i,
                    attention_kind: attention_kind(seq_len, local_window),
                    ..LayerProfile::default()
                };
                xs = layer.forward(&xs, cos, sin, local_window, Some(&mut layer_profile))?;
                profile.layers.push(layer_profile);
            } else {
                xs = layer.forward(&xs, cos, sin, local_window, None)?;
            }
        }
        let started = Instant::now();
        let xs = candle_core::Module::forward(&self.final_norm, &xs)?;
        if let Some(profile) = profile {
            profile.final_norm = started.elapsed();
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
enum Projection {
    Linear(QMatMul, Tensor),
    Mlp {
        fc1: QMatMul,
        fc1_bias: Tensor,
        fc2: QMatMul,
        fc2_bias: Tensor,
    },
}

impl Projection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l, bias) => l.forward(xs)?.broadcast_add(bias),
            Self::Mlp {
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
            } => {
                let h = fc1.forward(xs)?.broadcast_add(fc1_bias)?;
                let h = Activation::Gelu.forward(&h)?;
                fc2.forward(&h)?.broadcast_add(fc2_bias)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: Encoder,
    projection: Projection,
    device: Device,
}

impl T5EncoderModel {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(D::Minus1)?;
        let mut profile = profile_take(seq_len).then(|| ForwardProfile::new(seq_len));
        let encoder_output = self.encoder.forward(input_ids, profile.as_mut())?;
        let started = Instant::now();
        let output = self.projection.forward(&encoder_output)?;
        if let Some(mut profile) = profile {
            profile.projection = started.elapsed();
            profile.print();
        }
        Ok(output)
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
        let cfg_bytes = CONFIG
            .bytes(assets)
            .map_err(|_| Error::other("failed to read modernbert-config.json"))?;
        let config: Config = serde_json::from_slice(cfg_bytes)
            .map_err(|e| Error::other(format!("failed to parse config: {e}")))?;
        let tok_bytes = TOKENIZER
            .bytes(assets)
            .map_err(|_| Error::other("failed to read modernbert-tokenizer.json"))?;
        let tokenizer = Tokenizer::from_bytes(tok_bytes)
            .map_err(|e| Error::other(format!("failed to parse tokenizer: {e}")))?;
        Ok((Self { config }, tokenizer))
    }

    pub fn build_encoder(
        &self,
        device: &Device,
        assets: &std::path::Path,
    ) -> candle_core::Result<T5EncoderModel> {
        let model_bytes = MODEL
            .bytes(assets)
            .map_err(|_| Error::other("failed to read modernbert.gguf"))?;
        let vb = VarBuilder::from_gguf_buffer(model_bytes, device)?;

        let projection = if let Some(mid) = self.config.projection_mlp {
            let fc1 = new_qmm(self.config.hidden_size, mid, vb.pp("linear").pp("0"))
                .map_err(|e| Error::other(format!("projection fc1: {e}")))?;
            let fc1_bias = vb
                .pp("linear")
                .pp("0")
                .get(mid, "bias")?
                .dequantize(device)?;
            let fc2 = new_qmm(mid, self.config.projection_dim, vb.pp("linear").pp("2"))
                .map_err(|e| Error::other(format!("projection fc2: {e}")))?;
            let fc2_bias = vb
                .pp("linear")
                .pp("2")
                .get(self.config.projection_dim, "bias")?
                .dequantize(device)?;
            Projection::Mlp {
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
            }
        } else {
            let w = new_qmm(
                self.config.hidden_size,
                self.config.projection_dim,
                vb.pp("linear"),
            )
            .map_err(|e| Error::other(format!("projection linear: {e}")))?;
            let bias = vb
                .pp("linear")
                .get(self.config.projection_dim, "bias")?
                .dequantize(device)?;
            Projection::Linear(w, bias)
        };

        let encoder = Encoder::load(vb, &self.config, device)
            .map_err(|e| Error::other(format!("failed to load encoder: {e}")))?;
        Ok(T5EncoderModel {
            encoder,
            projection,
            device: device.clone(),
        })
    }
}
