//! Full T5 encoder: all ops on GPU, F32 residuals, FP16 compute.
//!
//! Bypasses candle's Metal backend (which fails on Intel GPUs). All operations
//! dispatch through pre-compiled .metallib kernels using raw GpuBuffer dispatch.
//!
//! Residual stream stays in F32 on GPU to prevent FP16 overflow.
//! Matmul, attention, and gelu run in FP16. Two mixed-precision bridge kernels
//! (rms_norm_f32_f16 and residual_add_f16_f32) handle the conversions.

use anyhow::Result;
use candle_core::{DType, Device, GpuBuffer, MetalDevice, Tensor};
use candle_metal_kernels::metal::{
    CommandQueue, CommandSemaphore, ComputeCommandEncoder,
};
use log::info;
use std::sync::Arc;

use crate::triton_kernels::{
    enc_flash_attention_bias, enc_gelu_mul, enc_matmul_q8, enc_residual_add_f16_f32,
    enc_rms_norm_f32_f16, T5Kernels,
};

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ── Q8 weight: int8 data + f16 per-column scale ─────────────────────────────

struct Q8Weight {
    data: GpuBuffer,   // [K, N] int8 row-major
    scales: GpuBuffer, // [N] f16 per-column scales
}

// ── Per-layer weights (matmul weights Q8 on GPU, norm weights F32 on GPU) ───

struct AttentionWeights {
    w_q: Q8Weight, // [d_model, inner_dim]
    w_k: Q8Weight,
    w_v: Q8Weight,
    w_o: Q8Weight, // [inner_dim, d_model]
}

struct FfnWeights {
    w_gate: Q8Weight, // wi_0: [d_model, d_ff]
    w_up: Q8Weight,   // wi_1: [d_model, d_ff]
    w_down: Q8Weight, // wo:   [d_ff, d_model]
}

struct LayerWeights {
    attn_norm: GpuBuffer, // [d_model] f32 on GPU
    attn: AttentionWeights,
    ffn_norm: GpuBuffer, // [d_model] f32 on GPU
    ffn: FfnWeights,
}

// ── Scratch buffers ──────────────────────────────────────────────────────────

struct Scratch {
    residual: GpuBuffer,    // [padded_seq, d_model] f32 — residual stream
    normed: GpuBuffer,      // [padded_seq, d_model] f16 — rms_norm output
    q: GpuBuffer,           // [padded_seq, inner_dim] f16
    k: GpuBuffer,
    v: GpuBuffer,
    attn_out: GpuBuffer,    // [padded_seq, inner_dim] f16
    o_proj: GpuBuffer,      // [padded_seq, d_model] f16
    gate: GpuBuffer,        // [padded_seq, d_ff] f16
    up: GpuBuffer,          // [padded_seq, d_ff] f16
    ffn_out: GpuBuffer,     // [padded_seq, d_model] f16
}

// ── Position bias ────────────────────────────────────────────────────────────

fn compute_position_bias(
    device: &MetalDevice,
    bias_embeddings: &[f32],
    n_heads: usize,
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<GpuBuffer> {
    let half_buckets = num_buckets / 2;
    let max_exact = half_buckets / 2;

    let mut bias_data = vec![0.0f32; n_heads * seq_len * seq_len];

    for qi in 0..seq_len {
        for ki in 0..seq_len {
            let bucket = if qi >= ki {
                let delta = qi - ki;
                if delta < max_exact {
                    delta
                } else {
                    let b = f32::log(
                        delta as f32 / max_exact as f32,
                        max_distance as f32 / max_exact as f32,
                    ) * (half_buckets - max_exact) as f32;
                    max_exact + (b as usize).min(half_buckets - 1 - max_exact)
                }
            } else {
                let delta = ki - qi;
                if delta < max_exact {
                    delta + half_buckets
                } else {
                    let b = f32::log(
                        delta as f32 / max_exact as f32,
                        max_distance as f32 / max_exact as f32,
                    ) * (half_buckets - max_exact) as f32;
                    half_buckets + max_exact + (b as usize).min(half_buckets - 1 - max_exact)
                }
            };

            for h in 0..n_heads {
                bias_data[h * seq_len * seq_len + qi * seq_len + ki] =
                    bias_embeddings[bucket * n_heads + h];
            }
        }
    }

    Ok(GpuBuffer::from_f32_data(device, &bias_data)?)
}

// ── Weight loading helpers ───────────────────────────────────────────────────

/// Dequantize 2D weight from GGUF, transpose to [K, N] row-major, quantize to Q8.
/// Returns int8 data buffer + f16 per-column scales buffer.
fn load_weight_2d_q8(
    device: &MetalDevice,
    shape: (usize, usize),
    vb: &QVarBuilder,
) -> Result<Q8Weight> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    // shape is (N, K) from GGUF, transpose to [K, N] row-major
    let t = t.t()?.contiguous()?;
    let (k, n) = (t.dim(0)?, t.dim(1)?);
    let f32_data = t.flatten_all()?.to_vec1::<f32>()?;

    // Compute per-column absmax for quantization
    let mut col_absmax = vec![0.0f32; n];
    for row in 0..k {
        for col in 0..n {
            let v = f32_data[row * n + col].abs();
            if v > col_absmax[col] {
                col_absmax[col] = v;
            }
        }
    }

    // Scale = absmax / 127, quantized = round(val / scale)
    let scales_f16: Vec<half::f16> = col_absmax
        .iter()
        .map(|&m| half::f16::from_f32(m / 127.0))
        .collect();

    let mut q8_data = vec![0i8; k * n];
    for row in 0..k {
        for col in 0..n {
            let scale = col_absmax[col] / 127.0;
            let q = if scale > 0.0 {
                (f32_data[row * n + col] / scale).round().clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            q8_data[row * n + col] = q;
        }
    }

    let data = GpuBuffer::from_i8_data(device, &q8_data)?;
    let scales = GpuBuffer::from_f16_data(device, &scales_f16)?;
    Ok(Q8Weight { data, scales })
}

/// Load 1D weight as F32 GpuBuffer (for GPU-side rms_norm).
fn load_weight_1d_f32(
    device: &MetalDevice,
    dim: usize,
    name: &str,
    vb: &QVarBuilder,
) -> Result<GpuBuffer> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    let data = t.to_vec1::<f32>()?;
    Ok(GpuBuffer::from_f32_data(device, &data)?)
}

// ── Main GPU encoder ─────────────────────────────────────────────────────────

pub struct GpuT5Encoder {
    device: MetalDevice,
    queue: CommandQueue,
    kernels: T5Kernels,
    layers: Vec<LayerWeights>,
    final_norm_weight: GpuBuffer,
    bias_embeddings: Vec<f32>,
    bias_cache: std::cell::RefCell<Option<(usize, GpuBuffer)>>,
    d_model: usize,
    d_ff: usize,
    d_kv: usize,
    n_heads: usize,
    inner_dim: usize,
    num_buckets: usize,
    max_distance: usize,
    eps: f32,
    scratch: Scratch,
    padded_seq: usize,
}

impl GpuT5Encoder {
    pub fn new(
        device: &MetalDevice,
        cfg: &super::quantized_t5::Config,
        vb: QVarBuilder,
        max_seq_len: usize,
    ) -> Result<Self> {
        let d_model = cfg.d_model;
        let d_kv = cfg.d_kv;
        let d_ff = cfg.d_ff;
        let n_heads = cfg.num_heads;
        let inner_dim = n_heads * d_kv;
        let num_buckets = cfg.relative_attention_num_buckets;
        let max_distance = cfg.relative_attention_max_distance;
        let eps = cfg.layer_norm_epsilon as f32;

        let block_m = 64;
        let padded_seq = cdiv(max_seq_len, block_m) * block_m;

        info!(
            "GpuT5Encoder: d_model={d_model} d_kv={d_kv} d_ff={d_ff} n_heads={n_heads} \
             layers={} max_seq={max_seq_len} padded={padded_seq}",
            cfg.num_layers
        );

        let kernels = T5Kernels::load(device)?;

        let scratch = Scratch {
            residual: GpuBuffer::alloc_f32(device, padded_seq * d_model)?,
            normed: GpuBuffer::alloc_f16(device, padded_seq * d_model)?,
            q: GpuBuffer::alloc_f16(device, padded_seq * inner_dim)?,
            k: GpuBuffer::alloc_f16(device, padded_seq * inner_dim)?,
            v: GpuBuffer::alloc_f16(device, padded_seq * inner_dim)?,
            attn_out: GpuBuffer::alloc_f16(device, padded_seq * inner_dim)?,
            o_proj: GpuBuffer::alloc_f16(device, padded_seq * d_model)?,
            gate: GpuBuffer::alloc_f16(device, padded_seq * d_ff)?,
            up: GpuBuffer::alloc_f16(device, padded_seq * d_ff)?,
            ffn_out: GpuBuffer::alloc_f16(device, padded_seq * d_model)?,
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let enc_vb = vb.pp("encoder");

        for i in 0..cfg.num_layers {
            let block_vb = enc_vb.pp(format!("block.{i}"));
            let sa_vb = block_vb.pp("layer.0");
            let ff_vb = block_vb.pp(if cfg.feed_forward_proj.gated {
                "layer.1"
            } else {
                "layer.1"
            });

            let attn_norm =
                load_weight_1d_f32(device, d_model, "weight", &sa_vb.pp("layer_norm"))?;

            let attn_vb = sa_vb.pp("SelfAttention");
            let attn = AttentionWeights {
                w_q: load_weight_2d_q8(device, (inner_dim, d_model), &attn_vb.pp("q"))?,
                w_k: load_weight_2d_q8(device, (inner_dim, d_model), &attn_vb.pp("k"))?,
                w_v: load_weight_2d_q8(device, (inner_dim, d_model), &attn_vb.pp("v"))?,
                w_o: load_weight_2d_q8(device, (d_model, inner_dim), &attn_vb.pp("o"))?,
            };

            let ffn_norm =
                load_weight_1d_f32(device, d_model, "weight", &ff_vb.pp("layer_norm"))?;

            let ffn_vb = ff_vb.pp("DenseReluDense");
            let ffn = if cfg.feed_forward_proj.gated {
                FfnWeights {
                    w_gate: load_weight_2d_q8(device, (d_ff, d_model), &ffn_vb.pp("wi_0"))?,
                    w_up: load_weight_2d_q8(device, (d_ff, d_model), &ffn_vb.pp("wi_1"))?,
                    w_down: load_weight_2d_q8(device, (d_model, d_ff), &ffn_vb.pp("wo"))?,
                }
            } else {
                FfnWeights {
                    w_gate: load_weight_2d_q8(device, (d_ff, d_model), &ffn_vb.pp("wi"))?,
                    w_up: Q8Weight {
                        data: GpuBuffer::alloc_i8(device, 1)?,
                        scales: GpuBuffer::alloc_f16(device, 1)?,
                    },
                    w_down: load_weight_2d_q8(device, (d_model, d_ff), &ffn_vb.pp("wo"))?,
                }
            };

            layers.push(LayerWeights {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        let final_norm_weight =
            load_weight_1d_f32(device, d_model, "weight", &enc_vb.pp("final_layer_norm"))?;

        let bias_vb = enc_vb
            .pp("block.0")
            .pp("layer.0")
            .pp("SelfAttention")
            .pp("relative_attention_bias");
        let bias_qt = bias_vb.get((num_buckets, n_heads), "weight")?;
        let bias_t = bias_qt.dequantize(&Device::Cpu)?;
        let bias_embeddings = bias_t.flatten_all()?.to_vec1::<f32>()?;

        // Own command queue — bypasses candle's pool for single-encoder forward pass
        let queue = device.device().new_command_queue()?;

        Ok(Self {
            device: device.clone(),
            queue,
            kernels,
            layers,
            final_norm_weight,
            bias_embeddings,
            bias_cache: std::cell::RefCell::new(None),
            d_model,
            d_ff,
            d_kv,
            n_heads,
            inner_dim,
            num_buckets,
            max_distance,
            eps,
            scratch,
            padded_seq,
        })
    }

    /// Create a single command buffer + encoder for the entire forward pass.
    fn begin_pass(&self) -> Result<(candle_metal_kernels::metal::CommandBuffer, ComputeCommandEncoder)> {
        let sem = Arc::new(CommandSemaphore::new());
        let cb = candle_metal_kernels::metal::create_command_buffer(&self.queue, sem)?;
        let enc = cb.compute_command_encoder();
        Ok((cb, enc))
    }

    fn get_bias(&self, seq_len: usize) -> Result<std::cell::Ref<'_, GpuBuffer>> {
        {
            let cache = self.bias_cache.borrow();
            if let Some((cached_len, _)) = &*cache {
                if *cached_len == seq_len {
                    return Ok(std::cell::Ref::map(cache, |c| &c.as_ref().unwrap().1));
                }
            }
        }
        let bias = compute_position_bias(
            &self.device,
            &self.bias_embeddings,
            self.n_heads,
            seq_len,
            self.num_buckets,
            self.max_distance,
        )?;
        *self.bias_cache.borrow_mut() = Some((seq_len, bias));
        Ok(std::cell::Ref::map(self.bias_cache.borrow(), |c| {
            &c.as_ref().unwrap().1
        }))
    }

    /// Run full T5 encoder forward pass on GPU.
    /// Input: [1, seq_len, d_model] F32 on CPU (from embedding lookup).
    /// Output: [1, seq_len, d_model] F32 on CPU.
    ///
    /// Residual stream stays in F32 on GPU. RMS norm bridges F32→F16 for matmul,
    /// residual add bridges F16→F32 back. Zero CPU sync until final readback.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "GpuT5Encoder only supports batch=1");
        assert_eq!(dim, self.d_model);

        let block_m = 64;
        let padded_seq = cdiv(seq_len, block_m) * block_m;
        assert!(
            padded_seq <= self.padded_seq,
            "seq_len {} (padded {}) exceeds pre-allocated {}",
            seq_len,
            padded_seq,
            self.padded_seq
        );

        let s = &self.scratch;
        let n_elem = padded_seq * self.d_model;

        // Upload input: CPU F32 → GPU F32 residual
        let x_flat = x.reshape(seq_len * dim)?;
        let data = x_flat
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        unsafe {
            let ptr = s.residual.contents_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            // Zero padding
            for i in data.len()..(padded_seq * dim) {
                *ptr.add(i) = 0.0;
            }
        }
        let t_upload = t0.elapsed();

        let bias = self.get_bias(seq_len)?;
        let t_bias = t0.elapsed();

        // Single command buffer + encoder for entire forward pass
        let (cb, enc) = self.begin_pass()?;

        for layer in self.layers.iter() {
            // ── Self-attention ──
            // F32 residual → F16 normed (GPU)
            enc_rms_norm_f32_f16(
                &enc,
                &self.kernels.rms_norm_f32_f16,
                &s.residual,
                &layer.attn_norm,
                &s.normed,
                padded_seq,
                self.d_model,
                self.eps,
            );

            // Q/K/V projections (Q8)
            let pipeline = &self.kernels.matmul_q8_64x64;
            enc_matmul_q8(
                &enc, pipeline, &s.normed, &layer.attn.w_q.data, &s.q,
                &layer.attn.w_q.scales,
                padded_seq, self.inner_dim, self.d_model, 64, 64,
            );
            enc_matmul_q8(
                &enc, pipeline, &s.normed, &layer.attn.w_k.data, &s.k,
                &layer.attn.w_k.scales,
                padded_seq, self.inner_dim, self.d_model, 64, 64,
            );
            enc_matmul_q8(
                &enc, pipeline, &s.normed, &layer.attn.w_v.data, &s.v,
                &layer.attn.w_v.scales,
                padded_seq, self.inner_dim, self.d_model, 64, 64,
            );

            // Flash attention with position bias
            let stride_h = self.d_kv as i32;
            let stride_m = self.inner_dim as i32;
            let stride_o = self.inner_dim as i32;
            let sm_scale = 1.0 / (self.d_kv as f32).sqrt();
            enc_flash_attention_bias(
                &enc,
                &self.kernels.flash_attention_bias,
                &s.q, &s.k, &s.v, &s.attn_out, &bias,
                self.n_heads, seq_len, self.d_kv,
                stride_h, stride_m, stride_o, sm_scale,
            );

            // O projection (Q8)
            enc_matmul_q8(
                &enc, pipeline, &s.attn_out, &layer.attn.w_o.data, &s.o_proj,
                &layer.attn.w_o.scales,
                padded_seq, self.d_model, self.inner_dim, 64, 64,
            );

            // F16 o_proj + F32 residual → F32 residual (in-place)
            enc_residual_add_f16_f32(
                &enc, &self.kernels.residual_add_f16_f32,
                &s.o_proj, &s.residual, &s.residual, n_elem,
            );

            // ── FFN ──
            // F32 residual → F16 normed (GPU)
            enc_rms_norm_f32_f16(
                &enc,
                &self.kernels.rms_norm_f32_f16,
                &s.residual,
                &layer.ffn_norm,
                &s.normed,
                padded_seq,
                self.d_model,
                self.eps,
            );

            // Gate and up projections (Q8)
            enc_matmul_q8(
                &enc, pipeline, &s.normed, &layer.ffn.w_gate.data, &s.gate,
                &layer.ffn.w_gate.scales,
                padded_seq, self.d_ff, self.d_model, 64, 64,
            );
            enc_matmul_q8(
                &enc, pipeline, &s.normed, &layer.ffn.w_up.data, &s.up,
                &layer.ffn.w_up.scales,
                padded_seq, self.d_ff, self.d_model, 64, 64,
            );

            // Fused gated-GELU (F16)
            enc_gelu_mul(
                &enc, &self.kernels.gelu_mul,
                &s.gate, &s.up, &s.gate,
                padded_seq * self.d_ff,
            );

            // Down projection (Q8)
            enc_matmul_q8(
                &enc, pipeline, &s.gate, &layer.ffn.w_down.data, &s.ffn_out,
                &layer.ffn.w_down.scales,
                padded_seq, self.d_model, self.d_ff, 64, 64,
            );

            // F16 ffn_out + F32 residual → F32 residual (in-place)
            enc_residual_add_f16_f32(
                &enc, &self.kernels.residual_add_f16_f32,
                &s.ffn_out, &s.residual, &s.residual, n_elem,
            );
        }

        // Final RMS norm: F32 → F16 → normed scratch
        enc_rms_norm_f32_f16(
            &enc,
            &self.kernels.rms_norm_f32_f16,
            &s.residual,
            &self.final_norm_weight,
            &s.normed,
            padded_seq,
            self.d_model,
            self.eps,
        );

        // End encoder, commit, wait — single GPU sync for entire forward pass
        let t_encode = t0.elapsed();
        drop(enc);
        cb.commit();
        cb.wait_until_completed();
        let t_gpu = t0.elapsed();

        // Download F16 result → CPU F32
        let ptr = s.normed.contents_ptr() as *const half::f16;
        let f16_data = unsafe { std::slice::from_raw_parts(ptr, padded_seq * self.d_model) };
        let f32_data: Vec<f32> = f16_data[..seq_len * self.d_model]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let t_total = t0.elapsed();
        info!(
            "GPU forward: seq={seq_len} pad={padded_seq} upload={:.1}ms bias={:.1}ms encode={:.1}ms gpu={:.1}ms download={:.1}ms total={:.1}ms",
            t_upload.as_secs_f64() * 1000.0,
            (t_bias - t_upload).as_secs_f64() * 1000.0,
            (t_encode - t_bias).as_secs_f64() * 1000.0,
            (t_gpu - t_encode).as_secs_f64() * 1000.0,
            (t_total - t_gpu).as_secs_f64() * 1000.0,
            t_total.as_secs_f64() * 1000.0,
        );
        Ok(Tensor::from_vec(
            f32_data,
            (1, seq_len, self.d_model),
            &Device::Cpu,
        )?)
    }
}
