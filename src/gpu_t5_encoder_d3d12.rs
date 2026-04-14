//! Full T5 encoder on D3D12 (Windows GPU).
//!
//! Same architecture as the Metal GPU encoder: F32 residual stream, FP16 compute,
//! Q8 weights with per-column scales. All operations dispatched via Triton-compiled
//! DXIL compute shaders.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_d3d12_kernels::{Gpu, GpuBuffer};
use log::info;
use std::sync::Arc;

use crate::triton_d3d12_kernels::{
    T5D3D12Kernels,
    create_f16_buffer, create_f32_buffer, upload_bytes, download_bytes,
    d3d12_rms_norm_f32_f16, d3d12_matmul_q8, d3d12_gelu_mul,
    d3d12_residual_add_f16_f32, d3d12_flash_attention_bias,
};

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ── Q8 weight: int32-expanded int8 data + f16 per-column scale ──────────────

struct Q8Weight {
    data: GpuBuffer,   // [K, N] int32 (each i8 sign-extended to i32)
    scales: GpuBuffer, // [N] f16 per-column scales
}

struct AttentionWeights {
    w_q: Q8Weight,
    w_k: Q8Weight,
    w_v: Q8Weight,
    w_o: Q8Weight,
}

struct FfnWeights {
    w_gate: Q8Weight,
    w_up: Q8Weight,
    w_down: Q8Weight,
}

struct LayerWeights {
    attn_norm: GpuBuffer,
    attn: AttentionWeights,
    ffn_norm: GpuBuffer,
    ffn: FfnWeights,
}

struct Scratch {
    residual: GpuBuffer,    // [padded_seq, d_model] f32
    normed: GpuBuffer,      // [padded_seq, d_model] f16
    q: GpuBuffer,           // [padded_seq, inner_dim] f16
    k: GpuBuffer,
    v: GpuBuffer,
    attn_out: GpuBuffer,    // [padded_seq, inner_dim] f16
    o_proj: GpuBuffer,      // [padded_seq, d_model] f16
    gate: GpuBuffer,        // [padded_seq, d_ff] f16
    up: GpuBuffer,          // [padded_seq, d_ff] f16
    ffn_out: GpuBuffer,     // [padded_seq, d_model] f16
}

// ── Position bias (computed on CPU, uploaded as F32) ─────────────────────────

fn compute_position_bias(
    gpu: &Gpu,
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

    let buf = create_f32_buffer(gpu, bias_data.len())?;
    let bytes: Vec<u8> = bias_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload_bytes(gpu, &bytes, &buf)?;
    Ok(buf)
}

// ── Weight loading helpers ──────────────────────────────────────────────────

/// Dequantize 2D weight, transpose to [K, N], quantize to Q8.
/// For D3D12: 4 int8 values packed per int32 along N dimension.
/// HLSL kernel unpacks with shift/mask. 4x bandwidth reduction vs int32-expanded.
fn load_weight_2d_q8(
    gpu: &Gpu,
    shape: (usize, usize),
    vb: &QVarBuilder,
) -> Result<Q8Weight> {
    let qt = vb.get(shape, "weight")?;
    let t = qt.dequantize(&Device::Cpu)?;
    let t = t.t()?.contiguous()?;
    let (k, n) = (t.dim(0)?, t.dim(1)?);
    assert!(n % 4 == 0, "N={n} must be multiple of 4 for Q8 packing");
    let f32_data = t.flatten_all()?.to_vec1::<f32>()?;

    // Per-column absmax
    let mut col_absmax = vec![0.0f32; n];
    for row in 0..k {
        for col in 0..n {
            let v = f32_data[row * n + col].abs();
            if v > col_absmax[col] {
                col_absmax[col] = v;
            }
        }
    }

    // Scales: absmax / 127
    let scales_f16: Vec<half::f16> = col_absmax
        .iter()
        .map(|&m| half::f16::from_f32(m / 127.0))
        .collect();

    // Quantize to int8, pack 4 per int32 along N dimension
    let n_packed = n / 4;
    let mut q8_packed = vec![0u32; k * n_packed];
    for row in 0..k {
        for col in 0..n {
            let scale = col_absmax[col] / 127.0;
            let q = if scale > 0.0 {
                (f32_data[row * n + col] / scale).round().clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            let byte = q as u8; // 2's complement
            let packed_idx = row * n_packed + col / 4;
            let byte_pos = col % 4;
            q8_packed[packed_idx] |= (byte as u32) << (byte_pos * 8);
        }
    }

    // Upload packed int32 data (4 int8s per element, 4x smaller than before)
    let data = create_f32_buffer(gpu, k * n_packed)?;
    let data_bytes: Vec<u8> = q8_packed.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload_bytes(gpu, &data_bytes, &data)?;

    let scales = create_f16_buffer(gpu, n)?;
    let scale_bytes: Vec<u8> = scales_f16.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload_bytes(gpu, &scale_bytes, &scales)?;

    Ok(Q8Weight { data, scales })
}

/// Load 1D weight as F32 GpuBuffer.
fn load_weight_1d_f32(
    gpu: &Gpu,
    dim: usize,
    name: &str,
    vb: &QVarBuilder,
) -> Result<GpuBuffer> {
    let qt = vb.get(dim, name)?;
    let t = qt.dequantize(&Device::Cpu)?;
    let data = t.to_vec1::<f32>()?;
    let buf = create_f32_buffer(gpu, dim)?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload_bytes(gpu, &bytes, &buf)?;
    Ok(buf)
}

// ── Main GPU encoder ────────────────────────────────────────────────────────

pub struct GpuT5EncoderD3D12 {
    gpu: Arc<Gpu>,
    kernels: T5D3D12Kernels,
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

impl GpuT5EncoderD3D12 {
    pub fn new(
        cfg: &super::quantized_t5::Config,
        vb: QVarBuilder,
        max_seq_len: usize,
    ) -> Result<Self> {
        let gpu = Arc::new(Gpu::new(0)?);
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
            "GpuT5EncoderD3D12: d_model={d_model} d_kv={d_kv} d_ff={d_ff} n_heads={n_heads} \
             layers={} max_seq={max_seq_len} padded={padded_seq}",
            cfg.num_layers
        );

        let kernels = T5D3D12Kernels::load(&gpu)?;

        let scratch = Scratch {
            residual: create_f32_buffer(&gpu, padded_seq * d_model)?,
            normed: create_f16_buffer(&gpu, padded_seq * d_model)?,
            q: create_f16_buffer(&gpu, padded_seq * inner_dim)?,
            k: create_f16_buffer(&gpu, padded_seq * inner_dim)?,
            v: create_f16_buffer(&gpu, padded_seq * inner_dim)?,
            attn_out: create_f16_buffer(&gpu, padded_seq * inner_dim)?,
            o_proj: create_f16_buffer(&gpu, padded_seq * d_model)?,
            gate: create_f16_buffer(&gpu, padded_seq * d_ff)?,
            up: create_f16_buffer(&gpu, padded_seq * d_ff)?,
            ffn_out: create_f16_buffer(&gpu, padded_seq * d_model)?,
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let enc_vb = vb.pp("encoder");

        for i in 0..cfg.num_layers {
            let block_vb = enc_vb.pp(format!("block.{i}"));
            let sa_vb = block_vb.pp("layer.0");
            let ff_vb = block_vb.pp("layer.1");

            let attn_norm = load_weight_1d_f32(&gpu, d_model, "weight", &sa_vb.pp("layer_norm"))?;

            let attn_vb = sa_vb.pp("SelfAttention");
            let attn = AttentionWeights {
                w_q: load_weight_2d_q8(&gpu, (inner_dim, d_model), &attn_vb.pp("q"))?,
                w_k: load_weight_2d_q8(&gpu, (inner_dim, d_model), &attn_vb.pp("k"))?,
                w_v: load_weight_2d_q8(&gpu, (inner_dim, d_model), &attn_vb.pp("v"))?,
                w_o: load_weight_2d_q8(&gpu, (d_model, inner_dim), &attn_vb.pp("o"))?,
            };

            let ffn_norm = load_weight_1d_f32(&gpu, d_model, "weight", &ff_vb.pp("layer_norm"))?;

            let ffn_vb = ff_vb.pp("DenseReluDense");
            let ffn = if cfg.feed_forward_proj.gated {
                FfnWeights {
                    w_gate: load_weight_2d_q8(&gpu, (d_ff, d_model), &ffn_vb.pp("wi_0"))?,
                    w_up: load_weight_2d_q8(&gpu, (d_ff, d_model), &ffn_vb.pp("wi_1"))?,
                    w_down: load_weight_2d_q8(&gpu, (d_model, d_ff), &ffn_vb.pp("wo"))?,
                }
            } else {
                // Non-gated: single wi, dummy w_up
                let dummy = Q8Weight {
                    data: create_f32_buffer(&gpu, 1)?,
                    scales: create_f16_buffer(&gpu, 1)?,
                };
                FfnWeights {
                    w_gate: load_weight_2d_q8(&gpu, (d_ff, d_model), &ffn_vb.pp("wi"))?,
                    w_up: dummy,
                    w_down: load_weight_2d_q8(&gpu, (d_model, d_ff), &ffn_vb.pp("wo"))?,
                }
            };

            layers.push(LayerWeights { attn_norm, attn, ffn_norm, ffn });
        }

        let final_norm_weight =
            load_weight_1d_f32(&gpu, d_model, "weight", &enc_vb.pp("final_layer_norm"))?;

        let bias_vb = enc_vb
            .pp("block.0").pp("layer.0").pp("SelfAttention")
            .pp("relative_attention_bias");
        let bias_qt = bias_vb.get((num_buckets, n_heads), "weight")?;
        let bias_t = bias_qt.dequantize(&Device::Cpu)?;
        let bias_embeddings = bias_t.flatten_all()?.to_vec1::<f32>()?;

        Ok(Self {
            gpu,
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
            &self.gpu, &self.bias_embeddings,
            self.n_heads, seq_len, self.num_buckets, self.max_distance,
        )?;
        *self.bias_cache.borrow_mut() = Some((seq_len, bias));
        Ok(std::cell::Ref::map(self.bias_cache.borrow(), |c| {
            &c.as_ref().unwrap().1
        }))
    }

    /// Run full T5 encoder forward pass on D3D12 GPU.
    /// Input: [1, seq_len, d_model] F32 on CPU.
    /// Output: [1, seq_len, d_model] F32 on CPU.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let (batch, seq_len, dim) = x.dims3()?;
        assert_eq!(batch, 1, "GpuT5EncoderD3D12 only supports batch=1");
        assert_eq!(dim, self.d_model);

        let block_m = 64;
        let padded_seq = cdiv(seq_len, block_m) * block_m;
        assert!(padded_seq <= self.padded_seq);

        let s = &self.scratch;
        let n_elem = padded_seq * self.d_model;

        // Upload input: CPU F32 → GPU F32 residual
        let x_flat = x.reshape(seq_len * dim)?;
        let data = x_flat
            .to_dtype(candle_core::DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;

        // Pad to padded_seq and upload
        let mut padded_data = vec![0.0f32; padded_seq * dim];
        padded_data[..data.len()].copy_from_slice(&data);
        let input_bytes: Vec<u8> = padded_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_bytes(&self.gpu, &input_bytes, &s.residual)?;
        let t_upload = t0.elapsed();

        let bias = self.get_bias(seq_len)?;
        let t_bias = t0.elapsed();

        // Single batched command list for the entire forward pass — one GPU submission.
        let gpu = &self.kernels.gpu;
        gpu.begin_batch()
            .map_err(|e| anyhow::anyhow!("begin_batch: {e}"))?;

        macro_rules! barrier {
            () => { gpu.record_uav_barrier(); };
        }

        for layer in self.layers.iter() {
            // ── Self-attention ──
            d3d12_rms_norm_f32_f16(
                &self.kernels, &s.residual, &layer.attn_norm, &s.normed,
                padded_seq, self.d_model, self.eps,
            )?;
            barrier!();

            // Q/K/V projections (Q8) — all read normed, write to separate buffers
            d3d12_matmul_q8(
                &self.kernels, &s.normed, &layer.attn.w_q.data, &s.q,
                &layer.attn.w_q.scales,
                padded_seq, self.inner_dim, self.d_model,
            )?;
            d3d12_matmul_q8(
                &self.kernels, &s.normed, &layer.attn.w_k.data, &s.k,
                &layer.attn.w_k.scales,
                padded_seq, self.inner_dim, self.d_model,
            )?;
            d3d12_matmul_q8(
                &self.kernels, &s.normed, &layer.attn.w_v.data, &s.v,
                &layer.attn.w_v.scales,
                padded_seq, self.inner_dim, self.d_model,
            )?;
            barrier!();

            // Flash attention with position bias
            let stride_h = self.d_kv as i32;
            let stride_m = self.inner_dim as i32;
            let stride_o = self.inner_dim as i32;
            let sm_scale = 1.0 / (self.d_kv as f32).sqrt();
            d3d12_flash_attention_bias(
                &self.kernels, &s.q, &s.k, &s.v, &s.attn_out, &bias,
                self.n_heads, seq_len, self.d_kv,
                stride_h, stride_m, stride_o, sm_scale,
            )?;
            barrier!();

            // O projection (Q8)
            d3d12_matmul_q8(
                &self.kernels, &s.attn_out, &layer.attn.w_o.data, &s.o_proj,
                &layer.attn.w_o.scales,
                padded_seq, self.d_model, self.inner_dim,
            )?;
            barrier!();

            // F16 o_proj + F32 residual → F32 residual
            d3d12_residual_add_f16_f32(
                &self.kernels, &s.o_proj, &s.residual, &s.residual, n_elem,
            )?;
            barrier!();

            // ── FFN ──
            d3d12_rms_norm_f32_f16(
                &self.kernels, &s.residual, &layer.ffn_norm, &s.normed,
                padded_seq, self.d_model, self.eps,
            )?;
            barrier!();

            // Gate and up projections (Q8) — both read normed, write to separate buffers
            d3d12_matmul_q8(
                &self.kernels, &s.normed, &layer.ffn.w_gate.data, &s.gate,
                &layer.ffn.w_gate.scales,
                padded_seq, self.d_ff, self.d_model,
            )?;
            d3d12_matmul_q8(
                &self.kernels, &s.normed, &layer.ffn.w_up.data, &s.up,
                &layer.ffn.w_up.scales,
                padded_seq, self.d_ff, self.d_model,
            )?;
            barrier!();

            // Fused gated-GELU
            d3d12_gelu_mul(&self.kernels, &s.gate, &s.up, &s.gate, padded_seq * self.d_ff)?;
            barrier!();

            // Down projection (Q8)
            d3d12_matmul_q8(
                &self.kernels, &s.gate, &layer.ffn.w_down.data, &s.ffn_out,
                &layer.ffn.w_down.scales,
                padded_seq, self.d_model, self.d_ff,
            )?;
            barrier!();

            // F16 ffn_out + F32 residual → F32 residual
            d3d12_residual_add_f16_f32(
                &self.kernels, &s.ffn_out, &s.residual, &s.residual, n_elem,
            )?;
            barrier!();
        }

        // Final RMS norm: F32 → F16
        d3d12_rms_norm_f32_f16(
            &self.kernels, &s.residual, &self.final_norm_weight, &s.normed,
            padded_seq, self.d_model, self.eps,
        )?;

        // Execute the entire batched command list — single GPU submission + wait
        gpu.end_batch()
            .map_err(|e| anyhow::anyhow!("end_batch: {e}"))?;
        let t_gpu = t0.elapsed();

        // Download F16 result → CPU F32
        let result_bytes = download_bytes(&self.gpu, &s.normed, (padded_seq * self.d_model * 2) as u64)?;
        let f16_data: Vec<half::f16> = result_bytes.chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
            .collect();
        let f32_data: Vec<f32> = f16_data[..seq_len * self.d_model]
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let t_total = t0.elapsed();

        info!(
            "D3D12 GPU forward: seq={seq_len} pad={padded_seq} upload={:.1}ms bias={:.1}ms \
             gpu={:.1}ms download={:.1}ms total={:.1}ms",
            t_upload.as_secs_f64() * 1000.0,
            (t_bias - t_upload).as_secs_f64() * 1000.0,
            (t_gpu - t_bias).as_secs_f64() * 1000.0,
            (t_total - t_gpu).as_secs_f64() * 1000.0,
            t_total.as_secs_f64() * 1000.0,
        );

        Ok(Tensor::from_vec(f32_data, (1, seq_len, self.d_model), &Device::Cpu)?)
    }
}
