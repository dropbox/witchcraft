//! Full quantized ModernBERT encoder dispatched through Neso AOT Metal kernels.

use anyhow::{Result, ensure};
use candle_core::{DType, Device, GpuBuffer, MetalDevice, Tensor};
use log::info;
use std::cell::{Ref, RefCell};

use crate::neso_metal_kernels::{
    NesoKernels, enc_copy_f32_f16, enc_flash_attention_bias, enc_gelu_mul, enc_layer_norm_f32_f16, enc_matmul,
    enc_residual_add_f16_f32, enc_rope_inplace,
};

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;
const DEFAULT_SCRATCH_MIB: usize = 96;
const DEFAULT_MAX_BATCH_SIZE: usize = 8;

fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

struct Weight {
    data: GpuBuffer,
}

struct AttentionWeights {
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
}

struct MlpWeights {
    gate: Weight,
    up: Weight,
    out: Weight,
}

struct LayerWeights {
    attn_norm: Option<GpuBuffer>,
    attn: AttentionWeights,
    mlp_norm: GpuBuffer,
    mlp: MlpWeights,
}

struct Scratch {
    residual: GpuBuffer,
    normed: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    attn_out: GpuBuffer,
    projection: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    mlp_out: GpuBuffer,
}

fn upload_weight(device: &MetalDevice, rows: &[f32]) -> Result<Weight> {
    let data: Vec<half::f16> = rows.iter().map(|value| half::f16::from_f32(*value)).collect();
    Ok(Weight {
        data: GpuBuffer::from_f16_data(device, &data)?,
    })
}

fn load_weight(device: &MetalDevice, vb: &QVarBuilder, out: usize, input: usize) -> Result<Weight> {
    let tensor = vb
        .get((out, input), "weight")?
        .dequantize(&Device::Cpu)?
        .t()?
        .contiguous()?;
    let rows = tensor.flatten_all()?.to_vec1::<f32>()?;
    upload_weight(device, &rows)
}

fn load_weight_slice(
    device: &MetalDevice,
    vb: &QVarBuilder,
    total_out: usize,
    input: usize,
    start: usize,
    len: usize,
) -> Result<Weight> {
    let tensor = vb
        .get((total_out, input), "weight")?
        .dequantize(&Device::Cpu)?
        .t()?
        .contiguous()?;
    let all = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut selected = Vec::with_capacity(input * len);
    for row in all.chunks_exact(total_out) {
        selected.extend_from_slice(&row[start..start + len]);
    }
    upload_weight(device, &selected)
}

fn load_norm(device: &MetalDevice, vb: &QVarBuilder, dim: usize) -> Result<GpuBuffer> {
    let data = vb.get(dim, "weight")?.dequantize(&Device::Cpu)?.to_vec1::<f32>()?;
    Ok(GpuBuffer::from_f32_data(device, &data)?)
}

// `GpuBuffer::from_f32_data` registers every allocation in Candle's reusable
// buffer pool. Rope tables and attention masks are replaced whenever the
// padded sequence length changes, so these short-lived buffers should bypass
// that pool.
fn transient_f32_buffer(device: &MetalDevice, data: &[f32]) -> Result<GpuBuffer> {
    let buffer = GpuBuffer::alloc_shared_f32(device, data.len())?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buffer.contents_ptr() as *mut f32,
            data.len(),
        );
    }
    Ok(buffer)
}

fn make_rope(device: &MetalDevice, seq_len: usize, head_dim: usize, theta: f64) -> Result<(GpuBuffer, GpuBuffer)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);
    for position in 0..seq_len {
        for i in 0..half {
            let angle = position as f64 / theta.powf((2 * i) as f64 / head_dim as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        transient_f32_buffer(device, &cos)?,
        transient_f32_buffer(device, &sin)?,
    ))
}

fn make_local_mask(device: &MetalDevice, seq_len: usize, local_window: usize) -> Result<GpuBuffer> {
    let half_window = local_window / 2;
    let values = (0..seq_len)
        .flat_map(|query| {
            (0..seq_len).map(move |key| {
                if query.abs_diff(key) <= half_window {
                    0.0
                } else {
                    -1.0e9
                }
            })
        })
        .collect::<Vec<_>>();
    transient_f32_buffer(device, &values)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn scratch_rows(max_seq_len: usize, hidden: usize, intermediate: usize) -> usize {
    let bytes_per_row = hidden * (4 + 7 * 2) + intermediate * (2 * 2);
    let budget = env_usize("WITCHCRAFT_NESO_SCRATCH_MB", DEFAULT_SCRATCH_MIB) * 1024 * 1024;
    let max_batch = env_usize("WITCHCRAFT_NESO_MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE);
    let budget_rows = budget / bytes_per_row;
    let rows = budget_rows.max(max_seq_len).min(max_seq_len.saturating_mul(max_batch));
    rows / 64 * 64
}

pub struct GpuModernBertMetal {
    device: MetalDevice,
    kernels: NesoKernels,
    layers: Vec<LayerWeights>,
    final_norm: GpuBuffer,
    scratch: Scratch,
    cache: RefCell<Option<(usize, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer)>>,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    head_dim: usize,
    local_window: usize,
    global_every: usize,
    local_theta: f64,
    global_theta: f64,
    eps: f32,
    padded_seq: usize,
    max_rows: usize,
}

impl GpuModernBertMetal {
    pub fn new(
        device: &MetalDevice,
        cfg: &super::quantized_modernbert::Config,
        vb: QVarBuilder,
        max_seq_len: usize,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let heads = cfg.num_attention_heads;
        let head_dim = hidden / heads;
        ensure!(head_dim == 64, "Neso attention currently requires head_dim=64");
        let padded_seq = cdiv(max_seq_len, 64) * 64;
        let max_rows = scratch_rows(padded_seq, hidden, intermediate);
        let kernels = NesoKernels::load(device)?;
        let scratch = Scratch {
            residual: GpuBuffer::alloc_shared_f32(device, max_rows * hidden)?,
            normed: GpuBuffer::alloc_shared_f16(device, max_rows * hidden)?,
            q: GpuBuffer::alloc_f16(device, max_rows * hidden)?,
            k: GpuBuffer::alloc_f16(device, max_rows * hidden)?,
            v: GpuBuffer::alloc_f16(device, max_rows * hidden)?,
            attn_out: GpuBuffer::alloc_f16(device, max_rows * hidden)?,
            projection: GpuBuffer::alloc_f16(device, max_rows * hidden)?,
            gate: GpuBuffer::alloc_f16(device, max_rows * intermediate)?,
            up: GpuBuffer::alloc_f16(device, max_rows * intermediate)?,
            mlp_out: GpuBuffer::alloc_f16(device, max_rows * hidden)?,
        };
        let enc = vb.pp("encoder");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = enc.pp("layers").pp(i.to_string());
            let attn_vb = layer.pp("attn");
            let qkv = attn_vb.pp("Wqkv");
            let wi = layer.pp("mlp").pp("Wi");
            layers.push(LayerWeights {
                attn_norm: (i > 0)
                    .then(|| load_norm(device, &layer.pp("attn_norm"), hidden))
                    .transpose()?,
                attn: AttentionWeights {
                    q: load_weight_slice(device, &qkv, 3 * hidden, hidden, 0, hidden)?,
                    k: load_weight_slice(device, &qkv, 3 * hidden, hidden, hidden, hidden)?,
                    v: load_weight_slice(device, &qkv, 3 * hidden, hidden, 2 * hidden, hidden)?,
                    out: load_weight(device, &attn_vb.pp("Wo"), hidden, hidden)?,
                },
                mlp_norm: load_norm(device, &layer.pp("mlp_norm"), hidden)?,
                mlp: MlpWeights {
                    gate: load_weight_slice(device, &wi, 2 * intermediate, hidden, 0, intermediate)?,
                    up: load_weight_slice(device, &wi, 2 * intermediate, hidden, intermediate, intermediate)?,
                    out: load_weight(device, &layer.pp("mlp").pp("Wo"), hidden, intermediate)?,
                },
            });
        }
        let final_norm = load_norm(device, &enc.pp("final_norm"), hidden)?;
        info!(
            "Neso Metal ModernBERT: {} layers, max_seq={max_seq_len}, scratch={} MiB, max_rows={max_rows}",
            layers.len(),
            env_usize("WITCHCRAFT_NESO_SCRATCH_MB", DEFAULT_SCRATCH_MIB)
        );
        Ok(Self {
            device: device.clone(),
            kernels,
            layers,
            final_norm,
            scratch,
            cache: RefCell::new(None),
            hidden,
            intermediate,
            heads,
            head_dim,
            local_window: cfg.local_attention,
            global_every: cfg.global_attn_every_n_layers,
            local_theta: cfg.rope_theta,
            global_theta: cfg.global_rope_theta.unwrap_or(cfg.rope_theta),
            eps: cfg.norm_eps as f32,
            padded_seq,
            max_rows,
        })
    }

    fn cached_data(
        &self,
        seq_len: usize,
    ) -> Result<Ref<'_, (usize, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer)>> {
        if self.cache.borrow().as_ref().is_none_or(|cached| cached.0 != seq_len) {
            let (local_cos, local_sin) = make_rope(&self.device, seq_len, self.head_dim, self.local_theta)?;
            let (global_cos, global_sin) = make_rope(&self.device, seq_len, self.head_dim, self.global_theta)?;
            let local_mask = make_local_mask(&self.device, seq_len, self.local_window)?;
            *self.cache.borrow_mut() = Some((seq_len, local_cos, local_sin, global_cos, global_sin, local_mask));
        }
        Ok(Ref::map(self.cache.borrow(), |cache| cache.as_ref().unwrap()))
    }

    pub fn max_batch_size(&self, seq_len: usize) -> usize {
        let padded = cdiv(seq_len, 64) * 64;
        (self.max_rows / padded)
            .min(env_usize("WITCHCRAFT_NESO_MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE))
            .max(1)
    }

    pub fn forward(&self, input: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (batch, seq_len, dim) = input.dims3()?;
        ensure!(dim == self.hidden, "ModernBERT hidden-size mismatch");
        ensure!(lengths.len() == batch, "Neso Metal lengths/batch mismatch");
        ensure!(
            lengths.iter().all(|&length| length > 0 && length <= seq_len),
            "invalid Neso Metal sequence length"
        );
        let padded = cdiv(seq_len, 64) * 64;
        ensure!(
            padded <= self.padded_seq,
            "sequence length {seq_len} exceeds Neso capacity"
        );
        ensure!(
            batch * padded <= self.max_rows,
            "batch requires {} rows but Neso Metal scratch holds {}",
            batch * padded,
            self.max_rows
        );
        let values = input
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        unsafe {
            let dst = self.scratch.residual.contents_ptr() as *mut f32;
            for index in 0..batch {
                let source = values.as_ptr().add(index * seq_len * dim);
                let target = dst.add(index * padded * dim);
                std::ptr::copy_nonoverlapping(source, target, seq_len * dim);
                std::ptr::write_bytes(target.add(seq_len * dim), 0, (padded - seq_len) * dim);
            }
        }
        let length_buffer = GpuBuffer::alloc_shared_f32(&self.device, batch)?;
        unsafe {
            let dst = length_buffer.contents_ptr() as *mut i32;
            for (index, &length) in lengths.iter().enumerate() {
                *dst.add(index) = length as i32;
            }
        }
        let cached = self.cached_data(padded)?;
        let encoder_guard = self.device.command_encoder()?;
        let encoder = encoder_guard.as_ref();
        let s = &self.scratch;
        let rows = batch * padded;
        let elements = rows * self.hidden;
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(weight) = &layer.attn_norm {
                enc_layer_norm_f32_f16(
                    encoder,
                    &self.kernels.layer_norm_f32_f16,
                    &s.residual,
                    weight,
                    &s.normed,
                    rows,
                    self.hidden,
                    self.eps,
                );
            } else {
                enc_copy_f32_f16(encoder, &self.kernels.copy_f32_f16, &s.residual, &s.normed, elements);
            }
            let matmul = &self.kernels.matmul_fp16_64x64;
            for (weight, output) in [(&layer.attn.q, &s.q), (&layer.attn.k, &s.k), (&layer.attn.v, &s.v)] {
                enc_matmul(
                    encoder,
                    matmul,
                    &s.normed,
                    &weight.data,
                    output,
                    rows,
                    self.hidden,
                    self.hidden,
                    64,
                    64,
                );
            }
            let global = self.global_every > 0 && index % self.global_every == 0;
            let (cos, sin) = if global {
                (&cached.3, &cached.4)
            } else {
                (&cached.1, &cached.2)
            };
            enc_rope_inplace(
                encoder,
                &self.kernels.rope_inplace,
                &s.q,
                &s.k,
                cos,
                sin,
                rows,
                padded,
                self.hidden,
                self.head_dim,
            );
            enc_flash_attention_bias(
                encoder,
                &self.kernels.flash_attention_bias,
                &s.q,
                &s.k,
                &s.v,
                &s.attn_out,
                &length_buffer,
                &cached.5,
                batch,
                self.heads,
                seq_len,
                padded,
                !global,
                self.head_dim,
                self.head_dim as i32,
                self.hidden as i32,
                self.hidden as i32,
                1.0 / (self.head_dim as f32).sqrt(),
            );
            enc_matmul(
                encoder,
                matmul,
                &s.attn_out,
                &layer.attn.out.data,
                &s.projection,
                rows,
                self.hidden,
                self.hidden,
                64,
                64,
            );
            enc_residual_add_f16_f32(
                encoder,
                &self.kernels.residual_add_f16_f32,
                &s.projection,
                &s.residual,
                &s.residual,
                elements,
            );
            enc_layer_norm_f32_f16(
                encoder,
                &self.kernels.layer_norm_f32_f16,
                &s.residual,
                &layer.mlp_norm,
                &s.normed,
                rows,
                self.hidden,
                self.eps,
            );
            enc_matmul(
                encoder,
                matmul,
                &s.normed,
                &layer.mlp.gate.data,
                &s.gate,
                rows,
                self.intermediate,
                self.hidden,
                64,
                64,
            );
            enc_matmul(
                encoder,
                matmul,
                &s.normed,
                &layer.mlp.up.data,
                &s.up,
                rows,
                self.intermediate,
                self.hidden,
                64,
                64,
            );
            enc_gelu_mul(
                encoder,
                &self.kernels.gelu_mul,
                &s.gate,
                &s.up,
                &s.gate,
                rows * self.intermediate,
            );
            enc_matmul(
                encoder,
                matmul,
                &s.gate,
                &layer.mlp.out.data,
                &s.mlp_out,
                rows,
                self.hidden,
                self.intermediate,
                64,
                64,
            );
            enc_residual_add_f16_f32(
                encoder,
                &self.kernels.residual_add_f16_f32,
                &s.mlp_out,
                &s.residual,
                &s.residual,
                elements,
            );
        }
        enc_layer_norm_f32_f16(
            encoder,
            &self.kernels.layer_norm_f32_f16,
            &s.residual,
            &self.final_norm,
            &s.normed,
            rows,
            self.hidden,
            self.eps,
        );
        drop(encoder_guard);
        self.device.wait_until_completed()?;
        let values = unsafe {
            std::slice::from_raw_parts(
                s.normed.contents_ptr() as *const half::f16,
                batch * padded * self.hidden,
            )
        };
        let mut output = Vec::with_capacity(batch * seq_len * self.hidden);
        for index in 0..batch {
            output.extend(
                values[index * padded * self.hidden..index * padded * self.hidden + seq_len * self.hidden]
                    .iter()
                    .map(|value| value.to_f32()),
            );
        }
        Ok(Tensor::from_vec(output, (batch, seq_len, self.hidden), &Device::Cpu)?)
    }
}
