//! Full quantized ModernBERT encoder dispatched through Neso AOT D3D12 kernels.

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_d3d12_kernels::{Gpu, GpuBuffer};
use log::info;
use std::{
    cell::{Ref, RefCell},
    sync::Arc,
};

use crate::neso_d3d12_kernels::{
    NesoD3D12Kernels, create_f16_buffer, create_f32_buffer, d3d12_copy_f32_f16, d3d12_flash_attention_bias,
    d3d12_gelu_mul, d3d12_layer_norm_f32_f16, d3d12_matmul_fp16, d3d12_residual_add_f16_f32, d3d12_rope_inplace,
    download_bytes, upload_bytes,
};

type QVarBuilder = candle_transformers::quantized_var_builder::VarBuilder;
const DEFAULT_SCRATCH_MIB: usize = 96;
const DEFAULT_MAX_BATCH_SIZE: usize = 4;

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

fn upload_f32(gpu: &Gpu, values: &[f32]) -> Result<GpuBuffer> {
    let buffer = create_f32_buffer(gpu, values.len())?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload_bytes(gpu, &bytes, &buffer)?;
    Ok(buffer)
}

fn upload_f16(gpu: &Gpu, values: &[half::f16]) -> Result<GpuBuffer> {
    let buffer = create_f16_buffer(gpu, values.len())?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    upload_bytes(gpu, &bytes, &buffer)?;
    Ok(buffer)
}

fn upload_weight(gpu: &Gpu, rows: &[f32]) -> Result<Weight> {
    let values: Vec<half::f16> = rows.iter().map(|value| half::f16::from_f32(*value)).collect();
    Ok(Weight {
        data: upload_f16(gpu, &values)?,
    })
}

fn load_weight(gpu: &Gpu, vb: &QVarBuilder, out: usize, input: usize) -> Result<Weight> {
    let tensor = vb
        .get((out, input), "weight")?
        .dequantize(&Device::Cpu)?
        .t()?
        .contiguous()?;
    upload_weight(gpu, &tensor.flatten_all()?.to_vec1::<f32>()?)
}

fn load_weight_slice(
    gpu: &Gpu,
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
    upload_weight(gpu, &selected)
}

fn load_norm(gpu: &Gpu, vb: &QVarBuilder, dim: usize) -> Result<GpuBuffer> {
    upload_f32(gpu, &vb.get(dim, "weight")?.dequantize(&Device::Cpu)?.to_vec1::<f32>()?)
}

fn make_rope(gpu: &Gpu, seq_len: usize, head_dim: usize, theta: f64) -> Result<(GpuBuffer, GpuBuffer)> {
    let mut cos = Vec::with_capacity(seq_len * head_dim / 2);
    let mut sin = Vec::with_capacity(seq_len * head_dim / 2);
    for position in 0..seq_len {
        for i in 0..head_dim / 2 {
            let angle = position as f64 / theta.powf((2 * i) as f64 / head_dim as f64);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((upload_f32(gpu, &cos)?, upload_f32(gpu, &sin)?))
}

fn make_local_mask(gpu: &Gpu, seq_len: usize, local_window: usize) -> Result<GpuBuffer> {
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
    upload_f32(gpu, &values)
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

pub struct GpuModernBertD3D12 {
    gpu: Arc<Gpu>,
    kernels: NesoD3D12Kernels,
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

impl GpuModernBertD3D12 {
    pub fn new(cfg: &super::quantized_modernbert::Config, vb: QVarBuilder, max_seq_len: usize) -> Result<Self> {
        let gpu = Arc::new(Gpu::new(0)?);
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let heads = cfg.num_attention_heads;
        let head_dim = hidden / heads;
        ensure!(head_dim == 64, "Neso attention currently requires head_dim=64");
        let padded_seq = max_seq_len.div_ceil(64) * 64;
        let max_rows = scratch_rows(padded_seq, hidden, intermediate);
        let kernels = NesoD3D12Kernels::load(&gpu)?;
        let scratch = Scratch {
            residual: create_f32_buffer(&gpu, max_rows * hidden)?,
            normed: create_f16_buffer(&gpu, max_rows * hidden)?,
            q: create_f16_buffer(&gpu, max_rows * hidden)?,
            k: create_f16_buffer(&gpu, max_rows * hidden)?,
            v: create_f16_buffer(&gpu, max_rows * hidden)?,
            attn_out: create_f16_buffer(&gpu, max_rows * hidden)?,
            projection: create_f16_buffer(&gpu, max_rows * hidden)?,
            gate: create_f16_buffer(&gpu, max_rows * intermediate)?,
            up: create_f16_buffer(&gpu, max_rows * intermediate)?,
            mlp_out: create_f16_buffer(&gpu, max_rows * hidden)?,
        };
        let enc = vb.pp("encoder");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let layer = enc.pp("layers").pp(index.to_string());
            let attn = layer.pp("attn");
            let qkv = attn.pp("Wqkv");
            let wi = layer.pp("mlp").pp("Wi");
            layers.push(LayerWeights {
                attn_norm: (index > 0)
                    .then(|| load_norm(&gpu, &layer.pp("attn_norm"), hidden))
                    .transpose()?,
                attn: AttentionWeights {
                    q: load_weight_slice(&gpu, &qkv, 3 * hidden, hidden, 0, hidden)?,
                    k: load_weight_slice(&gpu, &qkv, 3 * hidden, hidden, hidden, hidden)?,
                    v: load_weight_slice(&gpu, &qkv, 3 * hidden, hidden, 2 * hidden, hidden)?,
                    out: load_weight(&gpu, &attn.pp("Wo"), hidden, hidden)?,
                },
                mlp_norm: load_norm(&gpu, &layer.pp("mlp_norm"), hidden)?,
                mlp: MlpWeights {
                    gate: load_weight_slice(&gpu, &wi, 2 * intermediate, hidden, 0, intermediate)?,
                    up: load_weight_slice(&gpu, &wi, 2 * intermediate, hidden, intermediate, intermediate)?,
                    out: load_weight(&gpu, &layer.pp("mlp").pp("Wo"), hidden, intermediate)?,
                },
            });
        }
        let final_norm = load_norm(&gpu, &enc.pp("final_norm"), hidden)?;
        info!(
            "Neso D3D12 ModernBERT: {} layers, max_seq={max_seq_len}, scratch={} MiB, max_rows={max_rows}",
            layers.len(),
            env_usize("WITCHCRAFT_NESO_SCRATCH_MB", DEFAULT_SCRATCH_MIB)
        );
        Ok(Self {
            gpu,
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

    fn cached(
        &self,
        seq_len: usize,
    ) -> Result<Ref<'_, (usize, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer)>> {
        if self.cache.borrow().as_ref().is_none_or(|value| value.0 != seq_len) {
            let (lc, ls) = make_rope(&self.gpu, seq_len, self.head_dim, self.local_theta)?;
            let (gc, gs) = make_rope(&self.gpu, seq_len, self.head_dim, self.global_theta)?;
            let local_mask = make_local_mask(&self.gpu, seq_len, self.local_window)?;
            *self.cache.borrow_mut() = Some((seq_len, lc, ls, gc, gs, local_mask));
        }
        Ok(Ref::map(self.cache.borrow(), |cache| cache.as_ref().unwrap()))
    }

    pub fn max_batch_size(&self, seq_len: usize) -> usize {
        let padded = seq_len.div_ceil(64) * 64;
        (self.max_rows / padded)
            .min(env_usize("WITCHCRAFT_NESO_MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE))
            .max(1)
    }

    pub fn forward(&self, input: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (batch, seq_len, dim) = input.dims3()?;
        ensure!(dim == self.hidden, "Neso D3D12 hidden-size mismatch");
        ensure!(lengths.len() == batch, "Neso D3D12 lengths/batch mismatch");
        ensure!(
            lengths.iter().all(|&length| length > 0 && length <= seq_len),
            "invalid Neso D3D12 sequence length"
        );
        let padded = seq_len.div_ceil(64) * 64;
        ensure!(padded <= self.padded_seq, "sequence length exceeds Neso capacity");
        ensure!(
            batch * padded <= self.max_rows,
            "batch requires {} rows but Neso D3D12 scratch holds {}",
            batch * padded,
            self.max_rows
        );
        let values = input
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut padded_values = vec![0.0f32; batch * padded * dim];
        for index in 0..batch {
            padded_values[index * padded * dim..index * padded * dim + seq_len * dim]
                .copy_from_slice(&values[index * seq_len * dim..(index + 1) * seq_len * dim]);
        }
        let bytes: Vec<u8> = padded_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        upload_bytes(&self.gpu, &bytes, &self.scratch.residual)?;
        let length_buffer = create_f32_buffer(&self.gpu, batch)?;
        let length_bytes: Vec<u8> = lengths
            .iter()
            .flat_map(|&length| (length as u32).to_le_bytes())
            .collect();
        upload_bytes(&self.gpu, &length_bytes, &length_buffer)?;
        let cached = self.cached(padded)?;
        let s = &self.scratch;
        let rows = batch * padded;
        let elements = rows * self.hidden;
        self.gpu
            .begin_batch()
            .map_err(|e| anyhow::anyhow!("begin_batch: {e}"))?;
        macro_rules! barrier {
            () => {
                self.gpu.record_uav_barrier();
            };
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if let Some(weight) = &layer.attn_norm {
                d3d12_layer_norm_f32_f16(
                    &self.kernels,
                    &s.residual,
                    weight,
                    &s.normed,
                    rows,
                    self.hidden,
                    self.eps,
                )?;
            } else {
                d3d12_copy_f32_f16(&self.kernels, &s.residual, &s.normed, elements)?;
            }
            barrier!();
            for (weight, output) in [(&layer.attn.q, &s.q), (&layer.attn.k, &s.k), (&layer.attn.v, &s.v)] {
                d3d12_matmul_fp16(
                    &self.kernels,
                    &s.normed,
                    &weight.data,
                    output,
                    rows,
                    self.hidden,
                    self.hidden,
                )?;
            }
            barrier!();
            let global = self.global_every > 0 && index % self.global_every == 0;
            let (cos, sin) = if global {
                (&cached.3, &cached.4)
            } else {
                (&cached.1, &cached.2)
            };
            d3d12_rope_inplace(
                &self.kernels,
                &s.q,
                &s.k,
                cos,
                sin,
                rows,
                padded,
                self.hidden,
                self.head_dim,
            )?;
            barrier!();
            d3d12_flash_attention_bias(
                &self.kernels,
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
            )?;
            barrier!();
            d3d12_matmul_fp16(
                &self.kernels,
                &s.attn_out,
                &layer.attn.out.data,
                &s.projection,
                rows,
                self.hidden,
                self.hidden,
            )?;
            barrier!();
            d3d12_residual_add_f16_f32(&self.kernels, &s.projection, &s.residual, &s.residual, elements)?;
            barrier!();
            d3d12_layer_norm_f32_f16(
                &self.kernels,
                &s.residual,
                &layer.mlp_norm,
                &s.normed,
                rows,
                self.hidden,
                self.eps,
            )?;
            barrier!();
            d3d12_matmul_fp16(
                &self.kernels,
                &s.normed,
                &layer.mlp.gate.data,
                &s.gate,
                rows,
                self.intermediate,
                self.hidden,
            )?;
            d3d12_matmul_fp16(
                &self.kernels,
                &s.normed,
                &layer.mlp.up.data,
                &s.up,
                rows,
                self.intermediate,
                self.hidden,
            )?;
            barrier!();
            d3d12_gelu_mul(&self.kernels, &s.gate, &s.up, &s.gate, rows * self.intermediate)?;
            barrier!();
            d3d12_matmul_fp16(
                &self.kernels,
                &s.gate,
                &layer.mlp.out.data,
                &s.mlp_out,
                rows,
                self.hidden,
                self.intermediate,
            )?;
            barrier!();
            d3d12_residual_add_f16_f32(&self.kernels, &s.mlp_out, &s.residual, &s.residual, elements)?;
            barrier!();
        }
        d3d12_layer_norm_f32_f16(
            &self.kernels,
            &s.residual,
            &self.final_norm,
            &s.normed,
            rows,
            self.hidden,
            self.eps,
        )?;
        self.gpu.end_batch().map_err(|e| anyhow::anyhow!("end_batch: {e}"))?;
        let bytes = download_bytes(&self.gpu, &s.normed, (batch * padded * self.hidden * 2) as u64)?;
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let mut output = Vec::with_capacity(batch * seq_len * self.hidden);
        for index in 0..batch {
            output.extend_from_slice(
                &values[index * padded * self.hidden..index * padded * self.hidden + seq_len * self.hidden],
            );
        }
        Ok(Tensor::from_vec(output, (batch, seq_len, self.hidden), &Device::Cpu)?)
    }
}
