//! Neso-compiled Metal kernel dispatch for transformer encoders.
//!
//! Loads pre-compiled .metallib binaries (zero runtime compilation).
//! Apple Silicon metallibs use Metal 3.1 (simdgroup_matrix).
//! Intel Mac metallibs use Metal 2.4 (scalar fallback).

use anyhow::Result;
pub use candle_core::GpuBuffer;
use candle_core::{MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Device, Library};
use dispatch2::DispatchData;
use objc2_metal::MTLSize;

// Auto-generated: kernel data and Neso pipeline loader.
include!("../kernels/out/generated/neso_metal_gen.rs");

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Load a Metal library from pre-compiled metallib binary data.
fn new_library_with_data(device: &Device, data: &[u8]) -> Result<Library> {
    use objc2_metal::MTLDevice;
    let dispatch_data = DispatchData::from_bytes(data);
    let raw = device
        .as_ref()
        .newLibraryWithData_error(&dispatch_data)
        .map_err(|e| anyhow::anyhow!("Failed to load metallib: {e}"))?;
    Ok(Library::new(raw))
}

/// Clamp threadgroup size to pipeline maximum.
fn tg_size(pipeline: &ComputePipeline, requested: usize) -> MTLSize {
    let max = pipeline.max_total_threads_per_threadgroup();
    MTLSize {
        width: requested.min(max),
        height: 1,
        depth: 1,
    }
}

/// Threadgroup size for flash attention (MSL kernel uses 832 threads).
fn fa_tg_size(pipeline: &ComputePipeline) -> MTLSize {
    tg_size(pipeline, 832)
}

/// Set dynamic threadgroup memory for AIR kernels (aarch64 only).
#[cfg(target_arch = "aarch64")]
fn set_air_tg_mem(encoder: &ComputeCommandEncoder, bytes: usize) {
    if bytes > 0 {
        encoder.set_threadgroup_memory_length(0, bytes);
    }
}

/// Threadgroup size for matmul kernels.
fn matmul_tg_size(pipeline: &ComputePipeline, _block_m: usize, _block_n: usize) -> MTLSize {
    tg_size(pipeline, 256)
}

/// Dispatch Flash Attention 2 with additive position bias.
///
/// Q, K, V: [n_heads, padded_seq, head_dim] fp16, contiguous.
/// out:     [n_heads, padded_seq, head_dim] fp16, contiguous.
/// bias:    [n_heads, seq_len, seq_len] fp32, contiguous.
///
/// `seq_len` is the actual (unpadded) sequence length, used for bounds
/// checks and bias indexing. Q/K/V/O may be padded to a multiple of BM=32
/// along dim 1 — stride_h is derived from Q's actual layout.
///
/// The kernel fuses QK^T + bias, softmax, and attn*V in a single pass
/// using online softmax (no materialized seq×seq matrix).
pub fn triton_flash_attention_bias(
    device: &MetalDevice,
    pipeline: &ComputePipeline,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    sm_scale: f32,
) -> Result<()> {
    // Extract Metal buffers and byte offsets from all 5 tensors
    let (sq, lq) = q.storage_and_layout();
    let (sk, lk) = k.storage_and_layout();
    let (sv, lv) = v.storage_and_layout();
    let (so, lo) = out.storage_and_layout();
    let (sb, lb) = bias.storage_and_layout();

    let q_off = lq.start_offset() * q.dtype().size_in_bytes();
    let k_off = lk.start_offset() * k.dtype().size_in_bytes();
    let v_off = lv.start_offset() * v.dtype().size_in_bytes();
    let o_off = lo.start_offset() * out.dtype().size_in_bytes();
    let b_off = lb.start_offset() * bias.dtype().size_in_bytes();

    // stride_h from Q's actual layout (supports padded tensors)
    let stride_h = lq.stride()[0] as i32;

    match (&*sq, &*sk, &*sv, &*so, &*sb) {
        (Storage::Metal(mq), Storage::Metal(mk), Storage::Metal(mv), Storage::Metal(mo), Storage::Metal(mb)) => {
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(pipeline);
            let encoder = encoder.as_ref();
            encoder.set_input_buffer(0, Some(mq.buffer()), q_off);
            encoder.set_input_buffer(1, Some(mk.buffer()), k_off);
            encoder.set_input_buffer(2, Some(mv.buffer()), v_off);
            encoder.set_output_buffer(3, Some(mo.buffer()), o_off);
            encoder.set_input_buffer(4, Some(mb.buffer()), b_off);
            encoder.set_bytes(5, &(seq_len as i32));
            encoder.set_bytes(6, &stride_h);
            encoder.set_bytes(7, &(head_dim as i32));
            encoder.set_bytes(8, &(head_dim as i32));
            encoder.set_bytes(9, &sm_scale);

            // AIR kernels need dynamic threadgroup memory
            #[cfg(target_arch = "aarch64")]
            encoder.set_threadgroup_memory_length(0, 8192);

            let grid = MTLSize {
                width: cdiv(seq_len, 32),
                height: n_heads,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, fa_tg_size(pipeline));
            Ok(())
        },
        _ => anyhow::bail!("All tensors must be on Metal device"),
    }
}

/// Dispatch fused gated-GELU: out = GELU(gate) * up.
///
/// gate, up: [n_elements] fp16, contiguous.
/// Returns:  [n_elements] fp16 (new allocation).
pub fn triton_gelu_mul(device: &MetalDevice, pipeline: &ComputePipeline, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let n_elements = gate.elem_count();
    let out = gate.zeros_like()?;

    {
        let (sg, lg) = gate.storage_and_layout();
        let (su, lu) = up.storage_and_layout();
        let (so, lo) = out.storage_and_layout();

        let g_off = lg.start_offset() * gate.dtype().size_in_bytes();
        let u_off = lu.start_offset() * up.dtype().size_in_bytes();
        let o_off = lo.start_offset() * out.dtype().size_in_bytes();

        match (&*sg, &*su, &*so) {
            (Storage::Metal(mg), Storage::Metal(mu), Storage::Metal(mo)) => {
                let encoder = device.command_encoder()?;
                encoder.set_compute_pipeline_state(pipeline);
                let encoder = encoder.as_ref();
                encoder.set_input_buffer(0, Some(mg.buffer()), g_off);
                encoder.set_input_buffer(1, Some(mu.buffer()), u_off);
                encoder.set_output_buffer(2, Some(mo.buffer()), o_off);
                encoder.set_bytes(3, &(n_elements as i32));

                let grid = MTLSize {
                    width: cdiv(n_elements, 1024),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
            },
            _ => anyhow::bail!("All tensors must be on Metal device"),
        }
    }
    Ok(out)
}

/// Dispatch T5 RMS LayerNorm: out = x / sqrt(mean(x^2) + eps) * weight.
///
/// x:      [n_rows, dim] fp32, contiguous.
/// weight: [dim] fp32, contiguous.
/// Returns: [n_rows, dim] fp32 (new allocation).
pub fn triton_rms_norm(
    device: &MetalDevice,
    pipeline: &ComputePipeline,
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (n_rows, dim) = (x.dim(0)?, x.dim(1)?);
    let out = x.zeros_like()?;

    {
        let (sx, lx) = x.storage_and_layout();
        let (sw, lw) = weight.storage_and_layout();
        let (so, lo) = out.storage_and_layout();

        let x_off = lx.start_offset() * x.dtype().size_in_bytes();
        let w_off = lw.start_offset() * weight.dtype().size_in_bytes();
        let o_off = lo.start_offset() * out.dtype().size_in_bytes();

        match (&*sx, &*sw, &*so) {
            (Storage::Metal(mx), Storage::Metal(mw), Storage::Metal(mo)) => {
                let encoder = device.command_encoder()?;
                encoder.set_compute_pipeline_state(pipeline);
                let encoder = encoder.as_ref();
                encoder.set_input_buffer(0, Some(mx.buffer()), x_off);
                encoder.set_input_buffer(1, Some(mw.buffer()), w_off);
                encoder.set_output_buffer(2, Some(mo.buffer()), o_off);
                encoder.set_bytes(3, &(n_rows as i32));
                encoder.set_bytes(4, &(dim as i32));
                encoder.set_bytes(5, &eps);

                let grid = MTLSize {
                    width: n_rows,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
            },
            _ => anyhow::bail!("All tensors must be on Metal device"),
        }
    }
    Ok(out)
}

// ── GpuBuffer-based dispatch (for GPU encoder) ───────────────────────────────

pub fn enc_copy_f32_f16(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    x: &GpuBuffer,
    out: &GpuBuffer,
    n_elements: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(x.buf()), x.offset);
    enc.set_output_buffer(1, Some(out.buf()), out.offset);
    enc.set_bytes(2, &(n_elements as i32));
    let grid = MTLSize {
        width: cdiv(n_elements, 1024),
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
}

pub fn enc_layer_norm_f32_f16(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    n_rows: usize,
    dim: usize,
    eps: f32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(x.buf()), x.offset);
    enc.set_input_buffer(1, Some(weight.buf()), weight.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(n_rows as i32));
    enc.set_bytes(4, &(dim as i32));
    enc.set_bytes(5, &eps);
    let grid = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
}

pub fn enc_rope_inplace(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    q: &GpuBuffer,
    k: &GpuBuffer,
    cos: &GpuBuffer,
    sin: &GpuBuffer,
    n_rows: usize,
    row_stride: usize,
    hidden_size: usize,
    head_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_output_buffer(0, Some(q.buf()), q.offset);
    enc.set_output_buffer(1, Some(k.buf()), k.offset);
    enc.set_input_buffer(2, Some(cos.buf()), cos.offset);
    enc.set_input_buffer(3, Some(sin.buf()), sin.offset);
    enc.set_bytes(4, &(n_rows as i32));
    enc.set_bytes(5, &(row_stride as i32));
    enc.set_bytes(6, &(hidden_size as i32));
    enc.set_bytes(7, &(head_dim as i32));
    let pairs = n_rows * hidden_size / 2;
    let grid = MTLSize {
        width: pairs,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg_size(pipeline, 1));
}

/// Matmul: C[M,N] = A[M,K] @ B[K,N], all fp16, fp32 accumulate.
/// B must be [K,N] row-major (transposed from weight's natural [N,K] layout).
pub fn enc_matmul(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    m: usize,
    n: usize,
    k: usize,
    block_m: usize,
    block_n: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(a.buf()), a.offset);
    enc.set_input_buffer(1, Some(b.buf()), b.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(m as i32));
    enc.set_bytes(4, &(n as i32));
    enc.set_bytes(5, &(k as i32));
    enc.set_bytes(6, &(k as i32)); // stride_am = K
    enc.set_bytes(7, &1i32); // stride_ak = 1
    enc.set_bytes(8, &(n as i32)); // stride_bk = N
    enc.set_bytes(9, &1i32); // stride_bn = 1
    enc.set_bytes(10, &(n as i32)); // stride_cm = N
    enc.set_bytes(11, &1i32); // stride_cn = 1
    let grid = MTLSize {
        width: cdiv(m, block_m),
        height: cdiv(n, block_n),
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, matmul_tg_size(pipeline, block_m, block_n));
}

/// Q8 Matmul: C[M,N] = A[M,K](fp16) @ B[K,N](int8) * scales[N](fp16), fp32 accum.
/// B stored as raw int8 row-major [K,N]. Scales: [N] fp16 per-column.
pub fn enc_matmul_q8(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    scales: &GpuBuffer,
    m: usize,
    n: usize,
    k: usize,
    block_m: usize,
    block_n: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(a.buf()), a.offset);
    enc.set_input_buffer(1, Some(b.buf()), b.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(m as i32));
    enc.set_bytes(4, &(n as i32));
    enc.set_bytes(5, &(k as i32));
    enc.set_bytes(6, &(k as i32)); // stride_am = K
    enc.set_bytes(7, &1i32); // stride_ak = 1
    enc.set_bytes(8, &(n as i32)); // stride_bk = N
    enc.set_bytes(9, &1i32); // stride_bn = 1
    enc.set_bytes(10, &(n as i32)); // stride_cm = N
    enc.set_bytes(11, &1i32); // stride_cn = 1
    enc.set_input_buffer(12, Some(scales.buf()), scales.offset);
    let grid = MTLSize {
        width: cdiv(m, block_m),
        height: cdiv(n, block_n),
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, matmul_tg_size(pipeline, block_m, block_n));
}

/// Residual add: out = a + b, element-wise fp16.
pub fn enc_residual_add(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    n_elements: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(a.buf()), a.offset);
    enc.set_input_buffer(1, Some(b.buf()), b.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(n_elements as i32));
    let tg_w = tg_size(pipeline, 1024).width;
    let grid = MTLSize {
        width: cdiv(n_elements, tg_w),
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(
        grid,
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// RMS norm fp16: out = x / sqrt(mean(x^2) + eps) * weight.
/// x: [n_rows, dim] fp16, weight: [dim] fp16, out: [n_rows, dim] fp16.
pub fn enc_rms_norm_f16(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    n_rows: usize,
    dim: usize,
    eps: f32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(x.buf()), x.offset);
    enc.set_input_buffer(1, Some(weight.buf()), weight.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(n_rows as i32));
    enc.set_bytes(4, &(dim as i32));
    enc.set_bytes(5, &eps);
    let grid = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
}

/// Fused gated-GELU: out = GELU(gate) * up, all fp16.
pub fn enc_gelu_mul(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n_elements: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(gate.buf()), gate.offset);
    enc.set_input_buffer(1, Some(up.buf()), up.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(n_elements as i32));
    let grid = MTLSize {
        width: cdiv(n_elements, 1024),
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
}

/// Residual add mixed precision: out[i] = float(a[i]) + b[i].
/// a: [n_elements] fp16 (projection output), b: [n_elements] fp32 (residual).
/// out: [n_elements] fp32. b and out may alias for in-place update.
pub fn enc_residual_add_f16_f32(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    a: &GpuBuffer,
    b: &GpuBuffer,
    out: &GpuBuffer,
    n_elements: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(a.buf()), a.offset);
    enc.set_input_buffer(1, Some(b.buf()), b.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(n_elements as i32));
    let tg_w = tg_size(pipeline, 1024).width;
    let grid = MTLSize {
        width: cdiv(n_elements, tg_w),
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(
        grid,
        MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// RMS norm F32→F16: out = half(x / sqrt(mean(x^2) + eps) * weight).
/// x: [n_rows, dim] fp32, weight: [dim] fp32, out: [n_rows, dim] fp16.
pub fn enc_rms_norm_f32_f16(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    n_rows: usize,
    dim: usize,
    eps: f32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(x.buf()), x.offset);
    enc.set_input_buffer(1, Some(weight.buf()), weight.offset);
    enc.set_output_buffer(2, Some(out.buf()), out.offset);
    enc.set_bytes(3, &(n_rows as i32));
    enc.set_bytes(4, &(dim as i32));
    enc.set_bytes(5, &eps);
    let grid = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg_size(pipeline, 1024));
}

/// Batched Flash Attention 2 with procedural local/padding masks.
/// Q/K/V: [batch, padded_seq, n_heads * D] fp16.
pub fn enc_flash_attention_bias(
    enc: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    out: &GpuBuffer,
    lengths: &GpuBuffer,
    local_mask: &GpuBuffer,
    batch_size: usize,
    n_heads: usize,
    seq_len: usize,
    padded_seq: usize,
    use_local_mask: bool,
    _head_dim: usize,
    stride_h: i32,
    stride_m: i32,
    stride_o: i32,
    sm_scale: f32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_input_buffer(0, Some(q.buf()), q.offset);
    enc.set_input_buffer(1, Some(k.buf()), k.offset);
    enc.set_input_buffer(2, Some(v.buf()), v.offset);
    enc.set_output_buffer(3, Some(out.buf()), out.offset);
    enc.set_input_buffer(4, Some(lengths.buf()), lengths.offset);
    enc.set_input_buffer(5, Some(local_mask.buf()), local_mask.offset);
    enc.set_bytes(6, &(seq_len as i32));
    enc.set_bytes(7, &(n_heads as i32));
    enc.set_bytes(8, &((padded_seq * n_heads * _head_dim) as i32));
    enc.set_bytes(9, &(use_local_mask as i32));
    enc.set_bytes(10, &(padded_seq as i32));
    enc.set_bytes(11, &stride_h);
    enc.set_bytes(12, &stride_m);
    enc.set_bytes(13, &stride_o);
    enc.set_bytes(14, &sm_scale);
    #[cfg(target_arch = "aarch64")]
    set_air_tg_mem(enc, 8192);
    let grid = MTLSize {
        width: cdiv(seq_len, 32),
        height: batch_size * n_heads,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, fa_tg_size(pipeline));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neso_fp16_matmul_is_deterministic() -> Result<()> {
        let candle_core::Device::Metal(device) = candle_core::Device::new_metal(0)? else {
            unreachable!()
        };
        let kernels = NesoKernels::load(&device)?;
        let a_values: Vec<half::f16> = (0..64 * 64)
            .map(|i| half::f16::from_f32(((i * 17 % 31) as f32 - 15.0) / 19.0))
            .collect();
        let b_values: Vec<half::f16> = (0..64 * 64)
            .map(|i| half::f16::from_f32(((i * 11 % 29) as f32 - 14.0) / 23.0))
            .collect();
        let a = GpuBuffer::from_f16_data(&device, &a_values)?;
        let b = GpuBuffer::from_f16_data(&device, &b_values)?;
        let out = GpuBuffer::alloc_shared_f16(&device, 64 * 64)?;
        let encoder = device.command_encoder()?;
        enc_matmul(encoder.as_ref(), &kernels.matmul_fp16_64x64, &a, &b, &out, 64, 64, 64, 64, 64);
        drop(encoder);
        device.wait_until_completed()?;
        let values = unsafe { std::slice::from_raw_parts(out.contents_ptr() as *const half::f16, 64 * 64) };
        for row in 0..64 {
            for column in 0..64 {
                let expected: f32 = (0..64)
                    .map(|inner| a_values[row * 64 + inner].to_f32() * b_values[inner * 64 + column].to_f32())
                    .sum();
                let got = values[row * 64 + column].to_f32();
                assert!(
                    (got - expected).abs() < 0.02,
                    "[{row},{column}] got {got}, expected {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn neso_flash_attention_matches_reference() -> Result<()> {
        let candle_core::Device::Metal(device) = candle_core::Device::new_metal(0)? else {
            unreachable!()
        };
        let kernels = NesoKernels::load(&device)?;
        let (batch, heads, seq, dim) = (2usize, 12usize, 7usize, 64usize);
        let sequence_lengths = [seq, 5usize];
        let values = |factor, modulus, center, divisor| -> Vec<half::f16> {
            (0..batch * seq * heads * dim)
                .map(|i| half::f16::from_f32(((i * factor % modulus) as f32 - center) / divisor))
                .collect()
        };
        let qv = values(17, 31, 15.0, 19.0);
        let kv = values(11, 29, 14.0, 23.0);
        let vv = values(7, 37, 18.0, 17.0);
        let q = GpuBuffer::from_f16_data(&device, &qv)?;
        let k = GpuBuffer::from_f16_data(&device, &kv)?;
        let v = GpuBuffer::from_f16_data(&device, &vv)?;
        let lengths = GpuBuffer::alloc_shared_f32(&device, batch)?;
        unsafe {
            let ptr = lengths.contents_ptr() as *mut i32;
            for (index, &length) in sequence_lengths.iter().enumerate() {
                *ptr.add(index) = length as i32;
            }
        }
        let local_mask = GpuBuffer::from_f32_data(&device, &vec![0.0; seq * seq])?;
        let out = GpuBuffer::alloc_shared_f16(&device, batch * seq * heads * dim)?;
        let encoder = device.command_encoder()?;
        enc_flash_attention_bias(
            encoder.as_ref(),
            &kernels.flash_attention_bias,
            &q,
            &k,
            &v,
            &out,
            &lengths,
            &local_mask,
            batch,
            heads,
            seq,
            seq,
            false,
            dim,
            dim as i32,
            (heads * dim) as i32,
            (heads * dim) as i32,
            1.0 / 8.0,
        );
        drop(encoder);
        device.wait_until_completed()?;
        let got =
            unsafe { std::slice::from_raw_parts(out.contents_ptr() as *const half::f16, batch * seq * heads * dim) };
        for batch_index in 0..batch {
            for head in 0..heads {
                for query in 0..seq {
                    let valid_len = sequence_lengths[batch_index];
                    let base = batch_index * seq * heads * dim;
                    let mut scores = vec![0.0f32; valid_len];
                    for key in 0..valid_len {
                        for d in 0..dim {
                            scores[key] += qv[base + query * heads * dim + head * dim + d].to_f32()
                                * kv[base + key * heads * dim + head * dim + d].to_f32()
                                / 8.0;
                        }
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = scores
                        .iter_mut()
                        .map(|x| {
                            *x = (*x - max).exp();
                            *x
                        })
                        .sum();
                    for d in 0..dim {
                        let expected: f32 = (0..valid_len)
                            .map(|key| scores[key] / sum * vv[base + key * heads * dim + head * dim + d].to_f32())
                            .sum();
                        let actual = got[base + query * heads * dim + head * dim + d].to_f32();
                        assert!(
                            (actual - expected).abs() < 0.03,
                            "b={batch_index} h={head} q={query} d={d}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn neso_layer_norm_matches_reference() -> Result<()> {
        let candle_core::Device::Metal(device) = candle_core::Device::new_metal(0)? else {
            unreachable!()
        };
        let kernels = NesoKernels::load(&device)?;
        let (rows, dim) = (7usize, 768usize);
        let xv: Vec<f32> = (0..rows * dim).map(|i| ((i * 17 % 131) as f32 - 65.0) / 19.0).collect();
        let wv: Vec<f32> = (0..dim).map(|i| 0.5 + (i * 7 % 29) as f32 / 31.0).collect();
        let x = GpuBuffer::from_f32_data(&device, &xv)?;
        let weight = GpuBuffer::from_f32_data(&device, &wv)?;
        let out = GpuBuffer::alloc_shared_f16(&device, rows * dim)?;
        let encoder = device.command_encoder()?;
        enc_layer_norm_f32_f16(
            encoder.as_ref(),
            &kernels.layer_norm_f32_f16,
            &x,
            &weight,
            &out,
            rows,
            dim,
            1e-5,
        );
        drop(encoder);
        device.wait_until_completed()?;
        let got = unsafe { std::slice::from_raw_parts(out.contents_ptr() as *const half::f16, rows * dim) };
        for row in 0..rows {
            let slice = &xv[row * dim..(row + 1) * dim];
            let mean = slice.iter().sum::<f32>() / dim as f32;
            let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / dim as f32;
            for d in 0..dim {
                let expected = (slice[d] - mean) / (variance + 1e-5).sqrt() * wv[d];
                assert!((got[row * dim + d].to_f32() - expected).abs() < 0.01);
            }
        }
        Ok(())
    }

    #[test]
    fn neso_rope_matches_reference() -> Result<()> {
        let candle_core::Device::Metal(device) = candle_core::Device::new_metal(0)? else {
            unreachable!()
        };
        let kernels = NesoKernels::load(&device)?;
        let (batch, seq, hidden, head_dim) = (2usize, 7usize, 768usize, 64usize);
        let original: Vec<half::f16> = (0..batch * seq * hidden)
            .map(|i| half::f16::from_f32(((i * 17 % 131) as f32 - 65.0) / 37.0))
            .collect();
        let trig: Vec<f32> = (0..seq * head_dim / 2).map(|i| i as f32 / 113.0).collect();
        let cosv: Vec<f32> = trig.iter().map(|x| x.cos()).collect();
        let sinv: Vec<f32> = trig.iter().map(|x| x.sin()).collect();
        let q = GpuBuffer::alloc_shared_f16(&device, original.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(original.as_ptr(), q.contents_ptr() as *mut half::f16, original.len());
        }
        let k = GpuBuffer::alloc_shared_f16(&device, original.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(original.as_ptr(), k.contents_ptr() as *mut half::f16, original.len());
        }
        let cos = GpuBuffer::from_f32_data(&device, &cosv)?;
        let sin = GpuBuffer::from_f32_data(&device, &sinv)?;
        let encoder = device.command_encoder()?;
        enc_rope_inplace(
            encoder.as_ref(),
            &kernels.rope_inplace,
            &q,
            &k,
            &cos,
            &sin,
            batch * seq,
            seq,
            hidden,
            head_dim,
        );
        drop(encoder);
        device.wait_until_completed()?;
        let got = unsafe { std::slice::from_raw_parts(q.contents_ptr() as *const half::f16, original.len()) };
        for batch_index in 0..batch {
            for token in 0..seq {
                for head in 0..hidden / head_dim {
                    for d in 0..head_dim / 2 {
                        let base = batch_index * seq * hidden;
                        let first = base + token * hidden + head * head_dim + d;
                        let second = first + head_dim / 2;
                        let ti = token * head_dim / 2 + d;
                        let a = original[first].to_f32();
                        let b = original[second].to_f32();
                        let e1 = a * cosv[ti] - b * sinv[ti];
                        let e2 = b * cosv[ti] + a * sinv[ti];
                        assert!(
                            (got[first].to_f32() - e1).abs() < 0.003,
                            "t={token} h={head} d={d} first {} vs {e1}",
                            got[first].to_f32()
                        );
                        assert!(
                            (got[second].to_f32() - e2).abs() < 0.003,
                            "t={token} h={head} d={d} second {} vs {e2}",
                            got[second].to_f32()
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
