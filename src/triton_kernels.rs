//! Triton-compiled Metal kernel dispatch for T5 Flash Attention 2.
//!
//! Loads pre-compiled .metallib binaries (zero runtime compilation).
//! Apple Silicon metallibs use Metal 3.1 (simdgroup_matrix).
//! Intel Mac metallibs use Metal 2.4 (scalar fallback).

use anyhow::Result;
use candle_core::{MetalDevice, Storage, Tensor};
use candle_metal_kernels::metal::{ComputePipeline, Device, Library};
use dispatch2::DispatchData;
use objc2_metal::MTLSize;

// Auto-generated: kernel_data module, T5Kernels struct + load().
include!("../kernels/out/generated/triton_metal_gen.rs");

fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Load a Metal library from pre-compiled metallib binary data.
/// Upstream candle's Device wrapper only has `new_library_with_source`;
/// this calls the raw MTLDevice API for binary loading.
fn new_library_with_data(device: &Device, data: &[u8]) -> Result<Library> {
    use objc2_metal::MTLDevice;
    let dispatch_data = DispatchData::from_bytes(data);
    let raw = device
        .as_ref()
        .newLibraryWithData_error(&dispatch_data)
        .map_err(|e| anyhow::anyhow!("Failed to load metallib: {e}"))?;
    Ok(Library::new(raw))
}

/// Threadgroup size for flash attention.
fn fa_tg_size(pipeline: &ComputePipeline) -> MTLSize {
    let max = pipeline.max_total_threads_per_threadgroup();
    MTLSize {
        width: 832_usize.min(max),
        height: 1,
        depth: 1,
    }
}

/// Threadgroup size for element-wise kernels (BLOCK_SIZE threads).
fn ew_tg_size(pipeline: &ComputePipeline, block_size: usize) -> MTLSize {
    let max = pipeline.max_total_threads_per_threadgroup();
    MTLSize {
        width: block_size.min(max),
        height: 1,
        depth: 1,
    }
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
        (
            Storage::Metal(mq),
            Storage::Metal(mk),
            Storage::Metal(mv),
            Storage::Metal(mo),
            Storage::Metal(mb),
        ) => {
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(mq.buffer()), q_off);
            encoder.set_buffer(1, Some(mk.buffer()), k_off);
            encoder.set_buffer(2, Some(mv.buffer()), v_off);
            encoder.set_buffer(3, Some(mo.buffer()), o_off);
            encoder.set_buffer(4, Some(mb.buffer()), b_off);
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
        }
        _ => anyhow::bail!("All tensors must be on Metal device"),
    }
}

/// Dispatch fused gated-GELU: out = GELU(gate) * up.
///
/// gate, up: [n_elements] fp16, contiguous.
/// Returns:  [n_elements] fp16 (new allocation).
pub fn triton_gelu_mul(
    device: &MetalDevice,
    pipeline: &ComputePipeline,
    gate: &Tensor,
    up: &Tensor,
) -> Result<Tensor> {
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
                encoder.set_buffer(0, Some(mg.buffer()), g_off);
                encoder.set_buffer(1, Some(mu.buffer()), u_off);
                encoder.set_buffer(2, Some(mo.buffer()), o_off);
                encoder.set_bytes(3, &(n_elements as i32));

                let grid = MTLSize {
                    width: cdiv(n_elements, 1024),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, ew_tg_size(pipeline, 1024));
            }
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
                encoder.set_buffer(0, Some(mx.buffer()), x_off);
                encoder.set_buffer(1, Some(mw.buffer()), w_off);
                encoder.set_buffer(2, Some(mo.buffer()), o_off);
                encoder.set_bytes(3, &(n_rows as i32));
                encoder.set_bytes(4, &(dim as i32));
                encoder.set_bytes(5, &eps);

                let grid = MTLSize {
                    width: n_rows,
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, ew_tg_size(pipeline, 1024));
            }
            _ => anyhow::bail!("All tensors must be on Metal device"),
        }
    }
    Ok(out)
}
