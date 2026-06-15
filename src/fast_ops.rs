//! Fast contiguous f32 ops bypassing candle's generic dispatch.
//!
//! Also provides [`PackedRight`] for efficient `A × B^T` using fbgemm-rs
//! when available, with transparent fallback to candle matmul.

use candle_core::backend::BackendStorage;
use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Device, Layout, Result, Shape, Tensor,
};
#[cfg(feature = "hybrid-dequant")]
use rayon::prelude::*;

struct FastAddOp;

impl CustomOp2 for FastAddOp {
    fn name(&self) -> &'static str {
        "fast-add"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle_core::bail!("fast_add only supports f32")
        }
        let a = s1.as_slice::<f32>()?;
        let b = s2.as_slice::<f32>()?;
        let n = l1.shape().elem_count();
        let a = &a[l1.start_offset()..l1.start_offset() + n];
        let b = &b[l2.start_offset()..l2.start_offset() + n];
        let mut dst = vec![0f32; n];
        for i in 0..n {
            dst[i] = a[i] + b[i];
        }
        Ok((CpuStorage::F32(dst), l1.shape().clone()))
    }
}

/// Element-wise add bypassing candle's generic binary op dispatch.
/// On Metal/GPU, falls back to candle's built-in addition since custom ops
/// have overhead on GPU and the optimization is CPU-specific.
pub fn fast_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if matches!(a.device(), Device::Cpu) {
        a.apply_op2_no_bwd(b, &FastAddOp)
    } else {
        a + b
    }
}

// ---- Packed matmul: A × B^T ----

/// Pre-packed right-hand side for efficient `A × B^T` computation.
/// Pack once with [`PackedRight::new`], then call [`PackedRight::matmul`] repeatedly.
/// Uses fbgemm-rs on CPU when available, otherwise candle matmul.
pub enum PackedRight {
    #[cfg(feature = "fbgemm")]
    Packed {
        inner: fbgemm_rs::PackedMatrixBf16,
        n: usize,
        device: Device,
    },
    Tensor(Tensor),
}

impl PackedRight {
    /// Pack a `[N, D]` tensor for use as the right side of `A × B^T`.
    pub fn new(b: &Tensor) -> Result<Self> {
        #[cfg(feature = "fbgemm")]
        if matches!(b.device(), Device::Cpu) && b.dtype() == DType::F32 {
            let (n, d) = b.dims2()?;
            let data = b.flatten_all()?.to_vec1::<f32>()?;
            let packed = fbgemm_rs::PackedMatrixBf16::from_transposed(d, n, &data);
            return Ok(Self::Packed {
                inner: packed,
                n,
                device: Device::Cpu,
            });
        }
        Ok(Self::Tensor(b.clone()))
    }

    /// Compute `A × B^T` where A is `[M, D]`. Returns `[M, N]`.
    pub fn matmul(&self, a: &Tensor) -> Result<Tensor> {
        match self {
            #[cfg(feature = "fbgemm")]
            Self::Packed { inner, n, device } => {
                let (m, _d) = a.dims2()?;
                let a_data = a.flatten_all()?.to_vec1::<f32>()?;
                let mut c = vec![0f32; m * *n];
                fbgemm_rs::sgemm_bf16_simple(m, &a_data, inner, &mut c);
                Tensor::from_vec(c, (m, *n), device)
            }
            Self::Tensor(b) => a.matmul(&b.t()?),
        }
    }
}

/// Compute `A × B^T` where A is `[M, D]` and B is `[N, D]`. Returns `[M, N]`.
/// Uses fbgemm-rs on CPU when available, otherwise candle matmul.
pub fn matmul_t(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "fbgemm")]
    if matches!(a.device(), Device::Cpu) && a.dtype() == DType::F32 {
        let (m, d) = a.dims2()?;
        let (n, d2) = b.dims2()?;
        if d != d2 {
            candle_core::bail!("matmul_t dimension mismatch: a is [{m},{d}], b is [{n},{d2}]");
        }
        let a_data = a.flatten_all()?.to_vec1::<f32>()?;
        let b_data = b.flatten_all()?.to_vec1::<f32>()?;
        let packed = fbgemm_rs::PackedMatrixBf16::from_transposed(d, n, &b_data);
        let mut c = vec![0f32; m * n];
        fbgemm_rs::sgemm_bf16_simple(m, &a_data, &packed, &mut c);
        return Tensor::from_vec(c, (m, n), a.device());
    }
    a.matmul(&b.t()?)
}

struct ModernBertLocalAttentionOp {
    scale: f32,
    local_window: usize,
}

struct ModernBertRopeQkvOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModernBertActivation {
    Gelu,
    Silu,
}

struct ModernBertGatedActivationOp {
    intermediate_size: usize,
    activation: ModernBertActivation,
}

impl CustomOp3 for ModernBertLocalAttentionOp {
    fn name(&self) -> &'static str {
        "modernbert-local-attention"
    }

    fn cpu_fwd(
        &self,
        q_storage: &CpuStorage,
        q_layout: &Layout,
        k_storage: &CpuStorage,
        k_layout: &Layout,
        v_storage: &CpuStorage,
        v_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (q, shape) = checked_f32_contiguous("q", q_storage, q_layout)?;
        let (k, k_shape) = checked_f32_contiguous("k", k_storage, k_layout)?;
        let (v, v_shape) = checked_f32_contiguous("v", v_storage, v_layout)?;
        if k_shape != shape || v_shape != shape {
            candle_core::bail!("local attention expects q, k, v to have the same shape")
        }
        let dims = shape.dims();
        let [b, h, s, d] = dims else {
            candle_core::bail!(
                "local attention expects [batch, heads, seq, head_dim], got {shape:?}"
            )
        };
        if *s == 0 || *d == 0 {
            return Ok((CpuStorage::F32(Vec::new()), shape));
        }

        let mut out = vec![0f32; shape.elem_count()];
        let args = LocalAttentionArgs {
            q,
            k,
            v,
            s: *s,
            d: *d,
            half_window: self.local_window / 2,
            scale: self.scale,
        };
        let rows = *b * *h * *s;
        debug_assert_eq!(rows * *d, out.len());

        #[cfg(feature = "hybrid-dequant")]
        {
            out.par_chunks_mut(*d)
                .enumerate()
                .for_each(|(row_idx, row)| local_attention_row(&args, row_idx, row));
        }
        #[cfg(not(feature = "hybrid-dequant"))]
        {
            for (row_idx, row) in out.chunks_mut(*d).enumerate() {
                local_attention_row(&args, row_idx, row);
            }
        }

        Ok((CpuStorage::F32(out), shape))
    }
}

fn checked_f32_contiguous<'a>(
    name: &str,
    storage: &'a CpuStorage,
    layout: &Layout,
) -> Result<(&'a [f32], Shape)> {
    if storage.dtype() != DType::F32 {
        candle_core::bail!("{name} must be f32")
    }
    if !layout.is_contiguous() {
        candle_core::bail!("{name} must be contiguous")
    }
    let shape = layout.shape().clone();
    let start = layout.start_offset();
    let end = start + shape.elem_count();
    Ok((&storage.as_slice::<f32>()?[start..end], shape))
}

impl CustomOp3 for ModernBertRopeQkvOp {
    fn name(&self) -> &'static str {
        "modernbert-rope-qkv"
    }

    fn cpu_fwd(
        &self,
        qkv_storage: &CpuStorage,
        qkv_layout: &Layout,
        cos_storage: &CpuStorage,
        cos_layout: &Layout,
        sin_storage: &CpuStorage,
        sin_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (qkv, shape) = checked_f32_contiguous("qkv", qkv_storage, qkv_layout)?;
        let (cos, cos_shape) = checked_f32_contiguous("cos", cos_storage, cos_layout)?;
        let (sin, sin_shape) = checked_f32_contiguous("sin", sin_storage, sin_layout)?;
        let dims = shape.dims();
        let [three, b, h, s, d] = dims else {
            candle_core::bail!("RoPE QKV expects [3, batch, heads, seq, head_dim], got {shape:?}")
        };
        if *three != 3 {
            candle_core::bail!("RoPE QKV first dimension must be 3, got {three}")
        }
        if d % 2 != 0 {
            candle_core::bail!("RoPE QKV head_dim must be even, got {d}")
        }
        let half = d / 2;
        if cos_shape.dims() != [*s, half] || sin_shape.dims() != [*s, half] {
            candle_core::bail!(
                "RoPE QKV cos/sin shape mismatch: expected [{s},{half}], got {:?} and {:?}",
                cos_shape,
                sin_shape
            )
        }

        let plane_size = b * h * s * d;
        let rows_per_plane = b * h * s;
        let mut out = vec![0f32; shape.elem_count()];
        out[2 * plane_size..3 * plane_size].copy_from_slice(&qkv[2 * plane_size..3 * plane_size]);

        #[cfg(feature = "hybrid-dequant")]
        {
            out[..2 * plane_size]
                .par_chunks_mut(*d)
                .enumerate()
                .for_each(|(row_idx, dst)| {
                    let pos = (row_idx % rows_per_plane) % *s;
                    let src = &qkv[row_idx * *d..(row_idx + 1) * *d];
                    let cos = &cos[pos * half..(pos + 1) * half];
                    let sin = &sin[pos * half..(pos + 1) * half];
                    rope_row(src, cos, sin, dst);
                });
        }
        #[cfg(not(feature = "hybrid-dequant"))]
        {
            for (row_idx, dst) in out[..2 * plane_size].chunks_mut(*d).enumerate() {
                let pos = (row_idx % rows_per_plane) % *s;
                let src = &qkv[row_idx * *d..(row_idx + 1) * *d];
                let cos = &cos[pos * half..(pos + 1) * half];
                let sin = &sin[pos * half..(pos + 1) * half];
                rope_row(src, cos, sin, dst);
            }
        }

        Ok((CpuStorage::F32(out), shape))
    }
}

fn rope_row(src: &[f32], cos: &[f32], sin: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if cos.len() == 32 && is_x86_feature_detected!("avx512f") {
            unsafe {
                rope_row32_avx512(src.as_ptr(), cos.as_ptr(), sin.as_ptr(), dst.as_mut_ptr());
            }
            return;
        }
        if cos.len() == 32 && is_x86_feature_detected!("avx2") {
            unsafe {
                rope_row32_avx2(src.as_ptr(), cos.as_ptr(), sin.as_ptr(), dst.as_mut_ptr());
            }
            return;
        }
    }

    rope_row_scalar(src, cos, sin, dst);
}

fn rope_row_scalar(src: &[f32], cos: &[f32], sin: &[f32], dst: &mut [f32]) {
    let half = cos.len();
    for i in 0..half {
        let x1 = src[i];
        let x2 = src[half + i];
        let c = cos[i];
        let s = sin[i];
        dst[i] = x1 * c - x2 * s;
        dst[half + i] = x2 * c + x1 * s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rope_row32_avx2(src: *const f32, cos: *const f32, sin: *const f32, dst: *mut f32) {
    use std::arch::x86_64::*;

    for offset in (0..32).step_by(8) {
        let x1 = _mm256_loadu_ps(src.add(offset));
        let x2 = _mm256_loadu_ps(src.add(32 + offset));
        let c = _mm256_loadu_ps(cos.add(offset));
        let s = _mm256_loadu_ps(sin.add(offset));
        let x1c = _mm256_mul_ps(x1, c);
        let x2s = _mm256_mul_ps(x2, s);
        let x2c = _mm256_mul_ps(x2, c);
        let x1s = _mm256_mul_ps(x1, s);
        _mm256_storeu_ps(dst.add(offset), _mm256_sub_ps(x1c, x2s));
        _mm256_storeu_ps(dst.add(32 + offset), _mm256_add_ps(x2c, x1s));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn rope_row32_avx512(src: *const f32, cos: *const f32, sin: *const f32, dst: *mut f32) {
    use std::arch::x86_64::*;

    for offset in (0..32).step_by(16) {
        let x1 = _mm512_loadu_ps(src.add(offset));
        let x2 = _mm512_loadu_ps(src.add(32 + offset));
        let c = _mm512_loadu_ps(cos.add(offset));
        let s = _mm512_loadu_ps(sin.add(offset));
        let x1c = _mm512_mul_ps(x1, c);
        let x2s = _mm512_mul_ps(x2, s);
        let x2c = _mm512_mul_ps(x2, c);
        let x1s = _mm512_mul_ps(x1, s);
        _mm512_storeu_ps(dst.add(offset), _mm512_sub_ps(x1c, x2s));
        _mm512_storeu_ps(dst.add(32 + offset), _mm512_add_ps(x2c, x1s));
    }
}

impl CustomOp1 for ModernBertGatedActivationOp {
    fn name(&self) -> &'static str {
        "modernbert-gated-activation"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (src, shape) = checked_f32_contiguous("gated activation input", storage, layout)?;
        let Some((&last_dim, prefix)) = shape.dims().split_last() else {
            candle_core::bail!("gated activation input must have at least one dimension")
        };
        if last_dim != self.intermediate_size * 2 {
            candle_core::bail!(
                "gated activation last dimension mismatch: expected {}, got {last_dim}",
                self.intermediate_size * 2
            )
        }

        let rows = shape.elem_count() / last_dim;
        let mut out_shape = prefix.to_vec();
        out_shape.push(self.intermediate_size);
        let out_shape = Shape::from(out_shape);
        let mut out = vec![0f32; rows * self.intermediate_size];

        #[cfg(feature = "hybrid-dequant")]
        {
            out.par_chunks_mut(self.intermediate_size)
                .enumerate()
                .for_each(|(row_idx, dst)| {
                    let src =
                        &src[row_idx * last_dim..row_idx * last_dim + self.intermediate_size * 2];
                    gated_activation_row(src, self.intermediate_size, self.activation, dst);
                });
        }
        #[cfg(not(feature = "hybrid-dequant"))]
        {
            for (row_idx, dst) in out.chunks_mut(self.intermediate_size).enumerate() {
                let src = &src[row_idx * last_dim..row_idx * last_dim + self.intermediate_size * 2];
                gated_activation_row(src, self.intermediate_size, self.activation, dst);
            }
        }

        Ok((CpuStorage::F32(out), out_shape))
    }
}

fn gated_activation_row(
    src: &[f32],
    intermediate_size: usize,
    activation: ModernBertActivation,
    dst: &mut [f32],
) {
    let (gate, up) = src.split_at(intermediate_size);
    match activation {
        ModernBertActivation::Gelu => {
            for i in 0..intermediate_size {
                dst[i] = gelu_erf(gate[i]) * up[i];
            }
        }
        ModernBertActivation::Silu => {
            for i in 0..intermediate_size {
                dst[i] = silu(gate[i]) * up[i];
            }
        }
    }
}

#[inline(always)]
fn gelu_erf(x: f32) -> f32 {
    (candle_core::cpu::erf::erf_f32(x * std::f32::consts::FRAC_1_SQRT_2) + 1.0) * 0.5 * x
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

struct LocalAttentionArgs<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    s: usize,
    d: usize,
    half_window: usize,
    scale: f32,
}

fn local_attention_row(args: &LocalAttentionArgs<'_>, row_idx: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if args.d == 64 && is_x86_feature_detected!("avx512f") {
            unsafe {
                local_attention_row_avx512(args, row_idx, out);
            }
            return;
        }
        if args.d == 64 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                local_attention_row_avx2(args, row_idx, out);
            }
            return;
        }
    }

    local_attention_row_scalar(args, row_idx, out);
}

fn local_attention_row_scalar(args: &LocalAttentionArgs<'_>, row_idx: usize, out: &mut [f32]) {
    let i = row_idx % args.s;
    let bh = row_idx / args.s;
    let q_base = row_idx * args.d;
    let kv_base = bh * args.s * args.d;
    let j_start = i.saturating_sub(args.half_window);
    let j_end = (i + args.half_window + 1).min(args.s);
    let window_len = j_end - j_start;
    let mut scores = [0f32; 1024];

    let mut max_score = f32::NEG_INFINITY;
    for (offset, j) in (j_start..j_end).enumerate() {
        let k_base = kv_base + j * args.d;
        let mut dot = 0f32;
        for col in 0..args.d {
            dot += args.q[q_base + col] * args.k[k_base + col];
        }
        let score = dot * args.scale;
        scores[offset] = score;
        max_score = max_score.max(score);
    }

    let mut denom = 0f32;
    for score in scores[..window_len].iter_mut() {
        *score = (*score - max_score).exp();
        denom += *score;
    }

    let inv_denom = denom.recip();
    for (offset, j) in (j_start..j_end).enumerate() {
        let weight = scores[offset] * inv_denom;
        let v_base = kv_base + j * args.d;
        for col in 0..args.d {
            out[col] += weight * args.v[v_base + col];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn local_attention_row_avx2(args: &LocalAttentionArgs<'_>, row_idx: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;

    let i = row_idx % args.s;
    let bh = row_idx / args.s;
    let q_base = row_idx * 64;
    let kv_base = bh * args.s * 64;
    let j_start = i.saturating_sub(args.half_window);
    let j_end = (i + args.half_window + 1).min(args.s);
    let window_len = j_end - j_start;
    let mut scores = [0f32; 1024];

    let q_ptr = args.q.as_ptr().add(q_base);
    let mut max_score = f32::NEG_INFINITY;
    for (offset, j) in (j_start..j_end).enumerate() {
        let k_ptr = args.k.as_ptr().add(kv_base + j * 64);
        let score = dot64_avx2(q_ptr, k_ptr) * args.scale;
        scores[offset] = score;
        max_score = max_score.max(score);
    }

    let mut denom = 0f32;
    for score in scores[..window_len].iter_mut() {
        *score = (*score - max_score).exp();
        denom += *score;
    }

    let inv_denom = denom.recip();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    for (offset, j) in (j_start..j_end).enumerate() {
        let weight = _mm256_set1_ps(scores[offset] * inv_denom);
        let v_ptr = args.v.as_ptr().add(kv_base + j * 64);
        acc0 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr), acc0);
        acc1 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(8)), acc1);
        acc2 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(16)), acc2);
        acc3 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(24)), acc3);
        acc4 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(32)), acc4);
        acc5 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(40)), acc5);
        acc6 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(48)), acc6);
        acc7 = _mm256_fmadd_ps(weight, _mm256_loadu_ps(v_ptr.add(56)), acc7);
    }

    let out_ptr = out.as_mut_ptr();
    _mm256_storeu_ps(out_ptr, acc0);
    _mm256_storeu_ps(out_ptr.add(8), acc1);
    _mm256_storeu_ps(out_ptr.add(16), acc2);
    _mm256_storeu_ps(out_ptr.add(24), acc3);
    _mm256_storeu_ps(out_ptr.add(32), acc4);
    _mm256_storeu_ps(out_ptr.add(40), acc5);
    _mm256_storeu_ps(out_ptr.add(48), acc6);
    _mm256_storeu_ps(out_ptr.add(56), acc7);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn local_attention_row_avx512(
    args: &LocalAttentionArgs<'_>,
    row_idx: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let i = row_idx % args.s;
    let bh = row_idx / args.s;
    let q_base = row_idx * 64;
    let kv_base = bh * args.s * 64;
    let j_start = i.saturating_sub(args.half_window);
    let j_end = (i + args.half_window + 1).min(args.s);
    let window_len = j_end - j_start;
    let mut scores = [0f32; 1024];

    let q_ptr = args.q.as_ptr().add(q_base);
    let mut max_score = f32::NEG_INFINITY;
    for (offset, j) in (j_start..j_end).enumerate() {
        let k_ptr = args.k.as_ptr().add(kv_base + j * 64);
        let score = dot64_avx512(q_ptr, k_ptr) * args.scale;
        scores[offset] = score;
        max_score = max_score.max(score);
    }

    let mut denom = 0f32;
    for score in scores[..window_len].iter_mut() {
        *score = (*score - max_score).exp();
        denom += *score;
    }

    let inv_denom = denom.recip();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    for (offset, j) in (j_start..j_end).enumerate() {
        let weight = _mm512_set1_ps(scores[offset] * inv_denom);
        let v_ptr = args.v.as_ptr().add(kv_base + j * 64);
        acc0 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(v_ptr), acc0);
        acc1 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(v_ptr.add(16)), acc1);
        acc2 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(v_ptr.add(32)), acc2);
        acc3 = _mm512_fmadd_ps(weight, _mm512_loadu_ps(v_ptr.add(48)), acc3);
    }

    let out_ptr = out.as_mut_ptr();
    _mm512_storeu_ps(out_ptr, acc0);
    _mm512_storeu_ps(out_ptr.add(16), acc1);
    _mm512_storeu_ps(out_ptr.add(32), acc2);
    _mm512_storeu_ps(out_ptr.add(48), acc3);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot64_avx2(q: *const f32, k: *const f32) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    for offset in (0..64).step_by(8) {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(q.add(offset)),
            _mm256_loadu_ps(k.add(offset)),
            acc,
        );
    }
    let mut sum = [0f32; 8];
    _mm256_storeu_ps(sum.as_mut_ptr(), acc);
    sum.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot64_avx512(q: *const f32, k: *const f32) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_ps();
    for offset in (0..64).step_by(16) {
        acc = _mm512_fmadd_ps(
            _mm512_loadu_ps(q.add(offset)),
            _mm512_loadu_ps(k.add(offset)),
            acc,
        );
    }
    let mut sum = [0f32; 16];
    _mm512_storeu_ps(sum.as_mut_ptr(), acc);
    sum.iter().sum()
}

/// Fused sliding-window attention for ModernBERT local layers.
///
/// Computes `softmax(q @ k.T * scale) @ v` over the configured local window
/// without materializing a full `[batch, heads, seq, seq]` attention matrix.
pub fn modernbert_local_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    local_window: usize,
) -> Result<Tensor> {
    if !matches!(q.device(), Device::Cpu) || local_window + 1 > 1024 {
        candle_core::bail!("modernbert_local_attention only supports CPU windows up to 1024")
    }
    q.apply_op3_no_bwd(
        k,
        v,
        &ModernBertLocalAttentionOp {
            scale,
            local_window,
        },
    )
}

/// Rotate Q/K in a contiguous ModernBERT `[3, batch, heads, seq, head_dim]` QKV tensor.
///
/// V is copied through unchanged so callers can keep using the packed QKV layout.
pub fn modernbert_rope_qkv(qkv: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if !matches!(qkv.device(), Device::Cpu) {
        candle_core::bail!("modernbert_rope_qkv only supports CPU tensors")
    }
    qkv.apply_op3_no_bwd(cos, sin, &ModernBertRopeQkvOp)
}

/// Fused ModernBERT MLP activation: `activation(gate) * up`.
///
/// The input must have a final dimension of `2 * intermediate_size`, with gate first.
pub fn modernbert_gated_activation(
    xs: &Tensor,
    intermediate_size: usize,
    activation: ModernBertActivation,
) -> Result<Tensor> {
    if !matches!(xs.device(), Device::Cpu) {
        candle_core::bail!("modernbert_gated_activation only supports CPU tensors")
    }
    xs.apply_op1_no_bwd(&ModernBertGatedActivationOp {
        intermediate_size,
        activation,
    })
}

pub fn should_use_modernbert_local_attention(seq_len: usize, local_window: usize) -> bool {
    seq_len >= local_window.saturating_mul(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};
    use candle_nn::{Activation, Module};

    #[test]
    fn modernbert_local_attention_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 7usize, 4usize);
        let q: Vec<f32> = (0..b * h * s * d)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 19.0)
            .collect();
        let k: Vec<f32> = (0..b * h * s * d)
            .map(|i| ((i * 11 % 29) as f32 - 14.0) / 23.0)
            .collect();
        let v: Vec<f32> = (0..b * h * s * d)
            .map(|i| ((i * 7 % 37) as f32 - 18.0) / 17.0)
            .collect();
        let scale = 0.5f32;
        let local_window = 4usize;

        let q_t = Tensor::from_vec(q.clone(), (b, h, s, d), &device)?;
        let k_t = Tensor::from_vec(k.clone(), (b, h, s, d), &device)?;
        let v_t = Tensor::from_vec(v.clone(), (b, h, s, d), &device)?;
        let got = modernbert_local_attention(&q_t, &k_t, &v_t, scale, local_window)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut expected = vec![0f32; b * h * s * d];
        let args = LocalAttentionArgs {
            q: &q,
            k: &k,
            v: &v,
            s,
            d,
            half_window: local_window / 2,
            scale,
        };
        for (row_idx, row) in expected.chunks_mut(d).enumerate() {
            local_attention_row_scalar(&args, row_idx, row);
        }

        for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "mismatch at {idx}: got {got}, expected {expected}"
            );
        }

        Ok(())
    }

    #[test]
    fn modernbert_rope_qkv_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 5usize, 4usize);
        let half = d / 2;
        let qkv: Vec<f32> = (0..3 * b * h * s * d)
            .map(|i| ((i * 13 % 41) as f32 - 20.0) / 17.0)
            .collect();
        let cos: Vec<f32> = (0..s * half)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) / 11.0)
            .collect();
        let sin: Vec<f32> = (0..s * half)
            .map(|i| ((i * 5 % 23) as f32 - 11.0) / 13.0)
            .collect();

        let qkv_t = Tensor::from_vec(qkv.clone(), (3, b, h, s, d), &device)?;
        let cos_t = Tensor::from_vec(cos.clone(), (s, half), &device)?;
        let sin_t = Tensor::from_vec(sin.clone(), (s, half), &device)?;
        let got = modernbert_rope_qkv(&qkv_t, &cos_t, &sin_t)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let plane_size = b * h * s * d;
        let rows_per_plane = b * h * s;
        let mut expected = vec![0f32; qkv.len()];
        expected[2 * plane_size..3 * plane_size]
            .copy_from_slice(&qkv[2 * plane_size..3 * plane_size]);
        for row_idx in 0..2 * rows_per_plane {
            let pos = (row_idx % rows_per_plane) % s;
            let src = &qkv[row_idx * d..(row_idx + 1) * d];
            let dst = &mut expected[row_idx * d..(row_idx + 1) * d];
            rope_row_scalar(
                src,
                &cos[pos * half..(pos + 1) * half],
                &sin[pos * half..(pos + 1) * half],
                dst,
            );
        }

        for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "mismatch at {idx}: got {got}, expected {expected}"
            );
        }

        Ok(())
    }

    #[test]
    fn modernbert_gated_activation_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let (b, s, i) = (2usize, 3usize, 7usize);
        let input: Vec<f32> = (0..b * s * i * 2)
            .map(|idx| ((idx * 11 % 37) as f32 - 18.0) / 9.0)
            .collect();
        let input_t = Tensor::from_vec(input, (b, s, i * 2), &device)?;

        for (fast_activation, candle_activation) in [
            (ModernBertActivation::Gelu, Activation::Gelu),
            (ModernBertActivation::Silu, Activation::Silu),
        ] {
            let got = modernbert_gated_activation(&input_t, i, fast_activation)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let gate = input_t.narrow(D::Minus1, 0, i)?;
            let up = input_t.narrow(D::Minus1, i, i)?;
            let expected = candle_activation
                .forward(&gate)?
                .broadcast_mul(&up)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            for (idx, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "mismatch at {idx}: got {got}, expected {expected}"
                );
            }
        }

        Ok(())
    }
}
