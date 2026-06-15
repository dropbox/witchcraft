//! Column-tiled quantized matmul + fused gated-gelu.
//!
//! Provides `MatMul`, a drop-in replacement for candle's `QMatMul` that uses
//! column-tiled loops for better L1 cache behavior on x86.
//! Pre-dequantized weights use fbgemm-rs packed GEMM when available:
//! int8 on AVX512-VNNI CPUs, BF16 elsewhere.

use candle_core::backend::BackendStorage;
use candle_core::quantized::k_quants::*;
use candle_core::quantized::{GgmlDType, GgmlType, QTensor};
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Module, Result, Shape, Tensor};
#[cfg(feature = "fbgemm")]
use fbgemm_rs::{PackedBMatrixI8, PackedMatrixBf16};
use rayon::prelude::*;
use std::sync::Arc;
#[cfg(feature = "fbgemm")]
use std::sync::Mutex;

fn as_block_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    let ptr = data.as_ptr();
    debug_assert_eq!(data.len() % size, 0);
    debug_assert_eq!((ptr as usize) % std::mem::align_of::<T>(), 0);
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

// ---- Column-tiled matmul ----

fn tiled_matmul_inner<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) {
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    for row_idx in 0..m {
        let lhs_b_row = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let lhs_row = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs_row, lhs_b_row);
    }

    let tile_n = 128.min(n);
    let tile_starts: Vec<usize> = (0..n).step_by(tile_n).collect();
    let dst_ptr = dst.as_mut_ptr() as usize;
    tile_starts.into_par_iter().for_each(|tile_start| {
        let tile_end = (tile_start + tile_n).min(n);
        // SAFETY: Non-overlapping column tiles — no two threads write same dst element.
        let dst = dst_ptr as *mut f32;
        for row_idx in 0..m {
            let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            for col_idx in tile_start..tile_end {
                let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                unsafe {
                    *dst.add(row_idx * n + col_idx) = T::vec_dot(k, rhs_col, lhs_row);
                }
            }
        }
    });
}

struct QTiledOp(Arc<QTensor>);

impl CustomOp1 for QTiledOp {
    fn name(&self) -> &'static str {
        "qtiled-matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (n, k) = self.0.shape().dims2()?;
        if src_shape.rank() < 2 {
            candle_core::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            candle_core::bail!(
                "input tensor {layout:?} incompatible with {:?}",
                self.0.shape()
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let m = dst_shape.elem_count() / n;

        if storage.dtype() != DType::F32 {
            candle_core::bail!("QTiledOp only supports f32 input")
        }
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        let data = self.0.data()?;
        macro_rules! dispatch {
            ($ty:ty) => {
                tiled_matmul_inner::<$ty>(
                    (m, k, n),
                    slice,
                    as_block_slice::<$ty>(&data),
                    &mut dst_storage,
                )
            };
        }
        match self.0.dtype() {
            GgmlDType::Q4K => dispatch!(BlockQ4K),
            GgmlDType::Q5K => dispatch!(BlockQ5K),
            GgmlDType::Q6K => dispatch!(BlockQ6K),
            GgmlDType::Q8K => dispatch!(BlockQ8K),
            GgmlDType::Q2K => dispatch!(BlockQ2K),
            GgmlDType::Q3K => dispatch!(BlockQ3K),
            GgmlDType::Q4_0 => dispatch!(BlockQ4_0),
            GgmlDType::Q5_0 => dispatch!(BlockQ5_0),
            GgmlDType::Q8_0 => dispatch!(BlockQ8_0),
            dt => candle_core::bail!("QTiledOp: unsupported dtype {dt:?}"),
        }

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

// ---- Fast tanh via Padé [3,3] rational approximation ----
// Pure polynomial ratio (no transcendentals) — auto-vectorizes with AVX2.

#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x = x.clamp(-4.97, 4.97);
    let x2 = x * x;
    x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
        / (135135.0 + x2 * (62370.0 + x2 * (3150.0 + 28.0 * x2)))
}

// ---- Fused gated-gelu matmul ----

fn fused_gated_gelu_inner<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_gate: &[T],
    rhs_up: &[T],
    dst: &mut [f32],
) {
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    for row_idx in 0..m {
        let lhs_b_row = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let lhs_row = &lhs[row_idx * k..(row_idx + 1) * k];
        T::VecDotType::from_float(lhs_row, lhs_b_row);
    }

    // Half-sized tiles since we read from 2 weight matrices per column.
    let tile_n = 64.min(n);
    let tile_starts: Vec<usize> = (0..n).step_by(tile_n).collect();
    let dst_ptr = dst.as_mut_ptr() as usize;
    tile_starts.into_par_iter().for_each(|tile_start| {
        let tile_end = (tile_start + tile_n).min(n);
        let dst = dst_ptr as *mut f32;
        for row_idx in 0..m {
            let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            for col_idx in tile_start..tile_end {
                let gate_col = &rhs_gate[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let up_col = &rhs_up[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let gate = T::vec_dot(k, gate_col, lhs_row);
                let up = T::vec_dot(k, up_col, lhs_row);
                let gate = 0.5
                    * gate
                    * (1.0 + fast_tanh(0.7978845608_f32 * gate * (1.0 + 0.044715 * gate * gate)));
                unsafe {
                    *dst.add(row_idx * n + col_idx) = gate * up;
                }
            }
        }
    });
}

struct QGatedMatMul(Arc<QTensor>, Arc<QTensor>);

impl CustomOp1 for QGatedMatMul {
    fn name(&self) -> &'static str {
        "qgated-matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (n, k) = self.0.shape().dims2()?;
        let (n1, k1) = self.1.shape().dims2()?;
        if n != n1 || k != k1 {
            candle_core::bail!("gated matmul weight shape mismatch: ({n},{k}) vs ({n1},{k1})")
        }
        if src_shape.rank() < 2 {
            candle_core::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            candle_core::bail!(
                "input tensor {layout:?} incompatible with {:?}",
                self.0.shape()
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let m = dst_shape.elem_count() / n;

        if storage.dtype() != DType::F32 {
            candle_core::bail!("QGatedMatMul only supports f32 input")
        }
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        let gate_data = self.0.data()?;
        let up_data = self.1.data()?;

        macro_rules! dispatch {
            ($ty:ty) => {
                fused_gated_gelu_inner::<$ty>(
                    (m, k, n),
                    slice,
                    as_block_slice::<$ty>(&gate_data),
                    as_block_slice::<$ty>(&up_data),
                    &mut dst_storage,
                )
            };
        }
        match self.0.dtype() {
            GgmlDType::Q4K => dispatch!(BlockQ4K),
            GgmlDType::Q5K => dispatch!(BlockQ5K),
            GgmlDType::Q6K => dispatch!(BlockQ6K),
            GgmlDType::Q8K => dispatch!(BlockQ8K),
            GgmlDType::Q2K => dispatch!(BlockQ2K),
            GgmlDType::Q3K => dispatch!(BlockQ3K),
            GgmlDType::Q4_0 => dispatch!(BlockQ4_0),
            GgmlDType::Q5_0 => dispatch!(BlockQ5_0),
            GgmlDType::Q8_0 => dispatch!(BlockQ8_0),
            dt => candle_core::bail!("QGatedMatMul: unsupported dtype {dt:?}"),
        }

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

// ---- fbgemm-rs packed GEMM via pre-packed weights ----

#[cfg(feature = "fbgemm")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FbgemmPacking {
    Bf16,
    I8,
}

#[cfg(feature = "fbgemm")]
fn fbgemm_packing_mode() -> FbgemmPacking {
    if cpu_has_avx512_vnni_path() {
        FbgemmPacking::I8
    } else {
        FbgemmPacking::Bf16
    }
}

#[cfg(all(feature = "fbgemm", target_arch = "x86_64"))]
fn cpu_has_avx512_vnni_path() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vnni")
}

#[cfg(all(feature = "fbgemm", not(target_arch = "x86_64")))]
fn cpu_has_avx512_vnni_path() -> bool {
    false
}

#[cfg(feature = "fbgemm")]
struct FbgemmBf16Op(Arc<PackedMatrixBf16>);

#[cfg(feature = "fbgemm")]
struct FbgemmI8Op(
    Arc<PackedBMatrixI8>,
    f32,
    Arc<Mutex<fbgemm_rs::I8GemmScratch>>,
);

#[cfg(feature = "fbgemm")]
struct FbgemmBf16GatedGeluOp(Arc<PackedMatrixBf16>, Arc<PackedMatrixBf16>);

#[cfg(feature = "fbgemm")]
struct FbgemmI8GatedGeluOp(
    Arc<PackedBMatrixI8>,
    f32,
    Arc<Mutex<fbgemm_rs::I8GemmScratch>>,
    Arc<PackedBMatrixI8>,
    f32,
    Arc<Mutex<fbgemm_rs::I8GemmScratch>>,
);

#[cfg(feature = "fbgemm")]
fn checked_f32_input<'a>(
    op_name: &str,
    storage: &'a CpuStorage,
    layout: &Layout,
    k: usize,
    n: usize,
) -> Result<(&'a [f32], Shape, usize)> {
    if !layout.is_contiguous() {
        candle_core::bail!("input tensor is not contiguous {layout:?}")
    }
    let src_shape = layout.shape();
    if src_shape.rank() < 2 {
        candle_core::bail!("input tensor has only one dimension {layout:?}")
    }
    let mut dst_shape = src_shape.dims().to_vec();
    let last_k = dst_shape.pop().unwrap();
    if last_k != k {
        candle_core::bail!("input tensor {layout:?} incompatible with packed matrix ({k}x{n})")
    }
    dst_shape.push(n);
    let dst_shape = Shape::from(dst_shape);
    let m = dst_shape.elem_count() / n;

    if storage.dtype() != DType::F32 {
        candle_core::bail!("{op_name} only supports f32 input")
    }
    let slice = storage.as_slice::<f32>()?;
    let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
    Ok((slice, dst_shape, m))
}

#[cfg(feature = "fbgemm")]
impl CustomOp1 for FbgemmBf16Op {
    fn name(&self) -> &'static str {
        "fbgemm-bf16-matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let k = self.0.k();
        let n = self.0.n();
        let (slice, dst_shape, m) = checked_f32_input(self.name(), storage, layout, k, n)?;
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        fbgemm_rs::sgemm_bf16_simple(m, slice, &self.0, &mut dst_storage);

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

#[cfg(feature = "fbgemm")]
impl CustomOp1 for FbgemmI8Op {
    fn name(&self) -> &'static str {
        "fbgemm-i8-matmul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let k = self.0.k();
        let n = self.0.n();
        let (slice, dst_shape, m) = checked_f32_input(self.name(), storage, layout, k, n)?;
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        let mut scratch = self
            .2
            .lock()
            .map_err(|_| candle_core::Error::Msg("i8 GEMM scratch lock poisoned".into()))?;
        fbgemm_rs::i8gemm_f32_with_scratch(
            m,
            slice,
            &self.0,
            self.1,
            &mut dst_storage,
            &mut scratch,
        );

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

#[cfg(feature = "fbgemm")]
impl CustomOp1 for FbgemmBf16GatedGeluOp {
    fn name(&self) -> &'static str {
        "fbgemm-bf16-gated-gelu"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let k = self.0.k();
        let n = self.0.n();
        if self.1.k() != k || self.1.n() != n {
            candle_core::bail!(
                "gated fbgemm weight shape mismatch: gate is {}x{}, up is {}x{}",
                k,
                n,
                self.1.k(),
                self.1.n()
            )
        }
        let (slice, dst_shape, m) = checked_f32_input(self.name(), storage, layout, k, n)?;
        let mut gate = vec![0f32; dst_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        fbgemm_rs::sgemm_bf16_simple(m, slice, &self.0, &mut gate);
        fbgemm_rs::sgemm_bf16_simple(m, slice, &self.1, &mut dst_storage);

        for (gate, up) in gate.into_iter().zip(dst_storage.iter_mut()) {
            let gate = 0.5
                * gate
                * (1.0 + fast_tanh(0.7978845608_f32 * gate * (1.0 + 0.044715 * gate * gate)));
            *up *= gate;
        }

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

#[cfg(feature = "fbgemm")]
impl CustomOp1 for FbgemmI8GatedGeluOp {
    fn name(&self) -> &'static str {
        "fbgemm-i8-gated-gelu"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let k = self.0.k();
        let n = self.0.n();
        if self.3.k() != k || self.3.n() != n {
            candle_core::bail!(
                "gated fbgemm weight shape mismatch: gate is {}x{}, up is {}x{}",
                k,
                n,
                self.3.k(),
                self.3.n()
            )
        }
        let (slice, dst_shape, m) = checked_f32_input(self.name(), storage, layout, k, n)?;
        let mut gate = vec![0f32; dst_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];

        {
            let mut scratch = self
                .2
                .lock()
                .map_err(|_| candle_core::Error::Msg("i8 GEMM scratch lock poisoned".into()))?;
            fbgemm_rs::i8gemm_f32_with_scratch(m, slice, &self.0, self.1, &mut gate, &mut scratch);
        }
        {
            let mut scratch = self
                .5
                .lock()
                .map_err(|_| candle_core::Error::Msg("i8 GEMM scratch lock poisoned".into()))?;
            fbgemm_rs::i8gemm_f32_with_scratch(
                m,
                slice,
                &self.3,
                self.4,
                &mut dst_storage,
                &mut scratch,
            );
        }

        for (gate, up) in gate.into_iter().zip(dst_storage.iter_mut()) {
            let gate = 0.5
                * gate
                * (1.0 + fast_tanh(0.7978845608_f32 * gate * (1.0 + 0.044715 * gate * gate)));
            *up *= gate;
        }

        Ok((CpuStorage::F32(dst_storage), dst_shape))
    }
}

#[cfg(feature = "fbgemm")]
fn quantize_weight_i8_transposed(k: usize, n: usize, data: &[f32]) -> (PackedBMatrixI8, f32) {
    let max_abs = data.iter().fold(0f32, |acc, value| acc.max(value.abs()));
    if max_abs == 0.0 {
        let quantized = vec![0; k * n];
        return (PackedBMatrixI8::new(k, n, &quantized), 1.0);
    }
    let scale = max_abs / 127.0;
    let inv_scale = scale.recip();
    let mut quantized = vec![0i8; k * n];
    for row in 0..k {
        for col in 0..n {
            let value = data[col * k + row];
            quantized[row * n + col] = (value * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (PackedBMatrixI8::new(k, n, &quantized), scale)
}

// ---- MatMul: drop-in replacement for QMatMul ----

/// Drop-in replacement for `candle_core::quantized::QMatMul` that uses
/// column-tiled matmul for quantized weights. Pre-dequantized weights use
/// fbgemm-rs packed GEMM when available, otherwise plain candle matmul.
pub enum MatMul {
    QTensor(Arc<QTensor>),
    #[cfg(feature = "fbgemm")]
    PackedBf16(Arc<PackedMatrixBf16>),
    #[cfg(feature = "fbgemm")]
    PackedI8(
        Arc<PackedBMatrixI8>,
        f32,
        Arc<Mutex<fbgemm_rs::I8GemmScratch>>,
    ),
    Tensor(Tensor),
}

impl Clone for MatMul {
    fn clone(&self) -> Self {
        match self {
            Self::QTensor(qt) => Self::QTensor(qt.clone()),
            #[cfg(feature = "fbgemm")]
            Self::PackedBf16(p) => Self::PackedBf16(p.clone()),
            #[cfg(feature = "fbgemm")]
            Self::PackedI8(p, scale, scratch) => Self::PackedI8(p.clone(), *scale, scratch.clone()),
            Self::Tensor(t) => Self::Tensor(t.clone()),
        }
    }
}

impl std::fmt::Debug for MatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::QTensor(qt) => f.debug_tuple("QTensor").field(qt).finish(),
            #[cfg(feature = "fbgemm")]
            Self::PackedBf16(p) => write!(f, "PackedBf16({}x{})", p.k(), p.n()),
            #[cfg(feature = "fbgemm")]
            Self::PackedI8(p, _, _) => write!(f, "PackedI8({}x{})", p.k(), p.n()),
            Self::Tensor(t) => write!(f, "Tensor({:?})", t.shape()),
        }
    }
}

impl MatMul {
    pub fn from_qtensor(qt: Arc<QTensor>) -> Self {
        Self::QTensor(qt)
    }

    /// Store a dequantized [N, K] weight tensor in an fbgemm-packed format.
    pub fn from_tensor(t: Tensor) -> Self {
        #[cfg(feature = "fbgemm")]
        {
            // fbgemm is CPU-only; fall back to plain tensor for Metal/GPU
            if matches!(t.device(), candle_core::Device::Cpu) {
                let (n, k) = t
                    .dims2()
                    .expect("weight must be 2D for MatMul::from_tensor");
                let data = t
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .expect("weight to f32");
                match fbgemm_packing_mode() {
                    FbgemmPacking::Bf16 => {
                        let packed = PackedMatrixBf16::from_transposed(k, n, &data);
                        Self::PackedBf16(Arc::new(packed))
                    }
                    FbgemmPacking::I8 => {
                        let (packed, scale) = quantize_weight_i8_transposed(k, n, &data);
                        Self::PackedI8(
                            Arc::new(packed),
                            scale,
                            Arc::new(Mutex::new(fbgemm_rs::I8GemmScratch::new())),
                        )
                    }
                }
            } else {
                Self::Tensor(t)
            }
        }
        #[cfg(not(feature = "fbgemm"))]
        {
            Self::Tensor(t)
        }
    }
}

impl Module for MatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => {
                // QTiledOp custom op is CPU-only; use candle's QMatMul on Metal
                if matches!(xs.device(), candle_core::Device::Metal(_)) {
                    use candle_core::quantized::QMatMul as CandleQMatMul;
                    let qmm = CandleQMatMul::from_arc(t.clone())?;
                    qmm.forward(xs)
                } else {
                    xs.apply_op1_no_bwd(&QTiledOp(t.clone()))
                }
            }
            #[cfg(feature = "fbgemm")]
            Self::PackedBf16(p) => xs.apply_op1_no_bwd(&FbgemmBf16Op(p.clone())),
            #[cfg(feature = "fbgemm")]
            Self::PackedI8(p, scale, scratch) => {
                xs.apply_op1_no_bwd(&FbgemmI8Op(p.clone(), *scale, scratch.clone()))
            }
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
        }
    }
}

/// Fused gated-gelu: `gelu(xs @ w0.T) * (xs @ w1.T)`.
pub fn forward_gated_gelu(w0: &MatMul, w1: &MatMul, xs: &Tensor) -> Result<Tensor> {
    match (w0, w1) {
        #[cfg(feature = "fbgemm")]
        (MatMul::PackedBf16(w0), MatMul::PackedBf16(w1)) => {
            let op = FbgemmBf16GatedGeluOp(w0.clone(), w1.clone());
            xs.apply_op1_no_bwd(&op)
        }
        #[cfg(feature = "fbgemm")]
        (MatMul::PackedI8(w0, scale0, scratch0), MatMul::PackedI8(w1, scale1, scratch1)) => {
            let op = FbgemmI8GatedGeluOp(
                w0.clone(),
                *scale0,
                scratch0.clone(),
                w1.clone(),
                *scale1,
                scratch1.clone(),
            );
            xs.apply_op1_no_bwd(&op)
        }
        (MatMul::QTensor(w0_qt), MatMul::QTensor(w1_qt)) => {
            // QGatedMatMul custom op is CPU-only; fall back to unfused path on Metal
            if matches!(xs.device(), candle_core::Device::Metal(_)) {
                use candle_core::quantized::QMatMul as CandleQMatMul;
                let qmm0 = CandleQMatMul::from_arc(w0_qt.clone())?;
                let qmm1 = CandleQMatMul::from_arc(w1_qt.clone())?;
                let gate = qmm0.forward(xs)?.gelu()?;
                let up = qmm1.forward(xs)?;
                gate.broadcast_mul(&up)
            } else {
                let op = QGatedMatMul(w0_qt.clone(), w1_qt.clone());
                xs.apply_op1_no_bwd(&op)
            }
        }
        _ => {
            let gate = w0.forward(xs)?.gelu()?;
            let up = w1.forward(xs)?;
            gate.broadcast_mul(&up)
        }
    }
}
