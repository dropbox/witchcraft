//! Column-tiled quantized matmul + fused gated-gelu.
//!
//! Provides `MatMul`, a drop-in replacement for candle's `QMatMul` that uses
//! column-tiled loops for better L1 cache behavior on x86.

use candle_core::backend::BackendStorage;
use candle_core::quantized::k_quants::*;
use candle_core::quantized::{GgmlDType, GgmlType, QTensor};
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Module, Result, Shape, Tensor};
use rayon::prelude::*;
use std::sync::Arc;

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

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
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
                let gate_col =
                    &rhs_gate[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let up_col = &rhs_up[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let gate = T::vec_dot(k, gate_col, lhs_row);
                let up = T::vec_dot(k, up_col, lhs_row);
                let gate = 0.5 * gate
                    * (1.0
                        + f32::tanh(0.7978845608_f32 * gate * (1.0 + 0.044715 * gate * gate)));
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

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
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

// ---- MatMul: drop-in replacement for QMatMul ----

/// Drop-in replacement for `candle_core::quantized::QMatMul` that uses
/// column-tiled matmul for quantized weights on CPU.
#[derive(Clone, Debug)]
pub enum MatMul {
    QTensor(Arc<QTensor>),
    Tensor(Tensor),
}

impl MatMul {
    pub fn from_qtensor(qt: Arc<QTensor>) -> Self {
        Self::QTensor(qt)
    }
}

impl Module for MatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(&QTiledOp(t.clone())),
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
        (MatMul::QTensor(w0), MatMul::QTensor(w1)) => {
            let op = QGatedMatMul(w0.clone(), w1.clone());
            xs.apply_op1_no_bwd(&op)
        }
        _ => {
            let gate = w0.forward(xs)?.gelu()?;
            let up = w1.forward(xs)?;
            gate.broadcast_mul(&up)
        }
    }
}
