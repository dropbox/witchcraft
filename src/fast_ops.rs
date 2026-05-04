//! Fast contiguous f32 ops bypassing candle's generic dispatch.
//!
//! Also provides [`PackedRight`] for efficient `A × B^T` using fbgemm-rs
//! when available, with transparent fallback to candle matmul.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};

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
