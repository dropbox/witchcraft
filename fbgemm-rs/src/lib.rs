//! Pure Rust GEMM with SIMD micro-kernels ported from FBGEMM.
//!
//! Provides two GEMM paths:
//! - **FP32**: C = beta * C + A * B (f32 matrices, [`PackedMatrix`])
//! - **Quantized**: C_f32 = A_f32 × B_i8 (uint8×int8→int32→f32, [`quantized::PackedBMatrixI8`])

pub mod bf16;
pub mod gemm;
pub mod kernels;
pub mod pack;
pub mod partition;
#[cfg(feature = "quantized")]
pub mod quantized;

pub use bf16::sgemm_bf16;
pub use bf16::sgemm_bf16_simple;
pub use bf16::PackedMatrixBf16;
pub use pack::PackedMatrix;
#[cfg(feature = "quantized")]
pub use quantized::i8gemm_f32;
#[cfg(feature = "quantized")]
pub use quantized::i8gemm_f32_with_scratch;
#[cfg(feature = "quantized")]
pub use quantized::I8GemmScratch;
#[cfg(feature = "quantized")]
pub use quantized::PackedBMatrixI8;

/// Compute C = beta * C + A * B.
///
/// - `a`: M×K row-major matrix
/// - `packed_b`: pre-packed K×N matrix (see [`PackedMatrix`])
/// - `c`: M×N row-major output matrix
/// - `m`: number of rows in A and C
/// - `beta`: scaling factor for existing C values (0.0 to overwrite)
///
/// # Panics
///
/// Panics if slice lengths don't match the declared dimensions.
pub fn sgemm(m: usize, a: &[f32], packed_b: &PackedMatrix, beta: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    assert_eq!(a.len(), m * k, "a must have length m * k");
    assert_eq!(c.len(), m * n, "c must have length m * n");

    #[cfg(feature = "rayon")]
    {
        gemm::cblas_gemm_compute_par(m, a, packed_b, beta, c);
    }
    #[cfg(not(feature = "rayon"))]
    {
        gemm::cblas_gemm_compute(m, a, packed_b, beta, c);
    }
}

/// Compute C = A * B (overwriting C).
///
/// Convenience wrapper for `sgemm` with `beta = 0.0`.
pub fn sgemm_simple(m: usize, a: &[f32], packed_b: &PackedMatrix, c: &mut [f32]) {
    sgemm(m, a, packed_b, 0.0, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sgemm() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]   (3x2)
        // C = A * B = [[58, 64],
        //              [139, 154]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];

        let packed_b = PackedMatrix::new(3, 2, &b);
        assert_eq!(packed_b.k(), 3);
        assert_eq!(packed_b.n(), 2);

        sgemm_simple(2, &a, &packed_b, &mut c);

        assert!((c[0] - 58.0).abs() < 1e-4, "c[0] = {}", c[0]);
        assert!((c[1] - 64.0).abs() < 1e-4, "c[1] = {}", c[1]);
        assert!((c[2] - 139.0).abs() < 1e-4, "c[2] = {}", c[2]);
        assert!((c[3] - 154.0).abs() < 1e-4, "c[3] = {}", c[3]);
    }

    #[test]
    fn test_sgemm_with_beta() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = vec![3.0, 4.0, 5.0, 6.0]; // 2x2
        let mut c = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 ones

        let packed_b = PackedMatrix::new(2, 2, &b);

        // C = 2.0 * C + A * B = [[2+3, 2+4], [2+5, 2+6]] = [[5, 6], [7, 8]]
        sgemm(2, &a, &packed_b, 2.0, &mut c);

        assert!((c[0] - 5.0).abs() < 1e-4, "c[0] = {}", c[0]);
        assert!((c[1] - 6.0).abs() < 1e-4, "c[1] = {}", c[1]);
        assert!((c[2] - 7.0).abs() < 1e-4, "c[2] = {}", c[2]);
        assert!((c[3] - 8.0).abs() < 1e-4, "c[3] = {}", c[3]);
    }

    #[test]
    fn test_larger_matrix() {
        let m = 16;
        let k = 32;
        let n = 16;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];

        let packed_b = PackedMatrix::new(k, n, &b);
        sgemm_simple(m, &a, &packed_b, &mut c);

        // Verify against naive reference
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c_ref[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        for i in 0..m * n {
            assert!(
                (c[i] - c_ref[i]).abs() < 1e-2,
                "mismatch at {}: got {}, expected {}",
                i,
                c[i],
                c_ref[i]
            );
        }
    }
}
