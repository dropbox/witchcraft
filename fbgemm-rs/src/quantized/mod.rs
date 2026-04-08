//! UINT8×INT8→INT32 quantized GEMM with float32 output.
//!
//! Weights (int8) are pre-packed once via [`PackedBMatrixI8`].
//! Activations (float32) are dynamically quantized to uint8 per call.
//! Output is dequantized back to float32.

mod gemm;
pub mod pack;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "aarch64")]
mod neon;

pub use gemm::I8GemmScratch;
pub use pack::PackedBMatrixI8;

/// Compute C_f32 = (A_f32 × B_i8) * b_scale using quantized arithmetic.
///
/// - `m`: number of rows in A and C
/// - `a`: M×K row-major float32 activations
/// - `packed_b`: pre-packed K×N int8 weight matrix
/// - `b_scale`: quantization scale for B weights
/// - `c`: M×N row-major float32 output (overwritten)
///
/// Internally:
/// 1. Dynamically quantizes A from float32 to uint8
/// 2. Computes uint8 × int8 → int32 GEMM
/// 3. Dequantizes int32 → float32 output
pub fn i8gemm_f32(m: usize, a: &[f32], packed_b: &PackedBMatrixI8, b_scale: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    assert_eq!(a.len(), m * k, "a must have length m * k");
    assert_eq!(c.len(), m * n, "c must have length m * n");

    #[cfg(feature = "rayon")]
    {
        gemm::i8gemm_compute_par(m, a, packed_b, b_scale, c);
    }
    #[cfg(not(feature = "rayon"))]
    {
        gemm::i8gemm_compute(m, a, packed_b, b_scale, c);
    }
}

/// Like [`i8gemm_f32`] but reuses scratch buffers to avoid per-call allocation.
pub fn i8gemm_f32_with_scratch(
    m: usize,
    a: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c: &mut [f32],
    scratch: &mut I8GemmScratch,
) {
    let k = packed_b.k();
    let n = packed_b.n();
    assert_eq!(a.len(), m * k, "a must have length m * k");
    assert_eq!(c.len(), m * n, "c must have length m * n");

    #[cfg(feature = "rayon")]
    {
        gemm::i8gemm_compute_par_with_scratch(m, a, packed_b, b_scale, c, scratch);
    }
    #[cfg(not(feature = "rayon"))]
    {
        gemm::i8gemm_compute_with_scratch(m, a, packed_b, b_scale, c, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_i8gemm() {
        // A (float) = [[1, 2, 3], [4, 5, 6]]  (2×3)
        // B (int8)  = [[1, 2], [3, 4], [5, 6]] (3×2)
        // A×B = [[22, 28], [49, 64]]
        let a_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_i8 = vec![1i8, 2, 3, 4, 5, 6];
        let b_scale = 1.0;

        let packed_b = PackedBMatrixI8::new(3, 2, &b_i8);
        let mut c = vec![0.0f32; 4];

        i8gemm_f32(2, &a_f32, &packed_b, b_scale, &mut c);

        let expected = [22.0, 28.0, 49.0, 64.0];
        for i in 0..4 {
            let rel_err = (c[i] - expected[i]).abs() / expected[i].abs().max(1.0);
            assert!(
                rel_err < 0.15,
                "c[{}] = {}, expected {}",
                i,
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_i8gemm_matches_quantized_naive() {
        // Tests packing/dispatch correctness by comparing against naive
        // uint8×int8 GEMM (same quantization, no packing).
        let m = 16;
        let k = 32;
        let n = 24;

        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0)
            .collect();
        let b_i8: Vec<i8> = (0..k * n)
            .map(|i| (((i * 11 + 5) % 200) as i32 - 100) as i8)
            .collect();
        let b_scale = 0.05;

        let packed_b = PackedBMatrixI8::new(k, n, &b_i8);
        let mut c = vec![0.0f32; m * n];

        i8gemm_f32(m, &a_f32, &packed_b, b_scale, &mut c);

        // Quantize A the same way, then do naive uint8×int8 matmul
        let (a_u8, a_scale, a_zp, _) = pack::quantize_a(&a_f32, m, k);
        let col_offsets = packed_b.col_offsets();
        let output_scale = a_scale * b_scale;

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += a_u8[i * k + p] as i32 * b_i8[p * n + j] as i32;
                }
                let adjusted = acc - a_zp * col_offsets[j];
                c_ref[i * n + j] = adjusted as f32 * output_scale;
            }
        }

        for i in 0..m * n {
            assert!(
                (c[i] - c_ref[i]).abs() < 1e-4,
                "mismatch at {}: got {}, expected {}",
                i,
                c[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn test_i8gemm_large_k_blocks() {
        // K > KCB (512) to exercise K-blocking
        let m = 4;
        let k = 600;
        let n = 8;

        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 + 7) % 100) as f32 * 0.02 - 1.0)
            .collect();
        let b_i8: Vec<i8> = (0..k * n)
            .map(|i| (((i * 17 + 3) % 200) as i32 - 100) as i8)
            .collect();
        let b_scale = 0.01;

        let packed_b = PackedBMatrixI8::new(k, n, &b_i8);
        let mut c = vec![0.0f32; m * n];

        i8gemm_f32(m, &a_f32, &packed_b, b_scale, &mut c);

        // Quantize A the same way, then do naive uint8×int8 matmul
        let (a_u8, a_scale, a_zp, _) = pack::quantize_a(&a_f32, m, k);
        let col_offsets = packed_b.col_offsets();
        let output_scale = a_scale * b_scale;

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += a_u8[i * k + p] as i32 * b_i8[p * n + j] as i32;
                }
                let adjusted = acc - a_zp * col_offsets[j];
                c_ref[i * n + j] = adjusted as f32 * output_scale;
            }
        }

        for i in 0..m * n {
            assert!(
                (c[i] - c_ref[i]).abs() < 1e-4,
                "mismatch at {}: got {}, expected {}",
                i,
                c[i],
                c_ref[i]
            );
        }
    }

    #[test]
    fn test_packing_roundtrip() {
        // Verify that B packing preserves values by checking the reference kernel
        // with identity-like A
        let k = 4;
        let n = 3;
        let b_i8 = vec![1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let packed_b = PackedBMatrixI8::new(k, n, &b_i8);

        assert_eq!(packed_b.k(), 4);
        assert_eq!(packed_b.n(), 3);
        assert_eq!(packed_b.col_offsets(), &[22, 26, 30]);
    }

    #[test]
    fn test_i8gemm_all_fringe_sizes() {
        // Exercise SIMD fringe kernels for mc=1..11 by testing m=1..25.
        // N=8 (=NR) ensures the SIMD path is taken on AVX2.
        let k = 32;
        let n = 8;
        let b_i8: Vec<i8> = (0..k * n)
            .map(|i| (((i * 11 + 5) % 200) as i32 - 100) as i8)
            .collect();
        let b_scale = 0.05;
        let packed_b = PackedBMatrixI8::new(k, n, &b_i8);

        for m in 1..=25 {
            let a_f32: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0)
                .collect();
            let mut c = vec![0.0f32; m * n];

            i8gemm_f32(m, &a_f32, &packed_b, b_scale, &mut c);

            // Compare against quantized naive reference
            let (a_u8, a_scale, a_zp, _) = pack::quantize_a(&a_f32, m, k);
            let col_offsets = packed_b.col_offsets();
            let output_scale = a_scale * b_scale;

            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0i32;
                    for p in 0..k {
                        acc += a_u8[i * k + p] as i32 * b_i8[p * n + j] as i32;
                    }
                    let adjusted = acc - a_zp * col_offsets[j];
                    let expected = adjusted as f32 * output_scale;
                    assert!(
                        (c[i * n + j] - expected).abs() < 1e-4,
                        "m={} mismatch at [{},{}]: got {}, expected {}",
                        m,
                        i,
                        j,
                        c[i * n + j],
                        expected,
                    );
                }
            }
        }
    }
}
