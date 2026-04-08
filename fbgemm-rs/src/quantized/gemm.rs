use super::pack::*;

#[cfg(target_arch = "x86_64")]
use super::avx2;

#[cfg(target_arch = "aarch64")]
use super::neon;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Reusable scratch buffers for quantized GEMM.
/// Create once and pass to [`i8gemm_compute_with_scratch`] /
/// [`i8gemm_compute_par_with_scratch`] to avoid per-call allocation.
pub struct I8GemmScratch {
    a_u8: Vec<u8>,
    row_offsets: Vec<i32>,
    c_i32: Vec<i32>,
}

impl I8GemmScratch {
    pub fn new() -> Self {
        Self {
            a_u8: Vec::new(),
            row_offsets: Vec::new(),
            c_i32: Vec::new(),
        }
    }
}

/// Detect SIMD capability flags at runtime.
struct SimdFlags {
    #[cfg(target_arch = "x86_64")]
    avx2: bool,
    #[cfg(target_arch = "aarch64")]
    neon_dotprod: bool,
}

impl SimdFlags {
    fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "aarch64")]
            neon_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        }
    }

    fn use_simd(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self.avx2
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.neon_dotprod
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Effective zero point: shifted by 128 on the Neon SDOT path.
    fn effective_zp(&self, a_zero_point: i32) -> i32 {
        #[cfg(target_arch = "aarch64")]
        if self.neon_dotprod {
            return a_zero_point - 128;
        }
        a_zero_point
    }
}

// SAFETY: SimdFlags is read-only after construction.
unsafe impl Sync for SimdFlags {}

/// Main dispatcher for quantized GEMM: float32 activations × int8 weights → float32 output.
#[cfg(not(feature = "rayon"))]
pub fn i8gemm_compute(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
) {
    i8gemm_compute_with_scratch(
        m,
        a_float,
        packed_b,
        b_scale,
        c_float,
        &mut I8GemmScratch::new(),
    );
}

/// Like [`i8gemm_compute`] but reuses caller-provided scratch buffers.
#[cfg(not(feature = "rayon"))]
pub fn i8gemm_compute_with_scratch(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
    scratch: &mut I8GemmScratch,
) {
    let k = packed_b.k();
    let n = packed_b.n();

    let (a_scale, a_zero_point) =
        quantize_a_into(a_float, m, k, &mut scratch.a_u8, &mut scratch.row_offsets);
    scratch.c_i32.resize(m * n, 0);
    scratch.c_i32.fill(0);
    let flags = SimdFlags::detect();

    for mb_start in (0..m).step_by(MCB) {
        let mc = std::cmp::min(MCB, m - mb_start);
        process_m_block(
            mb_start,
            mc,
            &scratch.a_u8,
            k,
            n,
            packed_b,
            &mut scratch.c_i32,
            &flags,
        );
    }

    dequantize(
        m,
        n,
        &scratch.c_i32,
        c_float,
        a_scale,
        b_scale,
        flags.effective_zp(a_zero_point),
        packed_b.col_offsets(),
    );
}

/// Parallel version of [`i8gemm_compute`] using rayon.
///
/// M-blocks are dispatched across threads. Each writes to disjoint rows of the
/// accumulation buffer, so no synchronization is needed.
#[cfg(feature = "rayon")]
pub fn i8gemm_compute_par(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
) {
    i8gemm_compute_par_with_scratch(
        m,
        a_float,
        packed_b,
        b_scale,
        c_float,
        &mut I8GemmScratch::new(),
    );
}

/// Like [`i8gemm_compute_par`] but reuses caller-provided scratch buffers.
#[cfg(feature = "rayon")]
pub fn i8gemm_compute_par_with_scratch(
    m: usize,
    a_float: &[f32],
    packed_b: &PackedBMatrixI8,
    b_scale: f32,
    c_float: &mut [f32],
    scratch: &mut I8GemmScratch,
) {
    let k = packed_b.k();
    let n = packed_b.n();

    let (a_scale, a_zero_point) =
        quantize_a_into(a_float, m, k, &mut scratch.a_u8, &mut scratch.row_offsets);
    scratch.c_i32.resize(m * n, 0);
    scratch.c_i32.fill(0);
    let flags = SimdFlags::detect();

    let mb_starts: Vec<usize> = (0..m).step_by(MCB).collect();
    let c_ptr = scratch.c_i32.as_mut_ptr() as usize;

    let num_k_blocks = packed_b.num_k_blocks();
    let num_n_blocks = packed_b.num_n_blocks();

    for kb in 0..num_k_blocks {
        let k_start = kb * KCB;
        let kc = std::cmp::min(KCB, k - k_start);

        mb_starts.par_iter().for_each(|&mb_start| {
            let mc = std::cmp::min(MCB, m - mb_start);
            let c_ptr = c_ptr as *mut i32;
            // SAFETY: each M-block writes to rows [mb_start..mb_start+mc] — disjoint.
            unsafe {
                process_kb_block(
                    mb_start,
                    mc,
                    kb,
                    k_start,
                    kc,
                    &scratch.a_u8,
                    k,
                    n,
                    num_n_blocks,
                    packed_b,
                    c_ptr,
                    &flags,
                );
            }
        });
    }

    dequantize(
        m,
        n,
        &scratch.c_i32,
        c_float,
        a_scale,
        b_scale,
        flags.effective_zp(a_zero_point),
        packed_b.col_offsets(),
    );
}

/// Process one M-block across all K-blocks and N-blocks (sequential path).
#[cfg(not(feature = "rayon"))]
fn process_m_block(
    mb_start: usize,
    mc: usize,
    a_u8: &[u8],
    k: usize,
    n: usize,
    packed_b: &PackedBMatrixI8,
    c_i32: &mut [i32],
    flags: &SimdFlags,
) {
    let num_k_blocks = packed_b.num_k_blocks();
    let num_n_blocks = packed_b.num_n_blocks();

    for kb in 0..num_k_blocks {
        let k_start = kb * KCB;
        let kc = std::cmp::min(KCB, k - k_start);
        // SAFETY: single-threaded, exclusive access to c_i32.
        unsafe {
            process_kb_block(
                mb_start,
                mc,
                kb,
                k_start,
                kc,
                a_u8,
                k,
                n,
                num_n_blocks,
                packed_b,
                c_i32.as_mut_ptr(),
                flags,
            );
        }
    }
}

/// Process one (M-block, K-block) pair across all N-blocks.
///
/// # Safety
/// Caller must ensure rows `[mb_start..mb_start+mc]` of the C buffer pointed
/// to by `c_ptr` are not concurrently modified.
unsafe fn process_kb_block(
    mb_start: usize,
    mc: usize,
    kb: usize,
    k_start: usize,
    kc: usize,
    a_u8: &[u8],
    k: usize,
    n: usize,
    num_n_blocks: usize,
    packed_b: &PackedBMatrixI8,
    c_ptr: *mut i32,
    flags: &SimdFlags,
) {
    let kc_aligned = (kc + ROW_INTERLEAVE - 1) / ROW_INTERLEAVE * ROW_INTERLEAVE;

    // Thread-local scratchpad for A repacking
    let mut a_packed = if flags.use_simd() {
        vec![0u8; MR * KCB]
    } else {
        Vec::new()
    };

    for nb in 0..num_n_blocks {
        let n_start = nb * NR;
        let nc = std::cmp::min(NR, n - n_start);
        let b_tile = packed_b.tile(kb, nb);

        let mut row_offset = 0;
        while row_offset < mc {
            let kernel_rows = std::cmp::min(MR, mc - row_offset);
            let i = mb_start + row_offset;

            #[cfg(target_arch = "x86_64")]
            if flags.avx2 && nc == NR {
                a_packed.fill(0);
                avx2::pack_a_tile(a_u8, i, k_start, kernel_rows, kc, k, &mut a_packed);
                avx2::dispatch_i8_kernel(
                    kernel_rows,
                    a_packed.as_ptr(),
                    b_tile.as_ptr(),
                    c_ptr.add(i * n + n_start),
                    kc_aligned,
                    n * 4,
                );
                row_offset += kernel_rows;
                continue;
            }

            #[cfg(target_arch = "aarch64")]
            if flags.neon_dotprod && nc == NR {
                a_packed.fill(0);
                neon::pack_a_tile(a_u8, i, k_start, kernel_rows, kc, k, &mut a_packed);
                neon::dispatch_i8_kernel(
                    kernel_rows,
                    a_packed.as_ptr(),
                    b_tile.as_ptr(),
                    c_ptr.add(i * n + n_start),
                    kc_aligned,
                    n * 4,
                );
                row_offset += kernel_rows;
                continue;
            }

            // Fallback: reference kernel
            let c_slice = std::slice::from_raw_parts_mut(
                c_ptr.add(i * n + n_start),
                (kernel_rows - 1) * n + nc,
            );

            #[cfg(target_arch = "aarch64")]
            if flags.neon_dotprod {
                // Signed A to match SDOT's int8×int8 semantics
                ref_i8i8_kernel(
                    kernel_rows,
                    &a_u8[i * k + k_start..],
                    k,
                    b_tile,
                    c_slice,
                    n,
                    kc,
                    nc,
                );
                row_offset += kernel_rows;
                continue;
            }

            ref_u8i8_kernel(
                kernel_rows,
                &a_u8[i * k + k_start..],
                k,
                b_tile,
                c_slice,
                n,
                kc,
                nc,
            );

            row_offset += kernel_rows;
        }
    }
}

/// Dequantize int32 accumulation buffer to float32 output.
fn dequantize(
    m: usize,
    n: usize,
    c_i32: &[i32],
    c_float: &mut [f32],
    a_scale: f32,
    b_scale: f32,
    effective_zp: i32,
    col_offsets: &[i32],
) {
    let output_scale = a_scale * b_scale;
    for i in 0..m {
        for j in 0..n {
            let raw = c_i32[i * n + j];
            let adjusted = raw - effective_zp * col_offsets[j];
            c_float[i * n + j] = adjusted as f32 * output_scale;
        }
    }
}

/// Reference kernel: uint8 A × int8 B → int32 C.
fn ref_u8i8_kernel(
    mc: usize,
    a: &[u8],
    lda: usize,
    b: &[i8],
    c: &mut [i32],
    ldc: usize,
    kc: usize,
    nc: usize,
) {
    for i in 0..mc {
        for k_idx in 0..kc {
            let g = k_idx / ROW_INTERLEAVE;
            let ri = k_idx % ROW_INTERLEAVE;
            let a_val = a[i * lda + k_idx] as i32;
            for j in 0..nc {
                let b_val = b[g * NR * ROW_INTERLEAVE + j * ROW_INTERLEAVE + ri] as i32;
                c[i * ldc + j] += a_val * b_val;
            }
        }
    }
}

/// Reference kernel: int8 A × int8 B → int32 C (Neon SDOT path column fringe).
#[cfg(target_arch = "aarch64")]
fn ref_i8i8_kernel(
    mc: usize,
    a: &[u8],
    lda: usize,
    b: &[i8],
    c: &mut [i32],
    ldc: usize,
    kc: usize,
    nc: usize,
) {
    for i in 0..mc {
        for k_idx in 0..kc {
            let g = k_idx / ROW_INTERLEAVE;
            let ri = k_idx % ROW_INTERLEAVE;
            let a_val = (a[i * lda + k_idx] ^ 0x80) as i8 as i32;
            for j in 0..nc {
                let b_val = b[g * NR * ROW_INTERLEAVE + j * ROW_INTERLEAVE + ri] as i32;
                c[i * ldc + j] += a_val * b_val;
            }
        }
    }
}
