//! BFloat16 packed matrix GEMM.
//!
//! Stores B matrix weights as bf16 (u16) in the same blocked layout as
//! `PackedMatrix`. On x86_64 with AVX2+FMA, uses inline bf16→f32 conversion
//! kernels (vpmovzxwd + vpslld) directly in the micro-kernel inner loop —
//! no scratch buffer needed. Falls back to bulk bf16→f32 conversion per
//! K-block strip on other platforms.
//!
//! Halves weight memory and B-matrix cache pressure vs f32 packed GEMM.

use crate::kernels::{GemmParams, KernelFn};
use crate::pack::BLOCK_COL_SIZE;
use crate::partition::PARTITION_AVX2;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

const DEFAULT_BROW: usize = 512;
const MB_MAX: usize = 120;

/// Convert f32 to bf16 by truncating (round-to-nearest-even could be added later).
#[inline(always)]
fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Convert bf16 to f32 by zero-extending and shifting.
#[inline(always)]
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// A pre-packed B matrix stored in bfloat16 format.
///
/// Same blocked layout as `PackedMatrix` but elements are `u16` (bf16).
pub struct PackedMatrixBf16 {
    nrow: usize, // K dimension
    ncol: usize, // N dimension
    brow: usize,
    last_brow: usize,
    bcol: usize, // = BLOCK_COL_SIZE
    nbrow: usize,
    nbcol: usize,
    data: Vec<u16>,
}

unsafe impl Send for PackedMatrixBf16 {}
unsafe impl Sync for PackedMatrixBf16 {}

impl PackedMatrixBf16 {
    /// Pack a K×N row-major f32 matrix, converting to bf16.
    pub fn new(k: usize, n: usize, src: &[f32]) -> Self {
        assert_eq!(src.len(), k * n, "src length must be k * n");
        let mut m = Self::alloc(k, n);
        m.pack_from_src(false, src);
        m
    }

    /// Pack from column-major (transposed) f32 storage, converting to bf16.
    pub fn from_transposed(k: usize, n: usize, src: &[f32]) -> Self {
        assert_eq!(src.len(), k * n, "src length must be k * n");
        let mut m = Self::alloc(k, n);
        m.pack_from_src(true, src);
        m
    }

    pub fn k(&self) -> usize {
        self.nrow
    }

    pub fn n(&self) -> usize {
        self.ncol
    }

    pub fn block_row_size(&self) -> usize {
        self.brow
    }

    /// Memory used by packed bf16 data in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 2
    }

    fn alloc(nrow: usize, ncol: usize) -> Self {
        let brow = DEFAULT_BROW;
        let bcol = BLOCK_COL_SIZE;
        let nbrow = (nrow + brow - 1) / brow;
        let last_brow = if nrow % brow == 0 { brow } else { nrow % brow };
        let nbcol = (ncol + bcol - 1) / bcol;
        let size = brow * nbrow * bcol * nbcol;
        Self {
            nrow,
            ncol,
            brow,
            last_brow,
            bcol,
            nbrow,
            nbcol,
            data: vec![0u16; size],
        }
    }

    fn addr(&self, r: usize, c: usize) -> usize {
        let block_row_id = r / self.brow;
        let brow_offset = block_row_id * self.nbcol * self.brow * self.bcol;
        let block_col_id = c / self.bcol;
        let rows_in_block = if block_row_id as isize != self.nbrow as isize - 1 {
            self.brow
        } else {
            self.last_brow
        };
        let bcol_offset = block_col_id * rows_in_block * self.bcol;
        let block_offset = brow_offset + bcol_offset;
        let inblock_offset = (r % self.brow) * self.bcol + (c % self.bcol);
        block_offset + inblock_offset
    }

    fn pack_from_src(&mut self, transposed: bool, src: &[f32]) {
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                let src_val = if transposed {
                    src[i + self.nrow * j]
                } else {
                    src[i * self.ncol + j]
                };
                let idx = self.addr(i, j);
                self.data[idx] = f32_to_bf16(src_val);
            }
        }
    }

    /// Pointer to bf16 element at packed position (r, c), cast to *const f32 for GemmParams.
    /// The bf16 kernels read u16 values from this pointer via vpmovzxwd.
    unsafe fn at_as_f32_ptr(&self, r: usize, c: usize) -> *const f32 {
        self.data.as_ptr().add(self.addr(r, c)) as *const f32
    }
}

/// Verify that all pointer accesses in the GEMM computation will be in-bounds.
///
/// This provides a machine-checkable proof that the unsafe code below cannot
/// cause out-of-bounds memory accesses for the given dimensions. It checks
/// every invariant the unsafe code relies on:
///
/// 1. **A accesses**: For each row group (m2, kernel_nrows) and K-block (k_ind, kb),
///    the kernel reads A[m2*k + k_ind + i*k + kk] for i in 0..kernel_nrows, kk in 0..kb.
///    Maximum index: (m2 + kernel_nrows - 1) * k + k_ind + kb - 1 < m * k = a.len().
///
/// 2. **C accesses**: Kernel writes C[(m2+i)*n + j] for i in 0..kernel_nrows, j in 0..n.
///    Maximum index: (m2 + kernel_nrows - 1) * n + n - 1 < m * n = c.len().
///
/// 3. **B accesses**: Kernel reads nbcol * kb * bcol u16 elements starting at
///    packed_b.addr(k_ind, 0). The blocked layout guarantees contiguous storage
///    within a K-block row because k_ind is always brow-aligned and kb equals
///    the block's rows_in_block.
///
/// 4. **Scratchpad**: pack_a writes at most kernel_nrows * kb <= 6 * kb elements.
///
/// 5. **c_tmp**: Fringe kernel writes at most kernel_nrows * bcol <= 6 * 16 = 96 < 448.
///
/// 6. **Kernel table**: kernel_nrows is always in 1..=6 (from PARTITION_AVX2).
fn verify_gemm_bounds(m: usize, a_len: usize, k: usize, n: usize, packed_b: &PackedMatrixBf16) {
    let brow = packed_b.brow;
    let bcol = packed_b.bcol;
    let tasks = collect_row_groups(m);

    // Verify row groups partition [0, m) exactly and use valid kernel sizes
    let mut covered = 0;
    for &(m2, kernel_nrows) in &tasks {
        assert_eq!(m2, covered, "row groups must be contiguous");
        assert!(
            kernel_nrows >= 1 && kernel_nrows <= 6,
            "kernel_nrows {kernel_nrows} out of range 1..=6"
        );
        covered += kernel_nrows;
    }
    // Allow covered < m only if m == 0 (no tasks)
    if m > 0 {
        assert_eq!(
            covered, m,
            "row groups must cover all {m} rows, covered {covered}"
        );
    }

    for k_ind in (0..k).step_by(brow) {
        let kb = brow.min(k - k_ind);

        // Verify k_ind is brow-aligned (guaranteed by step_by, but check anyway)
        debug_assert_eq!(k_ind % brow, 0);

        // Verify kb matches the block's rows_in_block in the packed layout
        let block_row_id = k_ind / brow;
        let rows_in_block = if block_row_id != packed_b.nbrow - 1 {
            packed_b.brow
        } else {
            packed_b.last_brow
        };
        assert_eq!(
            kb, rows_in_block,
            "kb={kb} must equal rows_in_block={rows_in_block} for block_row_id={block_row_id}"
        );

        // Verify B read bounds: kernel reads nbcol * kb * bcol u16 elements
        // starting at addr(k_ind, 0), contiguously
        let b_start = packed_b.addr(k_ind, 0);
        let nbcol_full = n / bcol;
        let b_read_len = nbcol_full * kb * bcol;
        assert!(
            b_start + b_read_len <= packed_b.data.len(),
            "B main read OOB: start={b_start} + len={b_read_len} > data.len()={}",
            packed_b.data.len()
        );

        // Verify B fringe read bounds (if applicable)
        if n % bcol != 0 {
            let fringe_start = packed_b.addr(k_ind, nbcol_full * bcol);
            let fringe_read_len = kb * bcol; // kernel reads a full block column
            assert!(
                fringe_start + fringe_read_len <= packed_b.data.len(),
                "B fringe read OOB: start={fringe_start} + len={fringe_read_len} > data.len()={}",
                packed_b.data.len()
            );
        }

        for &(m2, kernel_nrows) in &tasks {
            // Verify A bounds
            let a_max = (m2 + kernel_nrows - 1) * k + k_ind + kb - 1;
            assert!(
                a_max < a_len,
                "A OOB: a[{a_max}] >= a.len()={a_len} (m2={m2}, nrows={kernel_nrows}, k_ind={k_ind}, kb={kb})"
            );

            // Verify C bounds
            let c_max = (m2 + kernel_nrows - 1) * n + n - 1;
            assert!(
                c_max < m * n,
                "C OOB: c[{c_max}] >= c.len()={} (m2={m2}, nrows={kernel_nrows})",
                m * n
            );
        }
    }
}

/// Transpose A from row-major to column-major scratchpad (used on x86_64).
#[cfg(not(target_arch = "aarch64"))]
fn pack_a(nrow: usize, ncol: usize, from: &[f32], ldim: usize, to: &mut [f32]) {
    for r in 0..nrow {
        for c in 0..ncol {
            to[r + c * nrow] = from[r * ldim + c];
        }
    }
}

fn collect_row_groups(m: usize) -> Vec<(usize, usize)> {
    let mut tasks = Vec::new();
    for m0 in (0..m).step_by(MB_MAX) {
        let mb = MB_MAX.min(m - m0);
        let partition = &PARTITION_AVX2[mb];
        let mut m1 = m0;
        for cycle in partition {
            let kernel_nrows = cycle[0] as usize;
            let nkernel_nrows = cycle[1] as usize;
            if kernel_nrows == 0 {
                break;
            }
            for _ in 0..nkernel_nrows {
                tasks.push((m1, kernel_nrows));
                m1 += kernel_nrows;
            }
        }
    }
    tasks
}

/// Process one row group using inline bf16 kernels.
///
/// The bf16 kernel reads u16 values from the B pointer via vpmovzxwd,
/// converting to f32 inline. No scratch buffer needed.
///
/// SAFETY: caller must ensure `c_ptr + m2*n .. c_ptr + (m2+kernel_nrows)*n` is valid.
unsafe fn process_row_group_inline(
    m2: usize,
    kernel_nrows: usize,
    k_ind: usize,
    kb: usize,
    #[cfg_attr(target_arch = "aarch64", allow(unused))] total_m: usize,
    beta_: f32,
    a: &[f32],
    k: usize,
    n: usize,
    packed_b: &PackedMatrixBf16,
    c_ptr: *mut f32,
    kernels: &[Option<KernelFn>],
) {
    debug_assert!(kernel_nrows >= 1 && kernel_nrows <= 6);
    debug_assert!(k_ind % packed_b.brow == 0, "k_ind must be brow-aligned");

    let ldc = n;
    let bcol = BLOCK_COL_SIZE;
    let nbcol = n / bcol; // full block columns only (fringe handled separately)

    #[cfg(not(target_arch = "aarch64"))]
    let mut scratchpad = vec![0.0f32; 6 * kb];

    // B pointer: cast bf16 data to *const f32 — the bf16 kernel reads u16 via vpmovzxwd
    let b_ptr = packed_b.at_as_f32_ptr(k_ind, 0);

    let mut gp = GemmParams {
        k: kb as u64,
        a: std::ptr::null_mut(),
        b: b_ptr,
        beta: beta_,
        _pad: 0,
        c: c_ptr.add(m2 * ldc),
        ldc: (ldc * 4) as u64,
        b_block_cols: nbcol as u64,
        lda: 0,
    };

    #[cfg(target_arch = "aarch64")]
    {
        debug_assert!(m2 * k + k_ind < a.len());
        gp.a = (a.as_ptr() as *mut f32).add(m2 * k + k_ind);
        gp.lda = (k * 4) as u64;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if total_m == 1 {
            debug_assert!(k_ind + kb <= a.len());
            gp.a = (a.as_ptr() as *mut f32).add(k_ind);
        } else {
            debug_assert!(m2 * k + k_ind + (kernel_nrows - 1) * k + kb <= a.len());
            pack_a(kernel_nrows, kb, &a[m2 * k + k_ind..], k, &mut scratchpad);
            gp.a = scratchpad.as_mut_ptr();
        }
    }

    let kernel = kernels[kernel_nrows].expect("no kernel for this nrows");

    if n % bcol == 0 {
        if nbcol > 0 {
            kernel(&mut gp);
        }
    } else {
        if nbcol > 0 {
            kernel(&mut gp);
        }

        // Fringe: remaining columns
        let last_blk_col = nbcol * bcol;
        let rem = n - last_blk_col;
        debug_assert!(rem < bcol);

        let mut c_tmp = [0.0f32; 14 * 32];
        gp.b = packed_b.at_as_f32_ptr(k_ind, last_blk_col);
        gp.c = c_tmp.as_mut_ptr();
        gp.ldc = (bcol * 4) as u64;
        gp.b_block_cols = 1;
        kernel(&mut gp);

        for i in 0..kernel_nrows {
            for j in 0..rem {
                let src = c_tmp[i * bcol + j];
                let dst = &mut *c_ptr.add((m2 + i) * ldc + last_blk_col + j);
                if beta_ == 0.0 {
                    *dst = src;
                } else {
                    *dst = beta_ * *dst + src;
                }
            }
        }
    }
}

/// Returns true if we have native bf16 kernels (inline bf16→f32 in micro-kernel).
fn have_bf16_kernels() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return true; // Neon bf16 kernels always available on aarch64
    }
    #[cfg(target_arch = "x86_64")]
    {
        return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    }
    #[allow(unreachable_code)]
    false
}

/// Compute C = beta * C + A * packed_B_bf16 (single-threaded).
#[cfg(not(feature = "rayon"))]
fn compute_st(m: usize, a: &[f32], packed_b: &PackedMatrixBf16, beta: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    let brow = packed_b.block_row_size();
    let tasks = collect_row_groups(m);
    let c_ptr = c.as_mut_ptr();

    if have_bf16_kernels() {
        // Inline bf16 kernels: read bf16 data directly, no scratch buffer
        let kernels = crate::kernels::get_bf16_kernels();
        for k_ind in (0..k).step_by(brow) {
            let beta_ = if k_ind == 0 { beta } else { 1.0 };
            let kb = brow.min(k - k_ind);
            for &(m2, kernel_nrows) in &tasks {
                unsafe {
                    process_row_group_inline(
                        m2,
                        kernel_nrows,
                        k_ind,
                        kb,
                        m,
                        beta_,
                        a,
                        k,
                        n,
                        packed_b,
                        c_ptr,
                        kernels,
                    );
                }
            }
        }
    } else {
        // Fallback: convert bf16→f32 per K-block strip, use standard f32 kernels
        let kernels = crate::kernels::get_kernels();
        let bcol = BLOCK_COL_SIZE;
        let nbcol = packed_b.nbcol;
        for k_ind in (0..k).step_by(brow) {
            let beta_ = if k_ind == 0 { beta } else { 1.0 };
            let kb = brow.min(k - k_ind);
            let mut f32_strip = vec![0.0f32; kb * bcol * (nbcol + 1)];
            let total_elems = packed_b.data.len();
            let strip_start = packed_b.addr(k_ind, 0);
            let strip_len = (kb * bcol * nbcol).min(total_elems - strip_start);
            for i in 0..strip_len {
                f32_strip[i] = bf16_to_f32(packed_b.data[strip_start + i]);
            }
            if n % bcol != 0 {
                let fringe_start = packed_b.addr(k_ind, nbcol * bcol);
                let fringe_len = (kb * bcol).min(total_elems.saturating_sub(fringe_start));
                for i in 0..fringe_len {
                    f32_strip[nbcol * kb * bcol + i] = bf16_to_f32(packed_b.data[fringe_start + i]);
                }
            }
            let f32_b_ptr = f32_strip.as_ptr();
            for &(m2, kernel_nrows) in &tasks {
                unsafe {
                    process_row_group_fallback(
                        m2,
                        kernel_nrows,
                        k_ind,
                        kb,
                        m,
                        beta_,
                        a,
                        k,
                        n,
                        f32_b_ptr,
                        c_ptr,
                        kernels,
                    );
                }
            }
        }
    }
}

/// Compute C = beta * C + A * packed_B_bf16 (multi-threaded via rayon).
#[cfg(feature = "rayon")]
fn compute_par(m: usize, a: &[f32], packed_b: &PackedMatrixBf16, beta: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    let brow = packed_b.block_row_size();
    let tasks = collect_row_groups(m);
    let c_ptr = c.as_mut_ptr() as usize;

    if have_bf16_kernels() {
        let kernels = crate::kernels::get_bf16_kernels();
        let packed_ptr = packed_b as *const PackedMatrixBf16 as usize;
        for k_ind in (0..k).step_by(brow) {
            let beta_ = if k_ind == 0 { beta } else { 1.0 };
            let kb = brow.min(k - k_ind);
            tasks.par_iter().for_each(|&(m2, kernel_nrows)| {
                let c_ptr = c_ptr as *mut f32;
                let packed_b = unsafe { &*(packed_ptr as *const PackedMatrixBf16) };
                unsafe {
                    process_row_group_inline(
                        m2,
                        kernel_nrows,
                        k_ind,
                        kb,
                        m,
                        beta_,
                        a,
                        k,
                        n,
                        packed_b,
                        c_ptr,
                        kernels,
                    );
                }
            });
        }
    } else {
        let kernels = crate::kernels::get_kernels();
        let bcol = BLOCK_COL_SIZE;
        let nbcol = packed_b.nbcol;
        for k_ind in (0..k).step_by(brow) {
            let beta_ = if k_ind == 0 { beta } else { 1.0 };
            let kb = brow.min(k - k_ind);
            let mut f32_strip = vec![0.0f32; kb * bcol * (nbcol + 1)];
            let total_elems = packed_b.data.len();
            let strip_start = packed_b.addr(k_ind, 0);
            let strip_len = (kb * bcol * nbcol).min(total_elems - strip_start);
            for i in 0..strip_len {
                f32_strip[i] = bf16_to_f32(packed_b.data[strip_start + i]);
            }
            if n % bcol != 0 {
                let fringe_start = packed_b.addr(k_ind, nbcol * bcol);
                let fringe_len = (kb * bcol).min(total_elems.saturating_sub(fringe_start));
                for i in 0..fringe_len {
                    f32_strip[nbcol * kb * bcol + i] = bf16_to_f32(packed_b.data[fringe_start + i]);
                }
            }
            let f32_b_ptr = f32_strip.as_ptr() as usize;
            tasks.par_iter().for_each(|&(m2, kernel_nrows)| {
                let c_ptr = c_ptr as *mut f32;
                let f32_b_ptr = f32_b_ptr as *const f32;
                unsafe {
                    process_row_group_fallback(
                        m2,
                        kernel_nrows,
                        k_ind,
                        kb,
                        m,
                        beta_,
                        a,
                        k,
                        n,
                        f32_b_ptr,
                        c_ptr,
                        kernels,
                    );
                }
            });
        }
    }
}

/// Fallback process_row_group using pre-converted f32 scratch buffer.
unsafe fn process_row_group_fallback(
    m2: usize,
    kernel_nrows: usize,
    k_ind: usize,
    kb: usize,
    #[cfg_attr(target_arch = "aarch64", allow(unused))] total_m: usize,
    beta_: f32,
    a: &[f32],
    k: usize,
    n: usize,
    f32_b_ptr: *const f32,
    c_ptr: *mut f32,
    kernels: &[Option<KernelFn>],
) {
    debug_assert!(kernel_nrows >= 1 && kernel_nrows <= 6);

    let ldc = n;
    let bcol = BLOCK_COL_SIZE;
    let nbcol = n / bcol; // full block columns only

    #[cfg(not(target_arch = "aarch64"))]
    let mut scratchpad = vec![0.0f32; 6 * kb];

    let mut gp = GemmParams {
        k: kb as u64,
        a: std::ptr::null_mut(),
        b: f32_b_ptr,
        beta: beta_,
        _pad: 0,
        c: c_ptr.add(m2 * ldc),
        ldc: (ldc * 4) as u64,
        b_block_cols: nbcol as u64,
        lda: 0,
    };

    #[cfg(target_arch = "aarch64")]
    {
        gp.a = (a.as_ptr() as *mut f32).add(m2 * k + k_ind);
        gp.lda = (k * 4) as u64;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if total_m == 1 {
            gp.a = (a.as_ptr() as *mut f32).add(k_ind);
        } else {
            pack_a(kernel_nrows, kb, &a[m2 * k + k_ind..], k, &mut scratchpad);
            gp.a = scratchpad.as_mut_ptr();
        }
    }

    let kernel = kernels[kernel_nrows].expect("no kernel for this nrows");

    if n % bcol == 0 {
        if nbcol > 0 {
            kernel(&mut gp);
        }
    } else {
        if nbcol > 0 {
            kernel(&mut gp);
        }
        let last_blk_col = nbcol * bcol;
        let rem = n - last_blk_col;
        let mut c_tmp = [0.0f32; 14 * 32];
        gp.b = f32_b_ptr.add(nbcol * kb * bcol);
        gp.c = c_tmp.as_mut_ptr();
        gp.ldc = (bcol * 4) as u64;
        gp.b_block_cols = 1;
        kernel(&mut gp);
        for i in 0..kernel_nrows {
            for j in 0..rem {
                let src = c_tmp[i * bcol + j];
                let dst = &mut *c_ptr.add((m2 + i) * ldc + last_blk_col + j);
                if beta_ == 0.0 {
                    *dst = src;
                } else {
                    *dst = beta_ * *dst + src;
                }
            }
        }
    }
}

/// Compute C = beta * C + A * B_bf16.
pub fn sgemm_bf16(m: usize, a: &[f32], packed_b: &PackedMatrixBf16, beta: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    assert_eq!(a.len(), m * k, "a must have length m * k");
    assert_eq!(c.len(), m * n, "c must have length m * n");

    if m > 0 && k > 0 && n > 0 {
        verify_gemm_bounds(m, a.len(), k, n, packed_b);
    }

    #[cfg(feature = "rayon")]
    {
        compute_par(m, a, packed_b, beta, c);
    }
    #[cfg(not(feature = "rayon"))]
    {
        compute_st(m, a, packed_b, beta, c);
    }
}

/// Compute C = A * B_bf16 (overwriting C).
pub fn sgemm_bf16_simple(m: usize, a: &[f32], packed_b: &PackedMatrixBf16, c: &mut [f32]) {
    sgemm_bf16(m, a, packed_b, 0.0, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prove that addr(r, c) is always in-bounds of data for all r < nrow, c < ncol.
    /// This exhaustively checks for a set of dimension combinations that the packed
    /// layout never produces an out-of-bounds index.
    #[test]
    fn test_addr_always_in_bounds() {
        let test_dims: Vec<(usize, usize)> = vec![
            (1, 1),
            (1, 16),
            (1, 17),
            (16, 1),
            (16, 16),
            (16, 17),
            (15, 15),
            (512, 16),
            (512, 17),
            (513, 16),
            (513, 32),
            (1024, 33),
            (100, 100),
            (511, 31),
        ];
        for &(nrow, ncol) in &test_dims {
            let m = PackedMatrixBf16::alloc(nrow, ncol);
            for r in 0..nrow {
                for c in 0..ncol {
                    let idx = m.addr(r, c);
                    assert!(
                        idx < m.data.len(),
                        "addr({r},{c}) = {idx} >= data.len()={} for nrow={nrow}, ncol={ncol}",
                        m.data.len()
                    );
                }
            }
        }
    }

    /// Prove that within each K-block, consecutive column blocks are contiguous
    /// in memory, matching the kernel's linear B pointer advance pattern.
    #[test]
    fn test_b_layout_contiguous_within_kblock() {
        let test_dims: Vec<(usize, usize)> = vec![
            (1, 16),
            (1, 32),
            (1, 33),
            (512, 16),
            (512, 48),
            (513, 16),
            (513, 33),
            (1024, 48),
            (100, 47),
        ];
        let bcol = BLOCK_COL_SIZE;
        let brow = DEFAULT_BROW;

        for &(nrow, ncol) in &test_dims {
            let m = PackedMatrixBf16::alloc(nrow, ncol);
            let nbcol_full = ncol / bcol;

            for k_ind in (0..nrow).step_by(brow) {
                let kb = brow.min(nrow - k_ind);
                let base = m.addr(k_ind, 0);

                // Verify each full column block starts at base + j * kb * bcol
                for j in 0..nbcol_full {
                    let expected_start = base + j * kb * bcol;
                    let actual_start = m.addr(k_ind, j * bcol);
                    assert_eq!(
                        actual_start, expected_start,
                        "column block {j} misaligned: expected {expected_start}, got {actual_start} \
                         (nrow={nrow}, ncol={ncol}, k_ind={k_ind}, kb={kb})"
                    );

                    // Verify elements within the block are contiguous
                    for row in 0..kb {
                        for col in 0..bcol {
                            let expected = expected_start + row * bcol + col;
                            let actual = m.addr(k_ind + row, j * bcol + col);
                            assert_eq!(actual, expected);
                        }
                    }
                }

                // Verify fringe block (if any) is also contiguous after the full blocks
                if ncol % bcol != 0 {
                    let fringe_col = nbcol_full * bcol;
                    let expected_fringe_start = base + nbcol_full * kb * bcol;
                    let actual_fringe_start = m.addr(k_ind, fringe_col);
                    assert_eq!(
                        actual_fringe_start, expected_fringe_start,
                        "fringe block misaligned (nrow={nrow}, ncol={ncol}, k_ind={k_ind})"
                    );
                }
            }
        }
    }

    /// Verify that verify_gemm_bounds passes for a wide sweep of dimensions,
    /// proving no combination can cause out-of-bounds access.
    #[test]
    fn test_verify_bounds_sweep() {
        let ms = [1, 2, 3, 5, 6, 7, 12, 60, 119, 120, 121, 240];
        let ks = [1, 2, 15, 16, 32, 100, 511, 512, 513, 1024];
        let ns = [1, 2, 15, 16, 17, 31, 32, 33, 48, 49, 64];

        for &m in &ms {
            for &k in &ks {
                for &n in &ns {
                    let src = vec![0.0f32; k * n];
                    let packed_b = PackedMatrixBf16::new(k, n, &src);
                    // This will panic if any bounds check fails
                    verify_gemm_bounds(m, m * k, k, n, &packed_b);
                }
            }
        }
    }

    #[test]
    fn test_bf16_conversion_roundtrip() {
        let vals = [1.0f32, -1.0, 0.0, 3.14, 100.0, -0.001, 65504.0];
        for &v in &vals {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            let rel_err = if v == 0.0 {
                back.abs()
            } else {
                ((back - v) / v).abs()
            };
            assert!(
                rel_err < 0.01,
                "bf16 roundtrip for {}: got {}, err={}",
                v,
                back,
                rel_err
            );
        }
    }

    #[test]
    fn test_bf16_basic_sgemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut c = vec![0.0f32; 4];

        let packed_b = PackedMatrixBf16::new(3, 2, &b);
        sgemm_bf16_simple(2, &a, &packed_b, &mut c);

        // Expected: [[58, 64], [139, 154]]
        assert!((c[0] - 58.0).abs() < 1.0, "c[0] = {}", c[0]);
        assert!((c[1] - 64.0).abs() < 1.0, "c[1] = {}", c[1]);
        assert!((c[2] - 139.0).abs() < 2.0, "c[2] = {}", c[2]);
        assert!((c[3] - 154.0).abs() < 2.0, "c[3] = {}", c[3]);
    }

    #[test]
    fn test_bf16_larger_matrix() {
        let m = 16;
        let k = 32;
        let n = 16;

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];

        let packed_b = PackedMatrixBf16::new(k, n, &b);
        sgemm_bf16_simple(m, &a, &packed_b, &mut c);

        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c_ref[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        for i in 0..m * n {
            let rel_err = if c_ref[i].abs() > 0.01 {
                ((c[i] - c_ref[i]) / c_ref[i]).abs()
            } else {
                (c[i] - c_ref[i]).abs()
            };
            assert!(
                rel_err < 0.02,
                "mismatch at {}: got {}, expected {}, rel_err={}",
                i,
                c[i],
                c_ref[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_bf16_memory_savings() {
        let k = 768;
        let n = 3072;
        let src: Vec<f32> = vec![0.0; k * n];

        let packed_f32 = crate::PackedMatrix::new(k, n, &src);
        let packed_bf16 = PackedMatrixBf16::new(k, n, &src);

        let f32_bytes = std::mem::size_of_val(packed_f32.data.as_slice());
        let bf16_bytes = packed_bf16.size_bytes();
        let ratio = f32_bytes as f64 / bf16_bytes as f64;
        assert!(
            ratio > 1.8 && ratio < 2.2,
            "expected ~2x ratio, got {:.2}",
            ratio
        );
    }
}
