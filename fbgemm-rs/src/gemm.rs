use crate::kernels::{get_kernels, GemmParams, KernelFn};
use crate::pack::{PackedMatrix, BLOCK_COL_SIZE};
use crate::partition::PARTITION_AVX2;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

const MB_MAX: usize = 120;

/// Transpose A from row-major to column-major scratchpad (used on x86_64).
/// to[r + c * nrow] = from[r * ldim + c]
#[cfg(not(target_arch = "aarch64"))]
fn pack_a(nrow: usize, ncol: usize, from: &[f32], ldim: usize, to: &mut [f32]) {
    for r in 0..nrow {
        for c in 0..ncol {
            to[r + c * nrow] = from[r * ldim + c];
        }
    }
}

/// Collect all (m2, kernel_nrows) row-group tasks for the given M range.
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

/// Process one row group: runs the micro-kernel for `kernel_nrows` rows starting at `m2`.
///
/// SAFETY: caller must ensure `c_ptr + m2*n .. c_ptr + (m2+kernel_nrows)*n` is valid
/// and not aliased by any concurrent call with overlapping row ranges.
unsafe fn process_row_group(
    m2: usize,
    kernel_nrows: usize,
    k_ind: usize,
    kb: usize,
    #[cfg_attr(target_arch = "aarch64", allow(unused))] total_m: usize,
    beta_: f32,
    a: &[f32],
    k: usize,
    n: usize,
    packed_b: &PackedMatrix,
    c_ptr: *mut f32,
    kernels: &[Option<KernelFn>],
) {
    let ldc = n;
    let bcol = BLOCK_COL_SIZE;
    let nbcol = n / bcol;

    #[cfg(not(target_arch = "aarch64"))]
    let mut scratchpad = vec![0.0f32; 6 * kb];

    let mut gp = GemmParams {
        k: kb as u64,
        a: std::ptr::null_mut(),
        b: packed_b.at(k_ind, 0),
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

        // Fringe: remaining columns
        let last_blk_col = nbcol * bcol;
        let rem = n - last_blk_col;
        debug_assert!(rem < bcol);

        let mut c_tmp = [0.0f32; 14 * 32];
        gp.b = packed_b.at(k_ind, last_blk_col);
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

/// Compute C = beta * C + A * packed_B (single-threaded).
pub fn cblas_gemm_compute(m: usize, a: &[f32], packed_b: &PackedMatrix, beta: f32, c: &mut [f32]) {
    let k = packed_b.k();
    let n = packed_b.n();
    let brow = packed_b.block_row_size();
    let kernels = get_kernels();
    let tasks = collect_row_groups(m);
    let c_ptr = c.as_mut_ptr();

    for k_ind in (0..k).step_by(brow) {
        let beta_ = if k_ind == 0 { beta } else { 1.0 };
        let kb = brow.min(k - k_ind);

        for &(m2, kernel_nrows) in &tasks {
            unsafe {
                process_row_group(
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
}

/// Compute C = beta * C + A * packed_B (multi-threaded via rayon).
///
/// Row groups within each K-block are dispatched in parallel.
/// Each writes to disjoint rows of C, so no synchronization is needed.
#[cfg(feature = "rayon")]
pub fn cblas_gemm_compute_par(
    m: usize,
    a: &[f32],
    packed_b: &PackedMatrix,
    beta: f32,
    c: &mut [f32],
) {
    let k = packed_b.k();
    let n = packed_b.n();
    let brow = packed_b.block_row_size();
    let kernels = get_kernels();
    let tasks = collect_row_groups(m);
    let c_ptr = c.as_mut_ptr() as usize; // for Send

    for k_ind in (0..k).step_by(brow) {
        let beta_ = if k_ind == 0 { beta } else { 1.0 };
        let kb = brow.min(k - k_ind);

        tasks.par_iter().for_each(|&(m2, kernel_nrows)| {
            let c_ptr = c_ptr as *mut f32;
            // SAFETY: each task writes to rows [m2..m2+kernel_nrows] — disjoint across tasks.
            unsafe {
                process_row_group(
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
}
