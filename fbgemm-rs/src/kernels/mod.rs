pub mod fallback;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx2_bf16;
#[cfg(target_arch = "aarch64")]
pub mod neon_bf16;

use crate::pack::BLOCK_COL_SIZE;

/// Parameters passed to each GEMM micro-kernel invocation.
/// Layout must match FBGEMM's GemmParams<float> exactly (accessed by fixed offsets in asm).
#[repr(C)]
pub struct GemmParams {
    pub k: u64,            // offset 0
    pub a: *mut f32,       // offset 8
    pub b: *const f32,     // offset 16
    pub beta: f32,         // offset 24
    pub _pad: u32,         // offset 28 (alignment padding)
    pub c: *mut f32,       // offset 32
    pub ldc: u64,          // offset 40  (in bytes)
    pub b_block_cols: u64, // offset 48
    pub lda: u64,          // offset 56  (in bytes, used on aarch64)
}

/// A micro-kernel function pointer: takes GemmParams and processes `kernel_nrows` rows.
pub type KernelFn = unsafe fn(*mut GemmParams);

/// Returns the kernel table and partition table for the current platform.
/// kernel_table[n] handles n rows (index 0 is unused).
pub fn get_kernels() -> &'static [Option<KernelFn>] {
    #[cfg(target_arch = "aarch64")]
    {
        if true {
            // Neon is always available on aarch64
            return &neon::KERNELS;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return &avx2::KERNELS;
        }
    }

    &fallback::KERNELS
}

/// Returns the bf16 kernel table for the current platform.
/// These kernels load B from bf16 (u16) packed format, converting inline to f32.
pub fn get_bf16_kernels() -> &'static [Option<KernelFn>] {
    #[cfg(target_arch = "aarch64")]
    {
        return &neon_bf16::KERNELS;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return &avx2_bf16::KERNELS;
        }
    }

    // Fallback: use standard kernels (caller must pre-convert bf16→f32)
    #[allow(unreachable_code)]
    &fallback::KERNELS
}

/// Reference kernel: pure Rust, no SIMD. Works on any platform.
/// A is accessed in row-major with stride lda (in bytes).
pub unsafe fn ref_kernel(kernel_nrows: usize, gp: *mut GemmParams) {
    let gp = &*gp;
    let k = gp.k as usize;
    let ldc = gp.ldc as usize / 4;
    let lda = gp.lda as usize / 4;
    let bcol = BLOCK_COL_SIZE;

    for jb in 0..gp.b_block_cols as usize {
        for kk in 0..k {
            for i in 0..kernel_nrows {
                let a_val = *gp.a.add(i * lda + kk);
                for j in 0..bcol {
                    let c_ptr = gp.c.add(i * ldc + jb * bcol + j);
                    let b_val = *gp.b.add((jb * k + kk) * bcol + j);
                    if kk == 0 {
                        if gp.beta != 0.0 {
                            *c_ptr = a_val.mul_add(b_val, gp.beta * *c_ptr);
                        } else {
                            *c_ptr = a_val * b_val;
                        }
                    } else {
                        *c_ptr = a_val.mul_add(b_val, *c_ptr);
                    }
                }
            }
        }
    }
}
