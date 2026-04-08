use core::arch::asm;

use super::pack::KCB;

/// Dispatch to the AVX2 uint8×int8→int32 micro-kernel for the given row count.
///
/// # Safety
/// - `kc` must be > 0 and a multiple of 4
/// - `buffer_a` must have KCB=512 byte stride per row, with `mc` rows packed
/// - `buffer_b` must be in row-interleaved NR=8 format
/// - `buffer_c` must point to valid i32 storage with `ldc_bytes` byte stride
pub unsafe fn dispatch_i8_kernel(
    mc: usize,
    buffer_a: *const u8,
    buffer_b: *const i8,
    buffer_c: *mut i32,
    kc: usize,
    ldc_bytes: usize,
) {
    debug_assert!(mc >= 1 && mc <= 12);
    match mc {
        1 => avx2_i8_kernel_1(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        2 => avx2_i8_kernel_2(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        3 => avx2_i8_kernel_3(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        4 => avx2_i8_kernel_4(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        5 => avx2_i8_kernel_5(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        6 => avx2_i8_kernel_6(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        7 => avx2_i8_kernel_7(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        8 => avx2_i8_kernel_8(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        9 => avx2_i8_kernel_9(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        10 => avx2_i8_kernel_10(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        11 => avx2_i8_kernel_11(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        12 => avx2_i8_kernel_12(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        _ => unreachable!("mc must be 1..=12"),
    }
}

/// Generate an AVX2 uint8×int8→int32 micro-kernel for a specific row count.
///
/// Each row is identified by (ymm_register_index, a_byte_offset) where
/// a_byte_offset = row * KCB = row * 512.
///
/// Register allocation:
///   ymm0-ymm11: C accumulators (one per row, up to 12)
///   ymm12: temp result
///   ymm13: 16-bit ones vector (for vpmaddwd)
///   ymm14: B register
///   ymm15: A broadcast register
macro_rules! define_i8_kernel {
    ($name:ident, [$(($row:literal, $offset:literal)),+ $(,)?]) => {
        #[inline(never)]
        unsafe fn $name(
            buffer_a: *const u8,
            buffer_b: *const i8,
            buffer_c: *mut i32,
            kc: usize,
            ldc_bytes: usize,
        ) {
            debug_assert!(kc > 0 && kc % 4 == 0);
            asm!(
                // Init ones: ymm13 = all 16-bit 1s
                "vpcmpeqw ymm13, ymm13, ymm13",
                "vpsrlw ymm13, ymm13, 15",

                // Zero accumulators
                $(concat!(
                    "vpxor ymm", stringify!($row),
                    ", ymm", stringify!($row),
                    ", ymm", stringify!($row),
                ),)+

                // K loop (each iteration processes ROW_INTERLEAVE=4 k-elements)
                "2:",
                "vmovdqu ymm14, ymmword ptr [{b}]",
                $(
                    concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!($offset), "]"),
                    "vpmaddubsw ymm12, ymm15, ymm14",
                    "vpmaddwd ymm12, ymm13, ymm12",
                    concat!(
                        "vpaddd ymm", stringify!($row),
                        ", ymm", stringify!($row),
                        ", ymm12",
                    ),
                )+

                // Advance pointers
                "add {a}, 4",
                "add {b}, 32",
                "sub {kc}, 4",
                "jnz 2b",

                // Store: C[row] += accumulator[row]
                "mov rax, {c}",
                $(
                    concat!(
                        "vpaddd ymm", stringify!($row),
                        ", ymm", stringify!($row),
                        ", ymmword ptr [rax]",
                    ),
                    concat!("vmovdqu ymmword ptr [rax], ymm", stringify!($row)),
                    "add rax, {ldc}",
                )+

                "vzeroupper",

                a = inout(reg) buffer_a => _,
                b = inout(reg) buffer_b => _,
                c = in(reg) buffer_c,
                kc = inout(reg) kc => _,
                ldc = in(reg) ldc_bytes,
                out("rax") _,
                out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
                out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
                out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
                out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
                options(nostack),
            );
        }
    };
}

// Offsets: row_i * KCB = row_i * 512
define_i8_kernel!(avx2_i8_kernel_1, [(0, 0)]);
define_i8_kernel!(avx2_i8_kernel_2, [(0, 0), (1, 512)]);
define_i8_kernel!(avx2_i8_kernel_3, [(0, 0), (1, 512), (2, 1024)]);
define_i8_kernel!(avx2_i8_kernel_4, [(0, 0), (1, 512), (2, 1024), (3, 1536)]);
define_i8_kernel!(
    avx2_i8_kernel_5,
    [(0, 0), (1, 512), (2, 1024), (3, 1536), (4, 2048)]
);
define_i8_kernel!(
    avx2_i8_kernel_6,
    [(0, 0), (1, 512), (2, 1024), (3, 1536), (4, 2048), (5, 2560)]
);
define_i8_kernel!(
    avx2_i8_kernel_7,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072)
    ]
);
define_i8_kernel!(
    avx2_i8_kernel_8,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584)
    ]
);
define_i8_kernel!(
    avx2_i8_kernel_9,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096)
    ]
);
define_i8_kernel!(
    avx2_i8_kernel_10,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096),
        (9, 4608)
    ]
);
define_i8_kernel!(
    avx2_i8_kernel_11,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096),
        (9, 4608),
        (10, 5120)
    ]
);
define_i8_kernel!(
    avx2_i8_kernel_12,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096),
        (9, 4608),
        (10, 5120),
        (11, 5632)
    ]
);

/// Pack A tile: copy mc rows × kc columns from quantized A (K stride) into
/// a KCB-strided scratchpad for the SIMD kernel.
pub fn pack_a_tile(
    a_u8: &[u8],
    m_start: usize,
    k_start: usize,
    mc: usize,
    kc: usize,
    k: usize,
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= mc * KCB);
    for i in 0..mc {
        let src_offset = (m_start + i) * k + k_start;
        let dst_offset = i * KCB;
        dst[dst_offset..dst_offset + kc].copy_from_slice(&a_u8[src_offset..src_offset + kc]);
    }
}
