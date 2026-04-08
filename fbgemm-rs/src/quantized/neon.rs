use core::arch::asm;

use super::pack::KCB;

/// Dispatch to the Neon SDOT int8×int8→int32 micro-kernel for the given row count.
///
/// # Safety
/// - `kc` must be > 0 and a multiple of 4
/// - `buffer_a` must have KCB=512 byte stride per row, containing int8 values (stored as u8)
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
        1 => neon_i8_kernel_1(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        2 => neon_i8_kernel_2(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        3 => neon_i8_kernel_3(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        4 => neon_i8_kernel_4(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        5 => neon_i8_kernel_5(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        6 => neon_i8_kernel_6(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        7 => neon_i8_kernel_7(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        8 => neon_i8_kernel_8(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        9 => neon_i8_kernel_9(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        10 => neon_i8_kernel_10(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        11 => neon_i8_kernel_11(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        12 => neon_i8_kernel_12(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        _ => unreachable!("mc must be 1..=12"),
    }
}

/// Generate a Neon SDOT int8×int8→int32 micro-kernel for a specific row count.
///
/// Uses SDOT with lane-0 broadcast: for each row, load 4 bytes of A into s26,
/// then `sdot vAcc.4s, vB.16b, v26.4b[0]` for both column halves.
///
/// Each row is identified by (acc_lo, acc_hi, a_byte_offset) where:
/// - acc_lo/acc_hi are v-register indices for columns 0-3 and 4-7
/// - a_byte_offset = row * KCB = row * 512
///
/// Register allocation:
///   v0-v23:  C accumulators (2 per row, up to 12 rows)
///   v24-v25: B registers (columns 0-3 and 4-7)
///   v26:     A broadcast register
macro_rules! define_neon_i8_kernel {
    ($name:ident, [$(($lo:literal, $hi:literal, $off:literal)),+ $(,)?]) => {
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
                // Zero accumulators
                $(
                    concat!("movi v", stringify!($lo), ".4s, #0"),
                    concat!("movi v", stringify!($hi), ".4s, #0"),
                )+

                // K loop (each iteration processes ROW_INTERLEAVE=4 k-elements)
                "2:",
                "ldr q24, [{b}]",
                "ldr q25, [{b}, #16]",
                $(
                    concat!("ldr s26, [{a}, #", stringify!($off), "]"),
                    concat!("sdot v", stringify!($lo), ".4s, v24.16b, v26.4b[0]"),
                    concat!("sdot v", stringify!($hi), ".4s, v25.16b, v26.4b[0]"),
                )+

                // Advance pointers
                "add {a}, {a}, #4",
                "add {b}, {b}, #32",
                "sub {kc}, {kc}, #4",
                "cbnz {kc}, 2b",

                // Store: C[row] += accumulator[row]
                "mov {tmp}, {c}",
                $(
                    concat!("ldr q24, [{tmp}]"),
                    concat!("ldr q25, [{tmp}, #16]"),
                    concat!(
                        "add v", stringify!($lo), ".4s, v",
                        stringify!($lo), ".4s, v24.4s",
                    ),
                    concat!(
                        "add v", stringify!($hi), ".4s, v",
                        stringify!($hi), ".4s, v25.4s",
                    ),
                    concat!("str q", stringify!($lo), ", [{tmp}]"),
                    concat!("str q", stringify!($hi), ", [{tmp}, #16]"),
                    "add {tmp}, {tmp}, {ldc}",
                )+

                a = inout(reg) buffer_a => _,
                b = inout(reg) buffer_b => _,
                c = in(reg) buffer_c,
                kc = inout(reg) kc => _,
                ldc = in(reg) ldc_bytes,
                tmp = out(reg) _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _,
                options(nostack),
            );
        }
    };
}

// (acc_lo, acc_hi, a_byte_offset) — offsets = row * KCB = row * 512
define_neon_i8_kernel!(neon_i8_kernel_1, [(0, 1, 0)]);
define_neon_i8_kernel!(neon_i8_kernel_2, [(0, 1, 0), (2, 3, 512)]);
define_neon_i8_kernel!(neon_i8_kernel_3, [(0, 1, 0), (2, 3, 512), (4, 5, 1024)]);
define_neon_i8_kernel!(
    neon_i8_kernel_4,
    [(0, 1, 0), (2, 3, 512), (4, 5, 1024), (6, 7, 1536)]
);
define_neon_i8_kernel!(
    neon_i8_kernel_5,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_6,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_7,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560),
        (12, 13, 3072)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_8,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560),
        (12, 13, 3072),
        (14, 15, 3584)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_9,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560),
        (12, 13, 3072),
        (14, 15, 3584),
        (16, 17, 4096)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_10,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560),
        (12, 13, 3072),
        (14, 15, 3584),
        (16, 17, 4096),
        (18, 19, 4608)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_11,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560),
        (12, 13, 3072),
        (14, 15, 3584),
        (16, 17, 4096),
        (18, 19, 4608),
        (20, 21, 5120)
    ]
);
define_neon_i8_kernel!(
    neon_i8_kernel_12,
    [
        (0, 1, 0),
        (2, 3, 512),
        (4, 5, 1024),
        (6, 7, 1536),
        (8, 9, 2048),
        (10, 11, 2560),
        (12, 13, 3072),
        (14, 15, 3584),
        (16, 17, 4096),
        (18, 19, 4608),
        (20, 21, 5120),
        (22, 23, 5632)
    ]
);

/// Pack A tile: copy mc rows × kc columns from quantized A (K stride) into
/// a KCB-strided scratchpad, converting uint8→int8 (subtract 128) for SDOT.
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
        for ki in 0..kc {
            // XOR 0x80 converts uint8 [0,255] to int8 [-128,127] stored as u8
            dst[dst_offset + ki] = a_u8[src_offset + ki] ^ 0x80;
        }
    }
}
