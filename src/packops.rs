use anyhow::Result;
use candle_core::{Device, Tensor};
use log::warn;

use super::haarops::{haar_forward_mirror_edge, haar_inverse_mirror_edge};
use super::rans64;

const MAX_WINDOW_ROWS: usize = 1024;
const RANS_BITS: u32 = 12;
const RANGE: f32 = 29.0;
#[cfg(feature = "polar-quant")]
const POLAR_COMPANDING_PARAM: f32 = 255.0;
#[cfg(feature = "polar-quant")]
pub(crate) const DEFAULT_RESIDUAL_QUANT_BITS: u8 = 2;
#[cfg(not(feature = "polar-quant"))]
pub(crate) const DEFAULT_RESIDUAL_QUANT_BITS: u8 = 0;

const fn signed_q4_decode_nibble(code: u8) -> f32 {
    if code >= 8 {
        (code as i16 - 16) as f32
    } else {
        code as f32
    }
}

const fn make_signed_q4_pair_decode_table() -> [[f32; 2]; 256] {
    let mut table = [[0.0; 2]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let low = (byte as u8) & 0x0f;
        let high = (byte as u8) >> 4;
        table[byte] = [signed_q4_decode_nibble(low), signed_q4_decode_nibble(high)];
        byte += 1;
    }
    table
}

const SIGNED_Q4_PAIR_DECODE_TABLE: [[f32; 2]; 256] = make_signed_q4_pair_decode_table();
const SIGNED_Q4_MAX_MAGNITUDE: f32 = 7.0;

pub(crate) fn signed_q4_row_bytes(cols: usize) -> usize {
    cols.div_ceil(2)
}

pub(crate) fn signed_q4_scale(max_abs: f32) -> f32 {
    if max_abs > 0.0 {
        max_abs / SIGNED_Q4_MAX_MAGNITUDE
    } else {
        0.0
    }
}

pub(crate) fn quantize_signed_q4(value: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        return 0;
    }
    let code = (value / scale)
        .round()
        .clamp(-SIGNED_Q4_MAX_MAGNITUDE, SIGNED_Q4_MAX_MAGNITUDE) as i8;
    (code as u8) & 0x0f
}

pub(crate) fn write_signed_q4_code(bytes: &mut [u8], index: usize, code: u8) {
    let byte = &mut bytes[index / 2];
    if index % 2 == 0 {
        *byte = (*byte & 0xf0) | code;
    } else {
        *byte = (*byte & 0x0f) | (code << 4);
    }
}

pub(crate) fn signed_q4_pair_values(byte: u8) -> [f32; 2] {
    SIGNED_Q4_PAIR_DECODE_TABLE[byte as usize]
}

#[cfg(feature = "polar-quant")]
fn quantize_polar_radius(radius: f32, max_radius: f32, max_code: u8) -> u8 {
    let normalized = (radius.clamp(0.0, max_radius)) / max_radius;
    let companded =
        (1.0 + POLAR_COMPANDING_PARAM * normalized).ln() / (1.0 + POLAR_COMPANDING_PARAM).ln();
    let max_code = max_code as f32;
    (companded * max_code).round().clamp(0.0, max_code) as u8
}

#[cfg(feature = "polar-quant")]
fn dequantize_polar_radius(code: u8, max_radius: f32, max_code: u8) -> f32 {
    let companded = (code.min(max_code) as f32) / (max_code as f32);
    let normalized =
        ((1.0 + POLAR_COMPANDING_PARAM).powf(companded) - 1.0) / POLAR_COMPANDING_PARAM;
    normalized * max_radius
}

#[cfg(feature = "polar-quant")]
pub(crate) fn validate_polar_radius_levels(levels: &[f32], bits: u8) -> Result<()> {
    validate_residual_quant_bits(bits)?;
    let expected = 1usize << usize::from(bits);
    anyhow::ensure!(
        levels.len() == expected,
        "Lloyd-Max radius level count {} does not match expected {}",
        levels.len(),
        expected
    );
    let mut previous = f32::NEG_INFINITY;
    for &level in levels {
        anyhow::ensure!(level.is_finite(), "Lloyd-Max radius level is not finite");
        anyhow::ensure!(
            level >= 0.0,
            "Lloyd-Max radius level {level} must be non-negative"
        );
        anyhow::ensure!(level >= previous, "Lloyd-Max radius levels must be sorted");
        previous = level;
    }
    Ok(())
}

#[cfg(feature = "polar-quant")]
fn quantize_polar_radius_with_levels(radius: f32, levels: &[f32], max_code: u8) -> u8 {
    debug_assert_eq!(levels.len(), max_code as usize + 1);
    let mut best = 0usize;
    let mut best_dist = (radius - levels[0]).abs();
    for (idx, &level) in levels.iter().enumerate().skip(1) {
        let dist = (radius - level).abs();
        if dist < best_dist {
            best = idx;
            best_dist = dist;
        }
    }
    best as u8
}

#[cfg(feature = "polar-quant")]
fn lloyd_max_levels_from_sorted(sorted: &[f32], level_count: usize, iterations: usize) -> Vec<f32> {
    debug_assert!(!sorted.is_empty());
    debug_assert!(level_count > 0);
    let mut levels = Vec::with_capacity(level_count);
    for i in 0..level_count {
        let idx = ((2 * i + 1) * sorted.len()) / (2 * level_count);
        levels.push(sorted[idx.min(sorted.len() - 1)]);
    }

    for _ in 0..iterations {
        let mut sums = vec![0f64; level_count];
        let mut counts = vec![0usize; level_count];
        let mut level = 0usize;
        for &sample in sorted {
            while level + 1 < level_count {
                let boundary = 0.5 * (levels[level] + levels[level + 1]);
                if sample <= boundary {
                    break;
                }
                level += 1;
            }
            sums[level] += sample as f64;
            counts[level] += 1;
        }
        for i in 0..level_count {
            if counts[i] != 0 {
                levels[i] = (sums[i] / counts[i] as f64) as f32;
            }
        }
    }
    levels
}

#[cfg(feature = "polar-quant")]
pub(crate) fn lloyd_max_radius_levels(samples: &[f32], bits: u8) -> Result<Vec<f32>> {
    validate_residual_quant_bits(bits)?;
    anyhow::ensure!(
        !samples.is_empty(),
        "cannot train Lloyd-Max radius quantizer without samples"
    );
    let level_count = 1usize << usize::from(bits);
    let mut sorted: Vec<f32> = samples
        .iter()
        .copied()
        .filter(|sample| sample.is_finite() && *sample >= 0.0)
        .collect();
    anyhow::ensure!(
        !sorted.is_empty(),
        "cannot train Lloyd-Max radius quantizer without finite non-negative samples"
    );
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let mut levels = lloyd_max_levels_from_sorted(&sorted, level_count, 20);
    levels.sort_unstable_by(|a, b| a.total_cmp(b));
    validate_polar_radius_levels(&levels, bits)?;
    Ok(levels)
}

#[cfg(feature = "polar-quant")]
fn centroid_confidence_weight(centroid_score: f32, score_range: (f32, f32)) -> f32 {
    let (best_score, tail_score) = score_range;
    let span = best_score - tail_score;
    if span <= f32::EPSILON {
        return 1.0;
    }
    ((centroid_score - tail_score) / span).clamp(0.0, 1.0)
}

#[cfg(feature = "polar-quant")]
const HYPERSPHERICAL_RADIUS_MAX: [f32; 5] = [0.0, 0.0, 1.0, 1.0, 0.5];

#[cfg(feature = "polar-quant")]
fn hyperspherical_row_bytes(cols: usize, bits: u8) -> usize {
    assert!(
        cols % 4 == 0,
        "column count must be divisible by four for hyperspherical packing"
    );
    cols * usize::from(bits) / 8
}

#[cfg(feature = "polar-quant")]
fn hyperspherical_encode_code(values: &[f32], bits: u8, levels: Option<&[f32]>) -> u32 {
    debug_assert_eq!(values.len(), 4);
    let max_code = (1u32 << bits) - 1;
    let radius = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let radius_code = match levels {
        Some(levels) => quantize_polar_radius_with_levels(radius, levels, max_code as u8),
        None => quantize_polar_radius(
            radius,
            HYPERSPHERICAL_RADIUS_MAX[bits as usize],
            max_code as u8,
        ),
    } as u32;
    if radius == 0.0 {
        return radius_code;
    }
    let theta1 = (values[0] / radius).clamp(-1.0, 1.0).acos();
    let theta2 = values[2].hypot(values[3]).atan2(values[1]);
    let phi = values[3].atan2(values[2]).rem_euclid(std::f32::consts::TAU);
    let theta_code = |theta: f32| {
        ((theta / std::f32::consts::PI * max_code as f32).round() as u32).min(max_code)
    };
    let phi_code =
        ((phi / std::f32::consts::TAU * (max_code + 1) as f32).round() as u32) & max_code;
    radius_code
        | (theta_code(theta1) << bits)
        | (theta_code(theta2) << (2 * bits))
        | (phi_code << (3 * bits))
}

#[cfg(feature = "polar-quant")]
fn hyperspherical_decode_code(code: u32, bits: u8, levels: Option<&[f32]>) -> [f32; 4] {
    let max_code = (1u32 << bits) - 1;
    let radius_code = (code & max_code) as u8;
    let radius = levels.map_or_else(
        || {
            dequantize_polar_radius(
                radius_code,
                HYPERSPHERICAL_RADIUS_MAX[bits as usize],
                max_code as u8,
            )
        },
        |levels| levels[radius_code as usize],
    );
    let theta =
        |shift: u8| ((code >> shift) & max_code) as f32 * std::f32::consts::PI / max_code as f32;
    let theta1 = theta(bits);
    let theta2 = theta(2 * bits);
    let phi =
        ((code >> (3 * bits)) & max_code) as f32 * std::f32::consts::TAU / (max_code + 1) as f32;
    let sin_theta1 = theta1.sin();
    let sin_theta2 = theta2.sin();
    [
        radius * theta1.cos(),
        radius * sin_theta1 * theta2.cos(),
        radius * sin_theta1 * sin_theta2 * phi.cos(),
        radius * sin_theta1 * sin_theta2 * phi.sin(),
    ]
}

#[cfg(feature = "polar-quant")]
fn make_hyperspherical_table(bits: u8, levels: Option<&[f32]>) -> Result<Vec<[f32; 4]>> {
    if let Some(levels) = levels {
        validate_polar_radius_levels(levels, bits)?;
    }
    Ok((0..(1usize << (4 * usize::from(bits))))
        .map(|code| hyperspherical_decode_code(code as u32, bits, levels))
        .collect())
}

#[cfg(feature = "polar-quant")]
fn hyperspherical_encode_row(flat: &[f32], bits: u8, levels: Option<&[f32]>) -> Vec<u8> {
    assert!(
        flat.len() % 4 == 0,
        "column count must be divisible by four for hyperspherical packing"
    );
    let code_bits = 4 * usize::from(bits);
    let mut packed = Vec::with_capacity(hyperspherical_row_bytes(flat.len(), bits));
    let mut acc = 0u32;
    let mut acc_bits = 0usize;
    for values in flat.chunks_exact(4) {
        acc |= hyperspherical_encode_code(values, bits, levels) << acc_bits;
        acc_bits += code_bits;
        while acc_bits >= 8 {
            packed.push(acc as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        packed.push(acc as u8);
    }
    debug_assert_eq!(packed.len(), hyperspherical_row_bytes(flat.len(), bits));
    packed
}

#[cfg(feature = "polar-quant")]
fn hyperspherical_decode_rows(bytes: &[u8], cols: usize, bits: u8, table: &[[f32; 4]]) -> Vec<f32> {
    let row_bytes = hyperspherical_row_bytes(cols, bits);
    assert!(
        bytes.len() % row_bytes == 0,
        "packed data length must be divisible by hyperspherical row bytes"
    );
    let code_bits = 4 * usize::from(bits);
    let code_mask = (1u32 << code_bits) - 1;
    let mut out = Vec::with_capacity(bytes.len() * 8 / usize::from(bits));
    for row in bytes.chunks_exact(row_bytes) {
        let mut acc = 0u32;
        let mut acc_bits = 0usize;
        let mut byte_idx = 0usize;
        for _ in 0..cols / 4 {
            while acc_bits < code_bits {
                acc |= (row[byte_idx] as u32) << acc_bits;
                acc_bits += 8;
                byte_idx += 1;
            }
            out.extend_from_slice(&table[(acc & code_mask) as usize]);
            acc >>= code_bits;
            acc_bits -= code_bits;
        }
    }
    out
}

#[cfg(feature = "polar-quant")]
pub(crate) struct PolarDequantTables {
    q2: Vec<[f32; 4]>,
    q3: Vec<[f32; 4]>,
    q4: Vec<[f32; 4]>,
}

#[cfg(feature = "polar-quant")]
pub(crate) type ResidualDequantTable = PolarDequantTables;

#[cfg(not(feature = "polar-quant"))]
pub(crate) type ResidualDequantTable = [f32; 16];

#[cfg(feature = "polar-quant")]
pub(crate) fn validate_residual_quant_bits(bits: u8) -> Result<()> {
    anyhow::ensure!(
        matches!(bits, 2 | 3 | 4),
        "polar residual quantization bits must be 2, 3, or 4, got {bits}"
    );
    Ok(())
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn validate_residual_quant_bits(bits: u8) -> Result<()> {
    anyhow::ensure!(
        bits == 0,
        "index uses polar residual quantization ({bits} bits), but this build was compiled without polar-quant"
    );
    Ok(())
}

#[cfg(feature = "polar-quant")]
pub(crate) fn residual_centroid_confidence_weight(
    centroid_score: f32,
    score_range: (f32, f32),
    bits: u8,
) -> f32 {
    if matches!(bits, 2 | 3) {
        centroid_confidence_weight(centroid_score, score_range)
    } else {
        1.0
    }
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn residual_centroid_confidence_weight(
    _centroid_score: f32,
    _score_range: (f32, f32),
    _bits: u8,
) -> f32 {
    1.0
}

///////////////////////////////////////////////////////////////////////////

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn make_residual_dequant_table() -> Result<ResidualDequantTable> {
    let x = Tensor::arange(0f32, 16f32, &Device::Cpu)?;
    let x = x.dequantize(4)?.inv_compand()?;

    let mut table = [0f32; 16];
    for (i, slot) in table.iter_mut().enumerate() {
        *slot = x.get(i)?.to_scalar::<f32>()?;
    }

    Ok(table)
}

#[cfg(feature = "polar-quant")]
pub(crate) fn make_residual_dequant_table_with_radius_levels(
    bits: u8,
    levels: Option<&[f32]>,
) -> Result<ResidualDequantTable> {
    validate_residual_quant_bits(bits)?;
    Ok(PolarDequantTables {
        q2: if bits == 2 {
            make_hyperspherical_table(2, levels)?
        } else {
            make_hyperspherical_table(2, None)?
        },
        q3: if bits == 3 {
            make_hyperspherical_table(3, levels)?
        } else {
            make_hyperspherical_table(3, None)?
        },
        q4: if bits == 4 {
            make_hyperspherical_table(4, levels)?
        } else {
            make_hyperspherical_table(4, None)?
        },
    })
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn make_residual_dequant_table_with_radius_levels(
    bits: u8,
    levels: Option<&[f32]>,
) -> Result<ResidualDequantTable> {
    anyhow::ensure!(
        levels.is_none(),
        "Lloyd-Max polar radius levels require polar-quant"
    );
    validate_residual_quant_bits(bits)?;
    make_residual_dequant_table()
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn temp_residual_bytes_for_dim(dim: usize, bits: u8) -> Result<usize> {
    validate_residual_quant_bits(bits)?;
    assert!(
        dim % 2 == 0,
        "embedding dimension must be even for q4 residuals"
    );
    Ok(dim / 2)
}

#[cfg(feature = "polar-quant")]
pub(crate) fn temp_residual_bytes_for_dim(dim: usize, bits: u8) -> Result<usize> {
    polar_row_bytes(dim, bits)
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn residual_to_temp_bytes(residual: &Tensor, bits: u8) -> Result<Vec<u8>> {
    validate_residual_quant_bits(bits)?;
    residual.compand()?.quantize(4)?.to_q4_bytes()
}

#[cfg(feature = "polar-quant")]
pub(crate) fn residual_to_temp_bytes_with_radius_levels(
    residual: &Tensor,
    bits: u8,
    levels: Option<&[f32]>,
) -> Result<Vec<u8>> {
    residual.to_polar_bytes_with_radius_levels(bits, levels)
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn residual_to_temp_bytes_with_radius_levels(
    residual: &Tensor,
    bits: u8,
    _levels: Option<&[f32]>,
) -> Result<Vec<u8>> {
    residual_to_temp_bytes(residual, bits)
}

#[cfg(not(feature = "polar-quant"))]
pub(crate) fn residuals_from_bytes(
    bytes: &[u8],
    cols: usize,
    table: &ResidualDequantTable,
    bits: u8,
    device: &Device,
) -> Result<Tensor> {
    validate_residual_quant_bits(bits)?;
    Tensor::from_companded_q4_bytes(bytes, cols, table, device)
}

#[cfg(feature = "polar-quant")]
pub(crate) fn residuals_from_bytes(
    bytes: &[u8],
    cols: usize,
    table: &ResidualDequantTable,
    bits: u8,
    device: &Device,
) -> Result<Tensor> {
    validate_residual_quant_bits(bits)?;
    let out = match bits {
        2 => hyperspherical_decode_rows(bytes, cols, 2, &table.q2),
        3 => hyperspherical_decode_rows(bytes, cols, 3, &table.q3),
        4 => hyperspherical_decode_rows(bytes, cols, 4, &table.q4),
        _ => unreachable!(),
    };
    assert!(
        out.len() % cols == 0,
        "Unpacked data length ({}) must be divisible by cols ({})",
        out.len(),
        cols
    );
    let rows = out.len() / cols;
    Ok(Tensor::from_vec(out, &[rows, cols], device)?)
}

#[cfg(feature = "polar-quant")]
pub(crate) fn polar_row_bytes(cols: usize, bits: u8) -> Result<usize> {
    validate_residual_quant_bits(bits)?;
    Ok(hyperspherical_row_bytes(cols, bits))
}

/// Normalize a histogram so that it sums to 2^log2_scale (<= 2^16),
/// suitable for rANS.
///
/// Takes symbols as a slice of (symbol, count) pairs and returns
/// a Vec of (symbol, scaled_count) pairs.
pub fn scale_histogram(symbols: &[(u16, u32)], log2_scale: u32) -> Vec<(u16, u16)> {
    assert!(!symbols.is_empty(), "Histogram must not be empty");
    assert!(log2_scale <= 16, "log2_scale must be <= 16");

    let target_total: u32 = 1 << log2_scale;
    let m = symbols.len() as u32;

    assert!(
        m <= target_total,
        "Not enough room to assign at least 1 frequency to every symbol"
    );

    // We give each symbol at least 1, the rest is distributed proportionally.
    let remaining_total = target_total - m;

    // Sum of input counts.
    let sum: u64 = symbols.iter().map(|&(_, c)| c as u64).sum();

    // Use Vec of (symbol, scaled_freq) instead of HashMap
    let mut scaled: Vec<(u16, u32)> = Vec::with_capacity(symbols.len());
    let mut remainders: Vec<(u64, usize)> = Vec::with_capacity(symbols.len());

    // Initialize with 1 for each symbol and compute quotients/remainders
    let mut acc: u32 = 0;

    for (idx, &(sym, count)) in symbols.iter().enumerate() {
        let num = (count as u64) * (remaining_total as u64);
        let q = (num / sum) as u32;
        let r = num % sum;

        scaled.push((sym, 1 + q));
        acc += q;
        remainders.push((r, idx));
    }

    // Distribute any leftover slots based on largest remainders.
    let deficit = remaining_total - acc;
    if deficit > 0 {
        remainders.sort_by(|a, b| b.0.cmp(&a.0));
        for &(_, idx) in remainders.iter().take(deficit as usize) {
            scaled[idx].1 += 1;
        }
    }

    // Final fixup: guarantee exact total == target_total.
    let mut total: i64 = scaled.iter().map(|(_, v)| *v as i64).sum();
    let target = target_total as i64;

    if total != target {
        // Pick symbol with highest value for fixup to minimize distortion.
        let max_idx = scaled
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, v))| *v)
            .map(|(i, _)| i)
            .unwrap();

        while total < target {
            scaled[max_idx].1 += 1;
            total += 1;
        }
        while total > target && scaled[max_idx].1 > 1 {
            scaled[max_idx].1 -= 1;
            total -= 1;
        }
    }

    // Convert to u16 safely (target_total <= 2^16).
    scaled
        .into_iter()
        .map(|(sym, freq)| (sym, freq as u16))
        .collect()
}

pub trait TensorPackOps {
    #[cfg(not(feature = "polar-quant"))]
    fn compand(&self) -> Result<Tensor>;
    #[cfg(not(feature = "polar-quant"))]
    fn inv_compand(&self) -> Result<Tensor>;
    #[cfg(any(not(feature = "polar-quant"), debug_assertions))]
    fn quantize(&self, bits: u32) -> Result<Tensor>;
    #[cfg(any(not(feature = "polar-quant"), debug_assertions))]
    fn dequantize(&self, bits: u32) -> Result<Tensor>;
    fn embeddings_to_packed(&self) -> Result<Vec<u8>>;
    fn embeddings_from_packed(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;

    #[cfg(not(feature = "polar-quant"))]
    fn to_q4_bytes(&self) -> Result<Vec<u8>>;
    #[cfg(not(feature = "polar-quant"))]
    fn from_companded_q4_bytes(
        bytes: &[u8],
        cols: usize,
        table: &[f32; 16],
        device: &Device,
    ) -> Result<Tensor>;
    #[cfg(feature = "polar-quant")]
    fn to_polar_bytes(&self, bits: u8) -> Result<Vec<u8>>;
    #[cfg(feature = "polar-quant")]
    fn to_polar_bytes_with_radius_levels(
        &self,
        bits: u8,
        levels: Option<&[f32]>,
    ) -> Result<Vec<u8>>;

    fn from_f32_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn to_f32_bytes(&self) -> Result<Vec<u8>>;
}

impl TensorPackOps for Tensor {
    /* mu-law companding to improve quantization of residuals. Scale input by 4 to expand
    to full [-1;1] range, as empirically residuals of normalized embeddings rarely exceed
    [-0.26:0.26] range.

    The inverse operation is really slow, so we use a decoding table, as seen in
    from_companded_q4_bytes()

    See also https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    */

    #[cfg(not(feature = "polar-quant"))]
    fn compand(&self) -> Result<Tensor> {
        let scale_param = 4.0;
        let companding_param = 255.0;
        let inv_denominator = 1.0f64 / (1.0f64 + companding_param).ln();
        let x = (self * scale_param)?;
        Ok((&x.sign()? * (((&x.abs()? * companding_param)? + 1.0)?.log()? * inv_denominator)?)?)
    }

    #[cfg(not(feature = "polar-quant"))]
    fn inv_compand(&self) -> Result<Tensor> {
        let inv_scale_param = 1.0 / 4.0;
        let companding_param = 255.0;
        let inv_companding_param = 1.0 / companding_param;
        let ones = Tensor::ones_like(self)?;
        let abs = self.abs()?;
        let sign = self.sign()?;
        let scaled =
            (sign * (((&ones + companding_param)?.pow(&abs)? - 1.0)? * inv_companding_param)?)?;
        Ok((&scaled * inv_scale_param)?)
    }

    #[cfg(any(not(feature = "polar-quant"), debug_assertions))]
    fn quantize(&self, bits: u32) -> Result<Tensor> {
        let range = 1 << bits;
        let qmax = (range - 1) as f64;
        let scale1 = qmax / 2.0;
        let zp = qmax / 2.0;
        Ok(((self * scale1)? + zp)?.round()?.clamp(0.0, qmax)?)
    }

    #[cfg(any(not(feature = "polar-quant"), debug_assertions))]
    fn dequantize(&self, bits: u32) -> Result<Tensor> {
        let range = 1 << bits;
        let qmax = (range - 1) as f64;
        let scale2 = 2.0 / qmax;
        let zp = qmax / 2.0;
        Ok(((self - zp)? * scale2)?)
    }

    fn embeddings_to_packed(&self) -> Result<Vec<u8>> {
        let (rows, cols) = self.dims2()?;
        assert!(cols <= 255, "column count must fit in u8");
        let scaled_range = (256.0 * RANGE).round() as u16;
        let range = scaled_range as f32 / 256.0;

        let mut bytes = Vec::with_capacity(2 + 4 + rows * cols);
        bytes.extend_from_slice(&scaled_range.to_ne_bytes());
        bytes.extend_from_slice(&(rows as u32).to_ne_bytes());

        let all_data = self.flatten_all()?.to_vec1::<f32>()?;

        for (offset, win_rows) in (0..rows)
            .step_by(MAX_WINDOW_ROWS)
            .map(|r| (r, (rows - r).min(MAX_WINDOW_ROWS)))
        {
            assert!(win_rows > 0);

            // Extract window into raw buffer
            let mut raw = all_data[offset * cols..(offset + win_rows) * cols].to_vec();

            // Sort columns by sum (compute sums, build permutation, apply)
            let mut col_sums = vec![0.0f32; cols];
            for r in 0..win_rows {
                for c in 0..cols {
                    col_sums[c] += raw[r * cols + c];
                }
            }
            let mut perm: Vec<usize> = (0..cols).collect();
            perm.sort_by(|&a, &b| col_sums[a].partial_cmp(&col_sums[b]).unwrap());

            let mut inv_perm = vec![0usize; cols];
            for (new_pos, &orig_col) in perm.iter().enumerate() {
                inv_perm[orig_col] = new_pos;
            }
            for i in &inv_perm {
                bytes.push(*i as u8);
            }

            // Apply column permutation in place
            let mut row_buf = vec![0.0f32; cols];
            for r in 0..win_rows {
                let row_start = r * cols;
                for c in 0..cols {
                    row_buf[c] = raw[row_start + perm[c]];
                }
                raw[row_start..row_start + cols].copy_from_slice(&row_buf);
            }

            // Haar forward on each row (contiguous)
            for r in 0..win_rows {
                haar_forward_mirror_edge(&mut raw[r * cols..(r + 1) * cols]);
            }

            // Haar forward on each column (strided, temp buffer)
            let mut col_buf = vec![0.0f32; win_rows];
            for c in 0..cols {
                for r in 0..win_rows {
                    col_buf[r] = raw[r * cols + c];
                }
                haar_forward_mirror_edge(&mut col_buf);
                for r in 0..win_rows {
                    raw[r * cols + c] = col_buf[r];
                }
            }
            let mut abs_max = 0.00001f32;
            for x in &raw {
                let a = x.abs();
                if a < 1.0 && a > abs_max {
                    abs_max = a;
                }
            }
            let inv_max_val = 1.0 / abs_max;
            for i in raw.iter_mut() {
                *i *= inv_max_val;
            }

            // Quantize values - pre-allocate capacity
            let mut qs: Vec<u16> = Vec::with_capacity(raw.len());
            let mut max_symbol = 0u16;

            for &x in &raw {
                let q = (range * x).round();
                let s = (2.0 * q.abs() + if q < 0.0 { 1.0 } else { 0.0 }) as usize;
                let symbol = if s > 0 { s - 1 } else { 0 } as u16;
                qs.push(symbol);
                max_symbol = max_symbol.max(symbol);
            }

            // Build histogram using array (now that we know max_symbol)
            let mut hist_array: Vec<u32> = vec![0; (max_symbol + 1) as usize];
            for &symbol in &qs {
                hist_array[symbol as usize] += 1;
            }

            // Convert array histogram to Vec for scale_histogram
            let mut symbols: Vec<(u16, u32)> = Vec::new();
            for (symbol, &count) in hist_array.iter().enumerate() {
                if count > 0 {
                    symbols.push((symbol as u16, count));
                }
            }

            let mut hist = scale_histogram(&symbols, RANS_BITS);
            hist.sort_by_key(|(k, _v)| *k);

            bytes.extend_from_slice(&(hist.len() as u16).to_ne_bytes());

            // Use Vec instead of HashMap for faster lookup during encoding
            let mut q2sym: Vec<Option<rans64::RansEncSymbol>> =
                vec![None; (max_symbol + 1) as usize];
            let mut cum = 0u32;

            for &(q, freq) in &hist {
                let freq_u32 = freq as u32;
                q2sym[q as usize] = Some(rans64::RansEncSymbol::new(cum, freq_u32, RANS_BITS));
                bytes.extend_from_slice(&q.to_ne_bytes());
                bytes.extend_from_slice(&freq.to_ne_bytes());
                cum += freq_u32;
            }

            let mut encoder = rans64::RansEncoder::new(2 * win_rows * cols);
            for &q in &qs {
                // Direct array access instead of HashMap lookup
                encoder.put(q2sym[q as usize].as_ref().unwrap());
            }
            encoder.flush();
            let data = encoder.data().to_owned();
            bytes.extend_from_slice(&(data.len() as u16).to_ne_bytes());
            bytes.extend_from_slice(&data);
        }
        Ok(bytes)
    }

    fn embeddings_from_packed(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let (head, bytes) = bytes.split_at(2);
        let scaled_range = u16::from_ne_bytes(head.try_into().unwrap()) as f32;
        let scale = 256.0 / scaled_range;
        let (head, mut bytes) = bytes.split_at(4);
        let rows = u32::from_ne_bytes(head.try_into().unwrap()) as usize;

        let mut data = Vec::with_capacity(rows * cols);

        for (_offset, win_rows) in (0..rows)
            .step_by(MAX_WINDOW_ROWS)
            .map(|r| (r, (rows - r).min(MAX_WINDOW_ROWS)))
        {
            let (idxs, tail) = bytes.split_at(cols);
            let idxs: Vec<usize> = idxs.iter().map(|&i| i as usize).collect();

            let (head, tail) = tail.split_at(2);
            let symbols_count = u16::from_ne_bytes(head.try_into().unwrap()) as usize;

            let (symbols, tail) = tail.split_at(4 * symbols_count);
            let pairs: Vec<u16> = symbols
                .chunks_exact(2)
                .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
                .collect();

            // Build merged lookup tables indexed directly by cumulative frequency
            let rans_size = 1 << RANS_BITS;
            let default_sym = rans64::RansDecSymbol::new(0, 1)?;
            let mut cum2float: Vec<f32> = vec![0.0; rans_size];
            let mut cum2sym: Vec<rans64::RansDecSymbol> = vec![default_sym; rans_size];

            // First pass: build symbol -> float and symbol -> dec_symbol tables
            let mut max_symbol = 0u16;
            for i in 0..symbols_count {
                max_symbol = max_symbol.max(pairs[2 * i]);
            }
            let mut q2float: Vec<f32> = Vec::with_capacity((max_symbol + 1) as usize);
            for q in 0..=max_symbol {
                let x = if q == 0 {
                    0.0
                } else {
                    let q = q + 1;
                    let sign = if (q & 1) == 1 { -1.0 } else { 1.0 };
                    let magnitude = (q >> 1) as f32;
                    scale * sign * magnitude
                };
                q2float.push(x);
            }

            let mut q2sym: Vec<rans64::RansDecSymbol> =
                vec![rans64::RansDecSymbol::new(0, 1)?; (max_symbol + 1) as usize];

            // Second pass: build cum -> (float, sym) merged tables
            let mut cum = 0u32;
            for i in 0..symbols_count {
                let symbol = pairs[2 * i];
                let freq = pairs[2 * i + 1] as u32;
                let dec_sym = rans64::RansDecSymbol::new(cum, freq)?;
                q2sym[symbol as usize] = dec_sym.clone();
                let float_val = q2float[symbol as usize];
                for c in cum..cum + freq {
                    cum2float[c as usize] = float_val;
                    cum2sym[c as usize] = dec_sym.clone();
                }
                cum += freq;
            }

            let (head, tail) = tail.split_at(2);
            let compressed_size = u16::from_ne_bytes(head.try_into().unwrap()) as usize;

            let (head, tail) = tail.split_at(compressed_size);
            let mut decoder = rans64::RansDecoder::new(head.to_owned())?;

            // Decode rANS directly into buffer in reverse order
            let win_size = win_rows * cols;
            let win_start = data.len();
            data.resize(win_start + win_size, 0.0f32);
            let t = &mut data[win_start..];

            for i in (0..win_size).rev() {
                let cum = decoder.get(RANS_BITS) as usize;
                t[i] = cum2float[cum];
                if let Err(e) = decoder.advance(&cum2sym[cum], RANS_BITS) {
                    warn!("RANS decoding failed");
                    return Err(e.into());
                }
            }

            // Haar inverse on each row (contiguous in memory)
            for r in 0..win_rows {
                haar_inverse_mirror_edge(&mut t[r * cols..(r + 1) * cols]);
            }

            // Haar inverse on each column (strided, uses temp buffer)
            let mut col_buf = vec![0.0f32; win_rows];
            for c in 0..cols {
                for r in 0..win_rows {
                    col_buf[r] = t[r * cols + c];
                }
                haar_inverse_mirror_edge(&mut col_buf);
                for r in 0..win_rows {
                    t[r * cols + c] = col_buf[r];
                }
            }

            // Restore column permutation in place
            let mut row_buf = vec![0.0f32; cols];
            for r in 0..win_rows {
                let row_start = r * cols;
                for c in 0..cols {
                    row_buf[c] = t[row_start + idxs[c]];
                }
                t[row_start..row_start + cols].copy_from_slice(&row_buf);
            }

            bytes = tail;
        }
        anyhow::ensure!(
            bytes.is_empty(),
            "leftover {} bytes after decoding {} rows x {} cols",
            bytes.len(),
            rows,
            cols
        );

        // L2 normalize all rows
        for r in 0..rows {
            let row = &mut data[r * cols..(r + 1) * cols];
            let norm_sq: f32 = row.iter().map(|x| x * x).sum();
            let inv_norm = 1.0 / norm_sq.sqrt();
            for x in row.iter_mut() {
                *x *= inv_norm;
            }
        }

        Ok(Tensor::from_vec(data, &[rows, cols], device)?)
    }

    #[cfg(not(feature = "polar-quant"))]
    fn to_q4_bytes(&self) -> Result<Vec<u8>> {
        let flat = self.flatten_all()?.to_vec1::<f32>()?;
        /*
        let mut hist: [u32; 16] = [0; 16];
        for i in &flat {
            hist[*i as usize] += 1;
        }
        println!("hist {:?}", hist);
        */

        assert!(
            flat.len() % 2 == 0,
            "Tensor must have an even number of elements to pack"
        );

        let mut packed = Vec::with_capacity(flat.len() / 2);
        for chunk in flat.chunks_exact(2) {
            let high = chunk[0] as u8 & 0x0f;
            let low = chunk[1] as u8 & 0x0f;
            packed.push((high << 4) | low);
        }
        Ok(packed)
    }

    fn to_f32_bytes(&self) -> Result<Vec<u8>> {
        let floats: Vec<f32> = self.flatten_all()?.to_vec1::<f32>()?;
        let mut bytes = Vec::with_capacity(floats.len() * 4);

        for f in floats {
            bytes.extend_from_slice(&f.to_ne_bytes());
        }
        Ok(bytes)
    }

    fn from_f32_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let f32_size = size_of::<f32>();

        assert!(bytes.len() % f32_size == 0);
        let total_f32s = bytes.len() / f32_size;

        let rows = total_f32s / cols;

        let mut f32s = Vec::with_capacity(total_f32s);
        for chunk in bytes.chunks_exact(f32_size) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32s.push(f32::from_ne_bytes(arr));
        }

        Ok(Tensor::from_vec(f32s, &[rows, cols], device)?)
    }

    #[cfg(not(feature = "polar-quant"))]
    fn from_companded_q4_bytes(
        bytes: &[u8],
        cols: usize,
        table: &[f32; 16],
        device: &Device,
    ) -> Result<Tensor> {
        let mut out = Vec::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            let high = (byte >> 4) & 0x0f;
            let low = byte & 0x0f;
            out.push(table[high as usize]);
            out.push(table[low as usize]);
        }

        assert!(
            out.len() % cols == 0,
            "Unpacked data length ({}) must be divisible by cols ({})",
            out.len(),
            cols
        );
        let rows = out.len() / cols;
        Ok(Tensor::from_vec(out, &[rows, cols], device)?)
    }

    #[cfg(feature = "polar-quant")]
    fn to_polar_bytes(&self, bits: u8) -> Result<Vec<u8>> {
        let flat = self.flatten_all()?.to_vec1::<f32>()?;
        validate_residual_quant_bits(bits)?;
        Ok(hyperspherical_encode_row(&flat, bits, None))
    }

    #[cfg(feature = "polar-quant")]
    fn to_polar_bytes_with_radius_levels(
        &self,
        bits: u8,
        levels: Option<&[f32]>,
    ) -> Result<Vec<u8>> {
        let Some(levels) = levels else {
            return self.to_polar_bytes(bits);
        };
        let flat = self.flatten_all()?.to_vec1::<f32>()?;
        validate_residual_quant_bits(bits)?;
        validate_polar_radius_levels(levels, bits)?;
        Ok(hyperspherical_encode_row(&flat, bits, Some(levels)))
    }

    /*
    fn from_companded_q8_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let x = Tensor::arange(0.0f32, 256.0f32, &Device::Cpu)?;
        let x = x.dequantize(8)?.inv_compand()?;
        let mut table : [f32; 256] = [0.0; 256];
        for i in 0..256 {
            table[i] = x.get(i)?.to_scalar()?;
        }

        let mut out = Vec::with_capacity(bytes.len());
        for i in bytes {
            out.push(table[*i as usize]);
        }

        assert!(
            out.len() % cols == 0,
            "Unpacked data length ({}) must be divisible by cols ({})",
            out.len(),
            cols
        );
        let rows = out.len() / cols;
        Ok(Tensor::from_vec(out, &[rows, cols], device)?)
    }
    */
}

#[cfg(all(test, feature = "polar-quant"))]
mod hyperspherical_tests {
    use super::*;

    #[test]
    fn hyperspherical_residuals_round_trip_at_all_bit_rates() -> Result<()> {
        let values: Vec<f32> = vec![
            0.10, -0.07, 0.03, -0.02, -0.04, 0.09, -0.08, 0.01, 0.02, 0.03, -0.11, 0.05, -0.06,
            -0.02, 0.04, 0.08,
        ];
        let residuals = Tensor::from_vec(values, &[2, 8], &Device::Cpu)?;

        for bits in [2, 3, 4] {
            let levels = lloyd_max_radius_levels(&[0.0, 0.05, 0.1, 0.2, 0.4, 0.8], bits)?;
            let bytes = residual_to_temp_bytes_with_radius_levels(&residuals, bits, Some(&levels))?;
            assert_eq!(bytes.len(), 2 * polar_row_bytes(8, bits)?);
            let table = make_residual_dequant_table_with_radius_levels(bits, Some(&levels))?;
            let decoded = residuals_from_bytes(&bytes, 8, &table, bits, &Device::Cpu)?;
            assert_eq!(decoded.dims2()?, (2, 8));
            assert!(decoded
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.is_finite()));
        }
        Ok(())
    }

    #[test]
    fn hyperspherical_axis_encoding_is_finite() {
        for bits in [2, 3, 4] {
            let code = hyperspherical_encode_code(&[0.25, 0.0, 0.0, 0.0], bits, None);
            let decoded = hyperspherical_decode_code(code, bits, None);
            assert!(decoded.iter().all(|value| value.is_finite()));
            assert!(decoded[0] > 0.0);
        }
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize() -> Result<()> {
        let x = Tensor::arange(0.0f32, 16.0f32, &Device::Cpu)?;
        println!("x={}", x);
        let x = x.dequantize(4)?;
        println!("x={}", x);
        Ok(())
    }
    #[test]
    fn test_quantize() -> Result<()> {
        let x1 = Tensor::randn(0f32, 0.26f32, (1, 128), &Device::Cpu)?;
        let bytes = x1.quantize(8)?.to_q8_bytes()?;
        let x2 = Tensor::from_q8_bytes(&bytes, 128, &Device::Cpu)?.dequantize(8)?;
        let mse = (&x2 - &x1)?.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;
        println!("mse {}", mse);
        assert!(mse < 0.05);
        Ok(())
    }
    #[test]
    fn test_stretch_quantize() -> Result<()> {
        let x1 = Tensor::randn(0f32, 1.0f32, (1, 128), &Device::Cpu)?.l2_normalize()?;
        let bytes = x1.stretch_rows()?.quantize(8)?.to_q8_bytes()?;
        let x2 = Tensor::from_q8_bytes(&bytes, 128, &Device::Cpu)?
            .dequantize(8)?
            .l2_normalize()?;
        let mse = (&x2 - &x1)?.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;
        println!("mse {}", mse);
        assert!(mse < 0.001);
        Ok(())
    }
    #[test]
    fn test_compand() -> Result<()> {
        let x1 = Tensor::randn(0f32, 0.26f32, 200, &Device::Cpu)?;
        let x2 = x1.compand()?.inv_compand()?;
        let mse = (&x2 - &x1)?.powf(2.0)?.sum_all()?.to_scalar::<f32>()?;
        assert!(mse < 0.001);
        Ok(())
    }
}
*/
