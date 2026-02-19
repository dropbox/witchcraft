use anyhow::Result;
use candle_core::{Device, Tensor};
use log::warn;

use super::haarops::{haar_forward_mirror_edge, haar_inverse_mirror_edge};
use super::rans64;

const MAX_WINDOW_ROWS: usize = 1024;
const RANS_BITS: u32 = 12;
const RANGE: f32 = 29.0;

///////////////////////////////////////////////////////////////////////////

pub fn make_q4_dequant_table() -> Result<[f32; 16]> {
    // Create 0..16 tensor
    let x = Tensor::arange(0f32, 16f32, &Device::Cpu)?;
    let x = x.dequantize(4)?.inv_compand()?;

    // Extract into a fixed array
    let mut table = [0f32; 16];
    for i in 0..16 {
        table[i] = x.get(i)?.to_scalar::<f32>()?;
    }

    Ok(table)
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
    fn compand(&self) -> Result<Tensor>;
    fn inv_compand(&self) -> Result<Tensor>;
    fn quantize(&self, bits: u32) -> Result<Tensor>;
    fn dequantize(&self, bits: u32) -> Result<Tensor>;
    fn embeddings_to_packed(&self) -> Result<Vec<u8>>;
    fn embeddings_from_packed(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;

    fn to_q4_bytes(&self) -> Result<Vec<u8>>;
    fn from_companded_q4_bytes(
        bytes: &[u8],
        cols: usize,
        table: &[f32; 16],
        device: &Device,
    ) -> Result<Tensor>;

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

    fn compand(&self) -> Result<Tensor> {
        let scale_param = 4.0;
        let companding_param = 255.0;
        let inv_denominator = 1.0f64 / (1.0f64 + companding_param).ln();
        let x = (self * scale_param)?;
        Ok((&x.sign()? * (((&x.abs()? * companding_param)? + 1.0)?.log()? * inv_denominator)?)?)
    }

    fn inv_compand(&self) -> Result<Tensor> {
        let inv_scale_param = 1.0 / 4.0;
        let companding_param = 255.0;
        let inv_companding_param = 1.0 / companding_param;
        let ones = Tensor::ones_like(&self)?;
        let abs = self.abs()?;
        let sign = self.sign()?;
        let scaled =
            (sign * (((&ones + companding_param)?.pow(&abs)? - 1.0)? * inv_companding_param)?)?;
        Ok((&scaled * inv_scale_param)?)
    }

    fn quantize(&self, bits: u32) -> Result<Tensor> {
        let range = 1 << bits;
        let qmax = (range - 1) as f64;
        let scale1 = qmax / 2.0;
        let zp = qmax / 2.0;
        Ok(((self * scale1)? + zp)?.round()?.clamp(0.0, qmax)?)
    }

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

        let mut bytes = Vec::with_capacity(rows * cols);
        bytes.extend_from_slice(&(rows as u16).to_ne_bytes());

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
                let q = (RANGE * x).round();
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
                bytes.extend_from_slice(&(q as u16).to_ne_bytes());
                bytes.extend_from_slice(&(freq as u16).to_ne_bytes());
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
        let scale = 1.0 / RANGE;

        let (head, mut bytes) = bytes.split_at(2);
        let rows = u16::from_ne_bytes(head.try_into().unwrap()) as usize;

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
        assert!(bytes.len() == 0);

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
        for chunk in flat.chunks(2) {
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

        Ok(Tensor::from_vec(f32s, &[rows, cols], &device)?)
    }

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
