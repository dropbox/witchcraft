use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use log::warn;

use super::rans64;
use super::TensorHaarOps;

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

pub fn scale_histogram(hist: &HashMap<u16, u32>, log2_scale: u32) -> HashMap<u16, u16> {
    assert!(!hist.is_empty(), "Histogram must not be empty");
    assert!(log2_scale <= 16, "log2_scale must be <= 16");

    let target_total: u32 = 1 << log2_scale;

    // Collect only symbols with non-zero counts.
    let symbols: Vec<(u16, u32)> = hist
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(&s, &c)| (s, c))
        .collect();

    assert!(
        !symbols.is_empty(),
        "Histogram must contain at least one non-zero symbol"
    );

    let m = symbols.len() as u32;
    assert!(
        m <= target_total,
        "Not enough room to assign at least 1 frequency to every symbol"
    );

    // We give each symbol at least 1, the rest is distributed proportionally.
    let remaining_total = target_total - m;

    // Sum of input counts.
    let sum: u64 = symbols.iter().map(|&(_, c)| c as u64).sum();

    let mut scaled: HashMap<u16, u32> = HashMap::with_capacity(symbols.len());
    let mut remainders: Vec<(u64, u16)> = Vec::with_capacity(symbols.len());

    // First assign 1 to everyone to guarantee no zeros.
    for &(sym, _) in &symbols {
        scaled.insert(sym, 1);
    }

    // Floor-scaling the "extra" part.
    let mut acc: u32 = 0;

    for &(sym, count) in &symbols {
        let num = (count as u64) * (remaining_total as u64);
        let q = (num / sum) as u32;
        let r = num % sum;

        *scaled.get_mut(&sym).unwrap() += q;
        acc += q;

        remainders.push((r, sym));
    }

    // Distribute any leftover slots based on largest remainders.
    let deficit = remaining_total - acc;
    if deficit > 0 {
        remainders.sort_by(|a, b| b.0.cmp(&a.0));
        for &(_, sym) in remainders.iter().take(deficit as usize) {
            *scaled.get_mut(&sym).unwrap() += 1;
        }
    }

    // Final fixup: guarantee exact total == target_total.
    let mut total: i64 = scaled.values().map(|&v| v as i64).sum();
    let target = target_total as i64;

    if total != target {
        // Pick symbol with highest value for fixup to minimize distortion.
        let max_sym = scaled
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(&s, _)| s)
            .unwrap();

        while total < target {
            *scaled.get_mut(&max_sym).unwrap() += 1;
            total += 1;
        }
        while total > target && scaled[&max_sym] > 1 {
            *scaled.get_mut(&max_sym).unwrap() -= 1;
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
    fn l2_normalize(&self) -> Result<Tensor>;
    fn sort_by_column_sum(&self) -> Result<(Tensor, Vec<usize>)>;
    fn restore_columns(&self, inv_perm: &[usize]) -> Result<Tensor>;

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

    fn sort_by_column_sum(&self) -> Result<(Tensor, Vec<usize>)> {
        let (_, cols) = self.dims2()?;
        let sums: Vec<f32> = self.sum(0)?.to_vec1()?;

        let mut perm: Vec<usize> = (0..cols).collect();
        perm.sort_by(|&a, &b| sums[a].partial_cmp(&sums[b]).unwrap());

        let mut inv_perm = vec![0usize; cols];
        for (new_pos, &orig_col) in perm.iter().enumerate() {
            inv_perm[orig_col] = new_pos;
        }

        let idx_i64: Vec<i64> = perm.iter().map(|&i| i as i64).collect();
        let idx_t = Tensor::from_vec(idx_i64, (cols,), self.device())?;
        let sorted = self.index_select(&idx_t, 1)?;
        Ok((sorted, inv_perm))
    }

    fn restore_columns(&self, inv_perm: &[usize]) -> Result<Tensor> {
        let (_, cols) = self.dims2()?;
        assert_eq!(cols, inv_perm.len());
        let idx_i64: Vec<i64> = inv_perm.iter().map(|&i| i as i64).collect();
        let idx_t = Tensor::from_vec(idx_i64, (cols,), self.device())?;
        Ok(self.index_select(&idx_t, 1)?)
    }

    fn embeddings_to_packed(&self) -> Result<Vec<u8>> {
        let (rows, cols) = self.dims2()?;
        assert!(cols <= 255, "column count must fit in u8");

        let mut bytes = vec![];
        bytes.extend_from_slice(&(rows as u16).to_ne_bytes());

        for (offset, win_rows) in (0..rows)
            .step_by(MAX_WINDOW_ROWS)
            .map(|r| (r, (rows - r).min(MAX_WINDOW_ROWS)))
        {
            assert!(win_rows > 0);
            let chunk = self.narrow(0, offset, win_rows)?;
            let (sorted, idxs) = chunk.sort_by_column_sum()?;

            for i in idxs {
                bytes.push(i as u8);
            }

            let haar = sorted
                .t()?
                .haar_forward_tensor_cols()?
                .t()?
                .haar_forward_tensor_cols()?;

            let mut raw = haar.flatten_all()?.to_vec1::<f32>()?;
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

            let qs: Vec<_> = raw
                .iter()
                .map(|&x| {
                    let q = (RANGE * x).round();
                    let s = (2.0 * q.abs() + if q < 0.0 { 1.0 } else { 0.0 }) as usize;
                    if s > 0 {
                        s - 1
                    } else {
                        0
                    }
                })
                .collect();

            let mut hist: HashMap<u16, u32> = HashMap::new();
            for &q in &qs {
                *hist.entry(q as u16).or_insert(0) += 1;
            }

            let hist = scale_histogram(&hist, RANS_BITS);
            let mut hist: Vec<_> = hist.iter().collect();
            hist.sort_by_key(|(k, _v)| *k);

            bytes.extend_from_slice(&(hist.len() as u16).to_ne_bytes());
            let mut cum = 0u32;
            let mut q2sym: HashMap<u16, rans64::RansEncSymbol> = HashMap::new();
            for &(&q, &freq) in hist.iter() {
                //println!("q={q} freq={freq}");
                let freq_u32 = freq as u32;
                q2sym.insert(q, rans64::RansEncSymbol::new(cum, freq_u32, RANS_BITS));
                bytes.extend_from_slice(&(q as u16).to_ne_bytes());
                bytes.extend_from_slice(&(freq as u16).to_ne_bytes());
                cum += freq_u32;
            }

            let mut encoder = rans64::RansEncoder::new(2 * win_rows * cols);
            for &q in &qs {
                encoder.put(&q2sym.get(&(q as u16)).unwrap());
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

        let mut ts = vec![];
        let (head, mut bytes) = bytes.split_at(2);
        let rows = u16::from_ne_bytes(head.try_into().unwrap()) as usize;

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

            let mut cum = 0;
            let mut cum2q: [u16; 1 << RANS_BITS] = [0; 1 << RANS_BITS];
            let mut q2sym: HashMap<u16, rans64::RansDecSymbol> = HashMap::new();
            for i in 0..symbols_count {
                let symbol = pairs[2 * i as usize];
                let freq = pairs[2 * i as usize + 1] as u32;
                for c in cum..cum + freq {
                    cum2q[c as usize] = symbol;
                }
                q2sym.insert(symbol, rans64::RansDecSymbol::new(cum.into(), freq.into())?);
                cum += freq;
            }

            let (head, tail) = tail.split_at(2);
            let compressed_size = u16::from_ne_bytes(head.try_into().unwrap()) as usize;

            let (head, tail) = tail.split_at(compressed_size);
            let mut decoder = rans64::RansDecoder::new(head.to_owned())?;
            let mut t = vec![];
            for i in 0..(win_rows * cols) {
                let cum = decoder.get(RANS_BITS);
                let q = cum2q[cum as usize];
                let x = if q == 0 {
                    0.0
                } else {
                    let q = q + 1;
                    let sign = if (q & 1) == 1 { -1.0 } else { 1.0 };
                    let magnitude = (q >> 1) as f32;
                    scale * sign * magnitude
                };
                t.push(x);
                match decoder.advance(q2sym.get(&q).unwrap(), RANS_BITS) {
                    Ok(()) => {
                    }
                    Err(v) => {
                        warn!("RANS decoding failed at i={i}");
                        return Err(v.into());
                    }
                };
            }
            t.reverse();

            let rows = t.len() / cols;
            assert!(rows == win_rows);
            assert!(t.len() == rows * cols);

            let t = Tensor::from_vec(t, &[rows, cols], device)?
                .t()?
                .haar_inverse_tensor_cols()?
                .t()?
                .haar_inverse_tensor_cols()?
                .restore_columns(&idxs)?;
            ts.push(t);

            bytes = tail;
        }
        assert!(bytes.len() == 0);

        let t = Tensor::cat(&ts, 0)?.l2_normalize()?;
        Ok(t)
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

    fn l2_normalize(&self) -> Result<Tensor> {
        Ok(self.broadcast_div(&self.sqr()?.sum_keepdim(1)?.sqrt()?)?)
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
