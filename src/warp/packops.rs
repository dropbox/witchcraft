use anyhow::Result;
use candle_core::{Device, Tensor};

pub trait TensorPackOps {
    fn compand(&self) -> Result<Tensor>;
    fn inv_compand(&self) -> Result<Tensor>;
    fn quantize(&self, bits: u32) -> Result<Tensor>;
    fn dequantize(&self, bits: u32) -> Result<Tensor>;
    fn l2_normalize(&self) -> Result<Tensor>;
    fn stretch_rows(&self) -> Result<Tensor>;
    //fn from_q4_bytes(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn from_q8_bytes(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn from_companded_q4_bytes(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    //fn from_companded_q8_bytes(buffer: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn from_f32_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor>;
    fn to_q4_bytes(&self) -> Result<Vec<u8>>;
    fn to_q8_bytes(&self) -> Result<Vec<u8>>;
    fn to_f32_bytes(&self) -> Result<Vec<u8>>;
}

impl TensorPackOps for Tensor {
    /* mu-law companding to improve quantization of residuals. Scale input by 4 to expand
    to full [-1;1] range, as empirically residuals of normalized embeddings rarely exceed
    [-0.26:0.26] range.

    The inverse operation is really slow, we should use a table and fold that into the dot
    product calculations, like is done in the WARP paper.

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

    /*
    fn from_q4_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            let high = (byte >> 4) & 0x0f;
            let low = byte & 0x0f;
            out.push(high as f32);
            out.push(low as f32);
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

    fn from_q8_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(bytes.len());
        for &byte in bytes {
            out.push(byte as f32);
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

    fn to_q8_bytes(&self) -> Result<Vec<u8>> {
        let flat = self.flatten_all()?.to_vec1::<f32>()?;
        /*
        let mut hist: [u32; 256] = [0; 256];
        for i in &flat {
            hist[*i as usize] += 1;
        }
        println!("hist {:?}", hist);
        */

        let mut packed = Vec::with_capacity(flat.len());
        for i in &flat {
            packed.push(*i as u8);
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

    fn from_companded_q4_bytes(bytes: &[u8], cols: usize, device: &Device) -> Result<Tensor> {
        let x = Tensor::arange(0.0f32, 16.0f32, &Device::Cpu)?;
        let x = x.dequantize(4)?.inv_compand()?;
        let mut table: [f32; 16] = [0.0; 16];
        for i in 0..16 {
            table[i] = x.get(i)?.to_scalar()?;
        }

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

    fn stretch_rows(&self) -> Result<Tensor> {
        let device = self.device();
        let (m, n) = self.dims2()?;
        assert!(n == 128);

        let mut scaled_rows = Vec::with_capacity(m);

        for i in 0..m {
            let row = self.get(i)?; // shape: (128,)
            let v = row.to_vec1::<f32>()?;

            let mut max = std::f32::MIN;
            for x in &v {
                let a = (*x).abs();
                max = if a > max { a } else { max };
            }
            let range = max + 1e-6;
            let scale = 1.0 / range;

            let mut v2 = Vec::with_capacity(n);
            for i in 0..n {
                v2.push(scale * v[i]);
            }

            scaled_rows.push(Tensor::from_vec(v2, 128, &device)?);
        }

        Ok(Tensor::cat(&scaled_rows, 0)?)
    }
}

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
        assert!(mse < 0.01);
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
