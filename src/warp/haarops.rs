use candle_core::{IndexOp, Result, Tensor};

pub trait TensorHaarOps {
    fn haar_forward_tensor_cols(&self) -> Result<Tensor>;
    fn haar_inverse_tensor_cols(&self) -> Result<Tensor>;
}

impl TensorHaarOps for Tensor {
    fn haar_forward_tensor_cols(&self) -> Result<Tensor> {
        let (rows, cols) = self.dims2()?;
        let dev = self.device();
        let mut out_cols = Vec::with_capacity(cols);

        for c in 0..cols {
            let mut col: Vec<f32> = self.i((.., c))?.to_vec1::<f32>()?;
            haar_forward_mirror_edge(&mut col);
            out_cols.push(Tensor::from_vec(col, (rows,), dev)?);
        }

        Ok(Tensor::stack(&out_cols, 1)?)
    }

    fn haar_inverse_tensor_cols(&self) -> Result<Tensor> {
        let (rows, cols) = self.dims2()?;
        let dev = self.device();
        let mut out_cols = Vec::with_capacity(cols);

        for c in 0..cols {
            let mut col: Vec<f32> = self.i((.., c))?.to_vec1::<f32>()?;
            haar_inverse_mirror_edge(&mut col);
            out_cols.push(Tensor::from_vec(col, (rows,), dev)?);
        }

        Ok(Tensor::stack(&out_cols, 1)?)
    }
}

const INV_SQRT2: f32 = 0.7071067811865475;

#[inline(always)]
fn haar_forward_mirror_edge(x: &mut [f32]) {
    let n0 = x.len();
    let mut tmp = vec![0.0f32; n0];
    let mut n = n0;

    while n > 1 {
        let pairs = n / 2; // full (2i,2i+1) pairs
        let odd = (n & 1) as usize; // 1 if odd, 0 if even
        let n_coarse = pairs + odd; // ceil(n/2)

        // regular pairs
        for i in 0..pairs {
            let a = x[2 * i];
            let b = x[2 * i + 1];
            tmp[i] = (a + b) * INV_SQRT2;
            tmp[n_coarse + i] = (a - b) * INV_SQRT2;
        }

        // odd tail via mirror extension: pair last sample with itself → no detail
        if odd == 1 {
            let a = x[n - 1];
            tmp[pairs] = (a + a) * INV_SQRT2;
        }

        x[..n].copy_from_slice(&tmp[..n]);
        n = n_coarse;
    }
}

#[inline(always)]
fn haar_inverse_mirror_edge(x: &mut [f32]) {
    let n0 = x.len();
    let mut tmp = vec![0.0f32; n0];

    // rebuild level schedule (same sizes as forward)
    let mut levels = Vec::new();
    let mut n = n0;
    levels.push(n);
    while n > 1 {
        let odd = (n & 1) as usize;
        n = (n / 2) + odd; // ceil(n/2)
        levels.push(n);
    }

    for w in (1..levels.len()).rev() {
        let n_prev = levels[w - 1]; // size before this forward step
        let odd = (n_prev & 1) as usize;
        let pairs = n_prev / 2; // floor(n_prev/2)
        let n_coarse = pairs + odd; // ceil(n_prev/2)

        // regular pairs
        for i in 0..pairs {
            let sum = x[i];
            let diff = x[n_coarse + i];
            tmp[2 * i] = (sum + diff) * INV_SQRT2;
            tmp[2 * i + 1] = (sum - diff) * INV_SQRT2;
        }

        // odd tail from mirror extension: last detail was implicitly 0
        if odd == 1 {
            let sum_last = x[n_coarse - 1]; // extra coarse
            tmp[n_prev - 1] = sum_last * INV_SQRT2; // a = sum/√2
        }

        x[..n_prev].copy_from_slice(&tmp[..n_prev]);
    }
}

#[cfg(test)]
mod tests {
    use super::TensorHaarOps;
    use candle_core::{Device, Result, Tensor}; // your trait providing haar_forward_tensor_cols()

    const TOL: f32 = 1e-5;

    #[test]
    fn ones_1d_cols_only_nonzeros_in_row0() -> Result<()> {
        let dev = Device::Cpu;
        let m = 17usize; // odd height to exercise carry path
        let n = 16usize;

        let x = Tensor::from_vec(vec![1.0f32; m * n], (m, n), &dev)?;
        let y = x.haar_forward_tensor_cols()?; // 1D along columns only

        let mat = y.to_vec2::<f32>()?;
        // assert rows 1.. are ~0, row 0 has non-zeros
        for r in 1..m {
            for c in 0..n {
                assert!(
                    mat[r][c].abs() <= TOL,
                    "row {}, col {} not ~0: {}",
                    r,
                    c,
                    mat[r][c]
                );
            }
        }
        // ensure row 0 has some signal
        let mut any_nonzero_row0 = false;
        for c in 0..n {
            if mat[0][c].abs() > TOL {
                any_nonzero_row0 = true;
                break;
            }
        }
        assert!(
            any_nonzero_row0,
            "row 0 should contain the coarse coefficients"
        );
        Ok(())
    }

    #[test]
    fn ones_2d_cols_then_rows_only_00_nonzero() -> Result<()> {
        let dev = Device::Cpu;
        let m = 17usize; // odd height to exercise carry path
        let n = 16usize;

        let x = Tensor::from_vec(vec![1.0f32; m * n], (m, n), &dev)?;
        // 2D: columns, then rows via transpose-columns-transpose
        let y1 = x.haar_forward_tensor_cols()?;
        let y2 = y1
            .transpose(0, 1)?
            .haar_forward_tensor_cols()?
            .transpose(0, 1)?;

        let mat = y2.to_vec2::<f32>()?;

        // find max abs location
        let mut max_val = -f32::INFINITY;
        let mut argmax = (0usize, 0usize);
        for r in 0..m {
            for c in 0..n {
                let v = mat[r][c].abs();
                if v > max_val {
                    max_val = v;
                    argmax = (r, c);
                }
            }
        }
        assert_eq!(
            argmax,
            (0, 0),
            "max coefficient should be at (0,0), found {:?}",
            argmax
        );

        // assert only (0,0) is non-zero within tolerance
        for r in 0..m {
            for c in 0..n {
                if r == 0 && c == 0 {
                    continue;
                }
                assert!(
                    mat[r][c].abs() <= TOL,
                    "({},{}) not ~0: {}",
                    r,
                    c,
                    mat[r][c]
                );
            }
        }
        // sanity: (0,0) should be positive and significant
        assert!(
            mat[0][0] > TOL,
            "(0,0) should carry DC energy, got {}",
            mat[0][0]
        );
        Ok(())
    }
}
