/// Row interleave factor for int32 accumulation.
/// 4 rows are interleaved to match vpmaddubsw instruction layout.
pub const ROW_INTERLEAVE: usize = 4;

/// Register block N dimension (columns per kernel invocation).
/// 8 because 8 * ROW_INTERLEAVE * sizeof(int8) = 32 bytes = 1 ymm register.
pub const NR: usize = 8;

/// Register block M dimension (max rows per kernel invocation).
pub const MR: usize = 12;

/// Cache block K dimension.
pub const KCB: usize = 512;

/// Cache block M dimension (multiple of MR).
pub const MCB: usize = 120;

/// A pre-packed int8 weight matrix in row-interleaved blocked format.
///
/// Layout: tiles of (KCB × NR), with ROW_INTERLEAVE interleaving within each tile.
/// Within a tile, for k-group g and column j:
///   `data[g * NR * RI + j * RI + ki] = B[k_start + g*4 + ki, n_start + j]`
pub struct PackedBMatrixI8 {
    nrow: usize,
    ncol: usize,
    num_k_blocks: usize,
    num_n_blocks: usize,
    col_offsets: Vec<i32>,
    data: Vec<i8>,
}

unsafe impl Send for PackedBMatrixI8 {}
unsafe impl Sync for PackedBMatrixI8 {}

impl PackedBMatrixI8 {
    /// Pack a K×N row-major int8 weight matrix.
    /// Also computes column offsets (sum of each column) for dequantization.
    pub fn new(k: usize, n: usize, src: &[i8]) -> Self {
        assert_eq!(src.len(), k * n, "src length must be k * n");

        let mut col_offsets = vec![0i32; n];
        for ki in 0..k {
            for j in 0..n {
                col_offsets[j] += src[ki * n + j] as i32;
            }
        }

        let num_k_blocks = (k + KCB - 1) / KCB;
        let num_n_blocks = (n + NR - 1) / NR;
        let tile_size = KCB * NR;
        let total_size = num_k_blocks * num_n_blocks * tile_size;
        let mut data = vec![0i8; total_size];

        for kb in 0..num_k_blocks {
            let k_start = kb * KCB;
            let kc = std::cmp::min(KCB, k - k_start);

            for nb in 0..num_n_blocks {
                let n_start = nb * NR;
                let nc = std::cmp::min(NR, n - n_start);
                let tile_offset = (kb * num_n_blocks + nb) * tile_size;

                for g in 0..((kc + ROW_INTERLEAVE - 1) / ROW_INTERLEAVE) {
                    let group_offset = tile_offset + g * NR * ROW_INTERLEAVE;
                    for j in 0..nc {
                        for ri in 0..ROW_INTERLEAVE {
                            let src_k = k_start + g * ROW_INTERLEAVE + ri;
                            if src_k < k {
                                data[group_offset + j * ROW_INTERLEAVE + ri] =
                                    src[src_k * n + n_start + j];
                            }
                        }
                    }
                }
            }
        }

        Self {
            nrow: k,
            ncol: n,
            num_k_blocks,
            num_n_blocks,
            col_offsets,
            data,
        }
    }

    pub fn k(&self) -> usize {
        self.nrow
    }
    pub fn n(&self) -> usize {
        self.ncol
    }
    pub fn col_offsets(&self) -> &[i32] {
        &self.col_offsets
    }

    pub(crate) fn tile(&self, kb: usize, nb: usize) -> &[i8] {
        let tile_size = KCB * NR;
        let offset = (kb * self.num_n_blocks + nb) * tile_size;
        &self.data[offset..offset + tile_size]
    }

    pub(crate) fn num_k_blocks(&self) -> usize {
        self.num_k_blocks
    }
    pub(crate) fn num_n_blocks(&self) -> usize {
        self.num_n_blocks
    }
}

/// Quantize float32 activations to uint8 with per-tensor affine quantization.
/// Returns (quantized_data, scale, zero_point, row_offsets).
pub fn quantize_a(src: &[f32], m: usize, k: usize) -> (Vec<u8>, f32, i32, Vec<i32>) {
    let mut a_u8 = Vec::new();
    let mut row_offsets = Vec::new();
    let (scale, zero_point) = quantize_a_into(src, m, k, &mut a_u8, &mut row_offsets);
    (a_u8, scale, zero_point, row_offsets)
}

/// Like [`quantize_a`] but reuses caller-provided buffers to avoid allocation.
/// Returns (scale, zero_point).
pub fn quantize_a_into(
    src: &[f32],
    m: usize,
    k: usize,
    a_u8: &mut Vec<u8>,
    row_offsets: &mut Vec<i32>,
) -> (f32, i32) {
    assert_eq!(src.len(), m * k);

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in src {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    a_u8.resize(m * k, 0);
    row_offsets.resize(m, 0);

    if max_val == min_val {
        a_u8.fill(0);
        row_offsets.fill(0);
        return (1.0, 0);
    }

    let scale = (max_val - min_val) / 255.0;
    let inv_scale = 1.0 / scale;
    let zero_point = ((-min_val * inv_scale).round() as i32).clamp(0, 255);

    for i in 0..m {
        let mut row_sum = 0i32;
        for ki in 0..k {
            let q = ((src[i * k + ki] * inv_scale).round() as i32 + zero_point).clamp(0, 255) as u8;
            a_u8[i * k + ki] = q;
            row_sum += q as i32;
        }
        row_offsets[i] = row_sum;
    }

    (scale, zero_point)
}
