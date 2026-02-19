const INV_SQRT2: f32 = 0.7071067811865475;

#[inline(always)]
pub(crate) fn haar_forward_mirror_edge(x: &mut [f32]) {
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
pub(crate) fn haar_inverse_mirror_edge(x: &mut [f32]) {
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
