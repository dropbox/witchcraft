/*
This is a straight port of rans64.h from ryg-rans, which is in public domain.

The port was done by ChatGPT, and aims to keep the interface from Cargo's
"rans" package, but without adding a large amount of external dependencies
and without needing any unsafe code. Failure handling during encoding is like
in the original code, with assert/panic instead of failure propagation, as
failures would be result of implementation errors. The decoder adds
Rust error handling, to avoid panics() as a result of garbled inputs.

Please refer to rans64.h in https://github.com/rygorous/ryg_rans for
the original code and license.

*/

// -----------------------------------------------------------------------------
// Safe single-stream rANS64 codec (bitstream-compatible with ryg's rans64.h)
// -----------------------------------------------------------------------------

/// Normalization lower bound (same as `#define RANS64_L (1ull << 31)` in rans64.h)
pub const RANS64_L: u64 = 1u64 << 31;

// -----------------------------------------------------------------------------
// Encoder symbol (RansEncSymbol) - fast path parameters
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RansEncSymbol {
    x_max: u64,
    rcp_freq: u64,
    bias: u64,
    cmpl_freq: u32,
    rcp_shift: u16,
}

impl RansEncSymbol {
    /// Initialize an encoder symbol from cumulative frequency `cum_freq` and
    /// symbol frequency `freq`, with the total normalized to `1 << scale_bits`.
    ///
    /// This matches the logic in ryg's rans64.h (fast encoder).
    pub fn new(cum_freq: u32, freq: u32, scale_bits: u32) -> Self {
        assert!(scale_bits <= 31);
        assert!(cum_freq <= (1u32 << scale_bits));
        assert!(freq <= (1u32 << scale_bits) - cum_freq);

        let x_max = ((RANS64_L >> scale_bits) << 32) * (freq as u64);
        let cmpl_freq = ((1u32 << scale_bits) - freq) as u32;

        if freq < 2 {
            // Same special-case logic as in ryg's rANS for freq=0/1.
            let rcp_freq = !0u64;
            let rcp_shift = 0;
            let bias = cum_freq as u64 + ((1u64 << scale_bits) - 1);
            Self {
                x_max,
                rcp_freq,
                bias,
                cmpl_freq,
                rcp_shift,
            }
        } else {
            // Alverson "Integer Division using reciprocals" style:
            // shift = ceil(log2(freq))
            let mut shift: u32 = 0;
            while freq > (1u32 << shift) {
                shift += 1;
            }

            // rcp_freq ≈ 2^(shift+63) / freq, in 64-bit fixed point.
            let rcp_freq = (((1u128 << (shift + 63)) + (freq as u128) - 1) / (freq as u128)) as u64;
            let rcp_shift = (shift - 1) as u16;
            let bias = cum_freq as u64;

            Self {
                x_max,
                rcp_freq,
                bias,
                cmpl_freq,
                rcp_shift,
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Encoder (RansEncoder) - writes backwards into a Vec<u8>
// -----------------------------------------------------------------------------

#[derive(Debug)]
pub struct RansEncoder {
    state: u64,
    buf: Vec<u8>,
    idx: usize, // write-backwards index; [0..idx) unused, [idx..] is encoded
}

impl RansEncoder {
    /// Create an encoder with space for up to `max_len` bytes.
    pub fn new(max_len: usize) -> Self {
        Self {
            state: RANS64_L,
            buf: vec![0; max_len],
            idx: max_len,
        }
    }

    /// Internal renormalization (matches RansEncRenorm in rans64.h).
    ///
    /// Emits 32-bit chunks (4 bytes, little-endian) while state >= x_max,
    /// then shrinks state by 32 bits each step.
    #[inline]
    fn renorm(&mut self, sym: &RansEncSymbol) {
        let x_max = sym.x_max;
        let x = &mut self.state;

        while *x >= x_max {
            if self.idx < 4 {
                panic!("RansEncoder overflow during renorm");
            }
            self.idx -= 4;

            let v = *x as u32;
            self.buf[self.idx + 0] = (v >> 0) as u8;
            self.buf[self.idx + 1] = (v >> 8) as u8;
            self.buf[self.idx + 2] = (v >> 16) as u8;
            self.buf[self.idx + 3] = (v >> 24) as u8;

            *x >>= 32;
        }
    }

    /// Encode one symbol with the given precomputed parameters.
    ///
    /// This is the fast "put symbol" path, matching the pattern in rans64.h.
    #[inline(always)]
    pub fn put(&mut self, sym: &RansEncSymbol) {
        if sym.x_max == 0 {
            panic!("RansEncoder::put: invalid symbol (x_max = 0)");
        }

        self.renorm(sym);

        let x = &mut self.state;

        // q = mul_hi(x, rcp_freq) >> rcp_shift
        let q = (((*x as u128) * (sym.rcp_freq as u128)) >> 64) >> sym.rcp_shift;
        let q = q as u64;

        *x = x
            .wrapping_add(sym.bias)
            .wrapping_add(q * (sym.cmpl_freq as u64));
    }

    /// Flush final state: write 8 bytes (little-endian) at the front.
    pub fn flush(&mut self) {
        let x = self.state;

        if self.idx < 8 {
            panic!("RansEncoder::flush: buffer overflow");
        }
        self.idx -= 8;

        for i in 0..8 {
            self.buf[self.idx + i] = (x >> (i * 8)) as u8;
        }
    }

    /// Get the encoded data slice.
    pub fn data(&self) -> &[u8] {
        &self.buf[self.idx..]
    }
}

// -----------------------------------------------------------------------------
// Decoder (RansDecoder) - reads forwards from a Vec<u8>
// -----------------------------------------------------------------------------
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum RansDecodeError {
    /// Symbol start or freq out of range
    BadSymbol,
    /// Not enough bytes in the input to complete decoding.
    Underflow,
    /// Arithmetic overflow/underflow or otherwise impossible state.
    /// Indicates corrupted or invalid rANS stream.
    Corrupt,
}

impl fmt::Display for RansDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RansDecodeError::BadSymbol => write!(f, "rANS decode bad symbol"),
            RansDecodeError::Underflow => write!(f, "rANS decode underflow"),
            RansDecodeError::Corrupt   => write!(f, "rANS corrupt stream"),
        }
    }
}

impl Error for RansDecodeError {}

#[derive(Debug)]
pub struct RansDecoder {
    state: u64,
    buf: Vec<u8>,
    idx: usize,
}

// -----------------------------------------------------------------------------
// Decoder symbol (RansDecSymbol)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RansDecSymbol {
    start: u32,
    freq: u32,
}

impl RansDecSymbol {
    pub fn new(start: u32, freq: u32) -> Result<Self, RansDecodeError> {
        // mirror the constraints in the C code
        if start > (1u32 << 31) || freq > (1u32 << 31) - start {
            return Err(RansDecodeError::BadSymbol);
        }
        Ok(Self { start, freq })
    }

    #[inline]
    fn start(&self) -> u64 {
        self.start as u64
    }

    #[inline]
    fn freq(&self) -> u64 {
        self.freq as u64
    }
}

impl RansDecoder {
    /// Create a decoder from an encoded buffer.
    /// Returns Err if the buffer is too short to contain the initial state.
    pub fn new(buf: Vec<u8>) -> Result<Self, RansDecodeError> {
        if buf.len() < 8 {
            return Err(RansDecodeError::Underflow);
        }
        let mut dec = Self { state: 0, buf, idx: 0 };
        dec.init_state()?;
        Ok(dec)
    }

    /// Initialize rANS64 state from first 8 bytes (little-endian).
    fn init_state(&mut self) -> Result<(), RansDecodeError> {
        if self.idx + 8 > self.buf.len() {
            return Err(RansDecodeError::Underflow);
        }
        let mut x = 0u64;
        for i in 0..8 {
            x |= (self.buf[self.idx + i] as u64) << (i * 8);
        }
        self.state = x;
        self.idx += 8;
        Ok(())
    }

    /// Get current cumulative frequency (lower `scale_bits` bits).
    /// Pure access, never fails.
    #[inline(always)]
    pub fn get(&self, scale_bits: u32) -> u32 {
        let mask = (1u64 << scale_bits) - 1;
        (self.state & mask) as u32
    }

    /// Advance by one symbol, returning an error instead of panicking on bad data.
    #[inline]
    pub fn advance(
        &mut self,
        sym: &RansDecSymbol,
        scale_bits: u32,
    ) -> Result<(), RansDecodeError> {
        let mask = (1u64 << scale_bits) - 1;

        // s, x = D(x)  (wrapping semantics, like the C code)
        let mut x = self.state;
        x = sym.freq() * (x >> scale_bits) + (x & mask) - sym.start();

        // Renormalize: at most one 32-bit word, like Rans64DecAdvance in C.
        if x < RANS64_L {
            if self.idx + 4 > self.buf.len() {
                return Err(RansDecodeError::Underflow);
            }

            let b0 = self.buf[self.idx + 0] as u64;
            let b1 = self.buf[self.idx + 1] as u64;
            let b2 = self.buf[self.idx + 2] as u64;
            let b3 = self.buf[self.idx + 3] as u64;
            self.idx += 4;

            let v = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            x = (x << 32) | v;

            // Matches Rans64Assert(x >= RANS64_L)
            if x < RANS64_L {
                return Err(RansDecodeError::Corrupt);
            }
        }
        self.state = x;
        Ok(())
    }
}
