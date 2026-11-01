//! When the Apple SME microkernels pay off — the per-call size gate.
//!
//! `sme::gemm_basic` is not a drop-in faster `neon::gemm_basic`. Each call pays a fixed
//! cost the NEON path does not have (streaming-mode entry/exit, `ZA` set-up and drain,
//! the 16x16-tile packing), and the 16x16 tile only repays it when the call has enough
//! `k` to amortise the fill and drain and enough `m*n` to fill the tile. Below that the
//! NEON path wins — by up to 2.6x on the shapes faer's QR and LU feed it, which is what
//! made the SME dispatch a regression on those factorizations even though the GEMM
//! throughput improved 3-7x.
//!
//! The numbers below are measured, not guessed. Method: `f64`/`f32`/`c64`/`c32` gemm at
//! one thread over the grid `(m, n) in {16x16, 16x64, 32x64, 64x64, 64x128, 128x128,
//! 128x256, 256x256} x k in {16, 32, 64, 128, 256}` (plus the complex shapes faer's
//! factorizations use), each shape timed for both kernels in the same binary with the
//! other kernel forced off, median of 5, batch of 50-100 calls per sample, on an M5 Max
//! (macOS 26.7.2). A shape counts as an SME win when `t_neon / t_sme >= 1`; the constants
//! are the smallest grid line that separates every such shape from every NEON win on
//! that grid. On the 40-shape grid that is 39/40 for `f64` and 39/40 for `f32`, and both
//! misses are parity, not losses: `f64` 16x64x256 goes to NEON where SME is 0.1% ahead,
//! `f32` 64x64x64 goes to NEON where SME is 3.3% ahead.
//!
//! Shape of the rule: a call needs all three of
//!
//! * `k >= min_k` — the depth the 16x16 tile needs to amortise fill/drain. At `k = 16`
//!   NEON wins on *every* `m*n` up to 256x256 (f64 ratio 0.40-0.52), at `k = 32` it still
//!   wins up to 256x256x32 (0.92).
//! * `m*n >= min_mn` — a tall-skinny call (`m*n < 2048`, e.g. faer's 16x64 blocks) never
//!   fills enough tiles: f64 16x64x128 is 0.82 in NEON's favour even at `k = 128`.
//! * `m*n*k >= min_work` — the total work the fixed cost has to disappear into. `f32`
//!   needs 4x what `f64` does (0.45-0.71 NEON ratios at equal `m*n*k` up to 262144,
//!   because the NEON f32 kernel is relatively stronger than the NEON f64 one), which is
//!   why the tables are per element type rather than one shared constant.
//!
//! The complex kernels do not fit that story: `c64`/`c32` SME is at parity with the
//! `neonfcma` path (ratios 0.99-1.19 c64, 0.99-1.06 c32 across the same grid — `neonfcma`
//! is already using the complex `FMLA` throughput that SME would bring). They take the
//! table of their real element type and so keep the gate that avoids the per-call fixed
//! cost; nothing is lost on the large shapes, where the two are equal.
//!
//! Measured at one thread on purpose: faer splits a matmul itself and calls `gemm::gemm`
//! with `Parallelism::None` once per tile, so what the dispatcher sees is always a
//! sequential call — the gate has to be right for a sequential call, and it is the same
//! shape at 1 thread and at 6.

/// Size gate for the Apple SME kernels of one element type.
#[derive(Clone, Copy, Debug)]
pub struct SmeThresholds {
    /// Minimum `k` for an SME call.
    pub min_k: usize,
    /// Minimum `m * n` for an SME call.
    pub min_mn: usize,
    /// Minimum `m * n * k` for an SME call.
    pub min_work: usize,
}

impl SmeThresholds {
    /// Whether a call of this shape should take the SME kernel.
    ///
    /// `saturating_mul` so that a huge `m * n * k` cannot wrap into a small product and
    /// send a giant call down the slower path.
    #[inline]
    pub fn wants_sme(&self, m: usize, n: usize, k: usize) -> bool {
        let mn = m.saturating_mul(n);
        k >= self.min_k && mn >= self.min_mn && mn.saturating_mul(k) >= self.min_work
    }
}

/// `f64`/`c64`: NEON wins below `k = 64` or below `m*n = 2048`; the crossover on the grid
/// is 32x64x64 (SME 1.13x) against 16x64x128 (NEON 1.22x).
pub const F64: SmeThresholds = SmeThresholds {
    min_k: 64,
    min_mn: 2048,
    min_work: 131_072,
};

/// `f32`/`c32`: same `k`/`m*n` floor, but the work floor is four times higher —
/// 64x64x64 is parity (1.03) for `f32` while 32x64x64 loses (0.66), and 64x64x128
/// (SME 1.54x) is the first clear win.
pub const F32: SmeThresholds = SmeThresholds {
    min_k: 64,
    min_mn: 2048,
    min_work: 524_288,
};

/// `c64`: the complex SME kernel is at parity with `neonfcma` from 64x64x64 up, so the
/// `f64` gate is the conservative choice — it only refuses the small calls, where the
/// per-call fixed cost is the whole difference.
pub const C64: SmeThresholds = F64;

/// `c32`: as [`C64`], with the `f32` work floor.
pub const C32: SmeThresholds = F32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_edges() {
        // The measured crossover shapes.
        assert!(!F64.wants_sme(16, 64, 128)); // NEON 1.22x
        assert!(F64.wants_sme(32, 64, 64)); // SME 1.13x
        assert!(!F64.wants_sme(256, 256, 32)); // NEON 1.09x
        assert!(F64.wants_sme(64, 128, 64)); // SME 1.35x
        assert!(!F32.wants_sme(64, 64, 64)); // parity
        assert!(F32.wants_sme(64, 64, 128)); // SME 1.54x
        assert!(F32.wants_sme(128, 128, 64)); // SME 1.23x
        // Large calls always take SME.
        assert!(F64.wants_sme(1024, 1024, 1024));
        assert!(F32.wants_sme(2048, 2048, 64));
        // No wrap-around on a huge product.
        assert!(F64.wants_sme(usize::MAX / 2, 3, 1024));
    }
}
