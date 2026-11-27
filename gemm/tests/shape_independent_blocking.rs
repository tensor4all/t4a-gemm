//! The blocking depth `kc` depends on `k` and the kernel only, never on the shape of the output
//! (tensor4all/colmat-next-rs#45). A caller that cuts a product into tiles and calls `gemm` once
//! per tile, as a thread team does, then gets bit for bit the result of one call on the whole
//! product: every element sums its `k` products in the same chunks either way.
//!
//! Before, a call with `m <= 64` and `n <= 64` cut `k` into chunks of 512 while a larger call cut
//! it into chunks of `kernel_params`'s `kc`, so for `k > 512` the tiles rounded differently.

use gemm::{c32, c64, gemm, Parallelism};

/// A small deterministic generator (SplitMix64), values in `[-1, 1)`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

trait Scalar: Copy + 'static {
    fn new(rng: &mut Rng) -> Self;
    fn one() -> Self;
    fn zero() -> Self;
    /// The bits of each component, so that the comparison is exact (and tells `-0.0` from `0.0`).
    fn bits(self) -> (u64, u64);
}
impl Scalar for f64 {
    fn new(rng: &mut Rng) -> Self {
        rng.next()
    }
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}
impl Scalar for f32 {
    fn new(rng: &mut Rng) -> Self {
        rng.next() as f32
    }
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits() as u64, 0)
    }
}
impl Scalar for c64 {
    fn new(rng: &mut Rng) -> Self {
        c64::new(rng.next(), rng.next())
    }
    fn one() -> Self {
        c64::new(1.0, 0.0)
    }
    fn zero() -> Self {
        c64::new(0.0, 0.0)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}
impl Scalar for c32 {
    fn new(rng: &mut Rng) -> Self {
        c32::new(rng.next() as f32, rng.next() as f32)
    }
    fn one() -> Self {
        c32::new(1.0, 0.0)
    }
    fn zero() -> Self {
        c32::new(0.0, 0.0)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits() as u64, self.im.to_bits() as u64)
    }
}

/// `dst = lhs * rhs` with every operand column-major, `lhs` with column stride `lhs_cs` and
/// `dst` with column stride `dst_cs`, so that a call can work on a window of larger matrices.
#[allow(clippy::too_many_arguments)]
fn product<T: Scalar>(
    (m, n, k): (usize, usize, usize),
    dst: &mut [T],
    dst_cs: usize,
    lhs: &[T],
    lhs_cs: usize,
    rhs: &[T],
    rhs_cs: usize,
) {
    unsafe {
        gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            dst_cs as isize,
            1,
            false,
            lhs.as_ptr(),
            lhs_cs as isize,
            1,
            rhs.as_ptr(),
            rhs_cs as isize,
            1,
            T::zero(),
            T::one(),
            false,
            false,
            false,
            Parallelism::None,
        );
    }
}

/// The window `rows x cols` at `(i0, j0)` of the `m x n x k` product, computed by one call on
/// that window alone, is bit for bit the same window of the whole product.
fn window_matches_whole<T: Scalar>(
    seed: u64,
    (m, n, k): (usize, usize, usize),
    (i0, rows): (usize, usize),
    (j0, cols): (usize, usize),
) {
    let mut rng = Rng(seed);
    let lhs: Vec<T> = (0..m * k).map(|_| T::new(&mut rng)).collect();
    let rhs: Vec<T> = (0..k * n).map(|_| T::new(&mut rng)).collect();
    let mut whole = vec![T::zero(); m * n];
    product((m, n, k), &mut whole, m, &lhs, m, &rhs, k);
    let mut window = vec![T::zero(); rows * cols];
    product(
        (rows, cols, k),
        &mut window,
        rows,
        &lhs[i0..],
        m,
        &rhs[j0 * k..],
        k,
    );
    for j in 0..cols {
        for i in 0..rows {
            let (got, want) = (window[i + rows * j], whole[(i0 + i) + m * (j0 + j)]);
            assert_eq!(
                got.bits(),
                want.bits(),
                "{} {m}x{n}x{k}: window ({rows}x{cols} at {i0},{j0}) element ({i}, {j}) \
                 differs from the whole product",
                core::any::type_name::<T>(),
            );
        }
    }
}

fn all_windows<T: Scalar>(seed: u64) {
    // a 40-row slice of a tall product (the window takes the small-size branch, the whole
    // product does not), the case of tensor4all/colmat-next-rs#45
    window_matches_whole::<T>(seed, (257, 3, 513), (100, 40), (0, 3));
    // a 40 x 40 block of a square one, at two depths past 512
    window_matches_whole::<T>(seed, (130, 130, 1024), (64, 40), (32, 40));
    window_matches_whole::<T>(seed, (130, 130, 513), (0, 40), (90, 40));
    // a 40-column slice of a wide product
    window_matches_whole::<T>(seed, (3, 257, 700), (0, 3), (200, 40));
}

#[test]
fn blocking_is_independent_of_the_output_shape_f64() {
    all_windows::<f64>(1);
}

#[test]
fn blocking_is_independent_of_the_output_shape_f32() {
    all_windows::<f32>(2);
}

#[test]
fn blocking_is_independent_of_the_output_shape_c64() {
    all_windows::<c64>(3);
}

#[test]
fn blocking_is_independent_of_the_output_shape_c32() {
    all_windows::<c32>(4);
}
