//! Mixed real products with an f64 destination and f64 arithmetic (colmat-rs#293).

use gemm::dyn_stack::{MemBuffer, MemStack, StackReq};
use gemm::{
    gemm_mixed, gemm_mixed_scratch, gemm_mixed_with_stack, MixedRealGemmToF64, Parallelism,
};
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static COUNTED: Cell<bool> = const { Cell::new(false) };
}

fn counted() -> bool {
    COUNTED.try_with(|c| c.get()).unwrap_or(false)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        if counted() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
        if counted() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
        if counted() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

trait Scalar: Copy + Send + Sync + 'static {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Scalar for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Scalar for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    row_major: bool,
    pad: usize,
    reverse_rows: bool,
    reverse_cols: bool,
}

impl Layout {
    const COL: Self = Self {
        row_major: false,
        pad: 0,
        reverse_rows: false,
        reverse_cols: false,
    };
    const ROW: Self = Self {
        row_major: true,
        ..Self::COL
    };
    const REVERSED_PADDED: Self = Self {
        row_major: false,
        pad: 3,
        reverse_rows: true,
        reverse_cols: true,
    };

    fn strides(self, rows: usize, cols: usize) -> (usize, isize, isize, isize) {
        let (mut rs, mut cs, len) = if self.row_major {
            let ld = cols + self.pad;
            (ld as isize, 1, rows.max(1) * ld)
        } else {
            let ld = rows + self.pad;
            (1, ld as isize, cols.max(1) * ld)
        };
        let mut off = 0;
        if self.reverse_rows && rows > 0 {
            off += (rows - 1) as isize * rs;
            rs = -rs;
        }
        if self.reverse_cols && cols > 0 {
            off += (cols - 1) as isize * cs;
            cs = -cs;
        }
        (len.max(1), off, rs, cs)
    }
}

#[derive(Clone)]
struct View<T: Scalar> {
    buf: Vec<T>,
    valid: Vec<bool>,
    off: isize,
    rs: isize,
    cs: isize,
}

impl<T: Scalar> View<T> {
    fn new(rows: usize, cols: usize, layout: Layout, f: impl Fn(usize, usize) -> f64) -> Self {
        let (len, off, rs, cs) = layout.strides(rows, cols);
        let mut this = Self {
            buf: vec![T::from_f64(-777.0); len],
            valid: vec![false; len],
            off,
            rs,
            cs,
        };
        for j in 0..cols {
            for i in 0..rows {
                let index = this.physical_index(i, j);
                this.buf[index] = T::from_f64(f(i, j));
                this.valid[index] = true;
            }
        }
        this
    }

    fn physical_index(&self, i: usize, j: usize) -> usize {
        (self.off + i as isize * self.rs + j as isize * self.cs) as usize
    }

    fn get(&self, i: usize, j: usize) -> T {
        self.buf[self.physical_index(i, j)]
    }

    fn ptr(&self) -> *const T {
        self.buf.as_ptr().wrapping_offset(self.off)
    }

    fn ptr_mut(&mut self) -> *mut T {
        self.buf.as_mut_ptr().wrapping_offset(self.off)
    }

    fn assert_guards(&self, before: &Self) {
        for (index, (&after, &before)) in self.buf.iter().zip(&before.buf).enumerate() {
            if !self.valid[index] {
                assert_eq!(
                    after.to_f64().to_bits(),
                    before.to_f64().to_bits(),
                    "guard element {index} changed"
                );
            }
        }
    }
}

unsafe fn call<Lhs: Scalar, Rhs: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    dst: &mut View<f64>,
    read_dst: bool,
    lhs: &View<Lhs>,
    rhs: &View<Rhs>,
    alpha: f64,
    beta: f64,
    parallelism: Parallelism,
    stack: Option<&mut MemStack>,
) where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    unsafe {
        match stack {
            None => gemm_mixed(
                m,
                n,
                k,
                dst.ptr_mut(),
                dst.cs,
                dst.rs,
                read_dst,
                lhs.ptr(),
                lhs.cs,
                lhs.rs,
                rhs.ptr(),
                rhs.cs,
                rhs.rs,
                alpha,
                beta,
                parallelism,
            ),
            Some(stack) => gemm_mixed_with_stack(
                m,
                n,
                k,
                dst.ptr_mut(),
                dst.cs,
                dst.rs,
                read_dst,
                lhs.ptr(),
                lhs.cs,
                lhs.rs,
                rhs.ptr(),
                rhs.cs,
                rhs.rs,
                alpha,
                beta,
                parallelism,
                stack,
            ),
        }
    }
}

fn value(i: usize, j: usize, seed: usize) -> f64 {
    ((i * 7 + j * 11 + seed * 13) % 19) as f64 * 0.25 - 2.0
}

fn check_case<Lhs: Scalar, Rhs: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    layouts: (Layout, Layout, Layout),
    read_dst: bool,
    parallelism: Parallelism,
) where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    let (dst_layout, lhs_layout, rhs_layout) = layouts;
    let mut dst = View::<f64>::new(m, n, dst_layout, |i, j| {
        if read_dst {
            value(i, j, 3)
        } else {
            f64::NAN
        }
    });
    let before = dst.clone();
    let lhs = View::<Lhs>::new(m, k, lhs_layout, |i, j| value(i, j, 5));
    let rhs = View::<Rhs>::new(k, n, rhs_layout, |i, j| value(i, j, 7));
    let alpha = if read_dst { 0.75 } else { 0.0 };
    let beta = 1.25;

    unsafe {
        call(
            m,
            n,
            k,
            &mut dst,
            read_dst,
            &lhs,
            &rhs,
            alpha,
            beta,
            parallelism,
            None,
        )
    };

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            let mut scale = 0.0;
            for depth in 0..k {
                let a = lhs.get(i, depth).to_f64();
                let b = rhs.get(depth, j).to_f64();
                sum += a * b;
                scale += a.abs() * b.abs();
            }
            let mut want = beta * sum;
            scale *= beta.abs();
            if read_dst {
                let old = before.get(i, j);
                want += alpha * old;
                scale += alpha.abs() * old.abs();
            }
            let got = dst.get(i, j);
            let tolerance = 256.0 * k.max(1) as f64 * f64::EPSILON * scale.max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "{} x {} x {} at ({i}, {j}), layouts {layouts:?}, read_dst={read_dst}, {parallelism:?}: got={got:e}, want={want:e}, error={:e}, tolerance={tolerance:e}",
                m,
                n,
                k,
                (got - want).abs(),
            );
        }
    }
    dst.assert_guards(&before);
}

fn check_pair<Lhs: Scalar, Rhs: Scalar>(parallelism: Parallelism)
where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    let layouts = [
        (Layout::COL, Layout::COL, Layout::COL),
        (Layout::ROW, Layout::COL, Layout::ROW),
        (Layout::COL, Layout::ROW, Layout::COL),
        (
            Layout::REVERSED_PADDED,
            Layout::REVERSED_PADDED,
            Layout::REVERSED_PADDED,
        ),
    ];
    for &(m, n, k) in &[
        (0, 4, 3),
        (4, 0, 3),
        (1, 1, 0),
        (1, 1, 1),
        (3, 4, 2),
        (1, 4, 17),
        (4, 1, 17),
        (5, 7, 33),
        (7, 5, 1031),
    ] {
        for &layout in &layouts {
            for read_dst in [false, true] {
                check_case::<Lhs, Rhs>(m, n, k, layout, read_dst, parallelism);
            }
        }
    }
}

#[test]
fn f32_product_is_computed_after_promotion() {
    let a = [1.0_f32 + f32::EPSILON];
    let b = [a[0]];
    let mut c = [0.0_f64];
    unsafe {
        gemm_mixed(
            1,
            1,
            1,
            c.as_mut_ptr(),
            1,
            1,
            false,
            a.as_ptr(),
            1,
            1,
            b.as_ptr(),
            1,
            1,
            0.0,
            1.0,
            Parallelism::None,
        );
    }
    assert_eq!(c[0], f64::from(a[0]) * f64::from(a[0]));
}

#[test]
fn f32_f64_product_preserves_the_right_operand() {
    let a = [1.0_f32 + f32::EPSILON];
    let b = [1.0 + f64::EPSILON];
    let mut c = [0.0_f64];
    unsafe {
        gemm_mixed(
            1,
            1,
            1,
            c.as_mut_ptr(),
            1,
            1,
            false,
            a.as_ptr(),
            1,
            1,
            b.as_ptr(),
            1,
            1,
            0.0,
            1.0,
            Parallelism::None,
        );
    }
    assert_eq!(c[0], f64::from(a[0]) * b[0]);
}

#[test]
fn extreme_f32_values_are_converted_before_arithmetic() {
    for value in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX] {
        let lhs = [value];
        let rhs_f32 = [value];
        let rhs_f64 = [f64::from(value)];
        let expected = f64::from(value) * f64::from(value);
        assert!(expected.is_finite() && expected > 0.0);
        for (rhs, use_f64) in [
            (rhs_f32.as_ptr() as *const (), false),
            (rhs_f64.as_ptr() as *const (), true),
        ] {
            let mut dst = [0.0_f64];
            unsafe {
                if use_f64 {
                    gemm_mixed(
                        1,
                        1,
                        1,
                        dst.as_mut_ptr(),
                        1,
                        1,
                        false,
                        lhs.as_ptr(),
                        1,
                        1,
                        rhs as *const f64,
                        1,
                        1,
                        0.0,
                        1.0,
                        Parallelism::None,
                    );
                } else {
                    gemm_mixed(
                        1,
                        1,
                        1,
                        dst.as_mut_ptr(),
                        1,
                        1,
                        false,
                        lhs.as_ptr(),
                        1,
                        1,
                        rhs as *const f32,
                        1,
                        1,
                        0.0,
                        1.0,
                        Parallelism::None,
                    );
                }
            }
            assert_eq!(dst[0], expected, "f32 value {value:?}");
        }
    }
}

#[test]
fn mixed_f32_f32_matches_an_independent_reference() {
    check_pair::<f32, f32>(Parallelism::None);
}

#[test]
fn mixed_f32_f64_matches_an_independent_reference() {
    check_pair::<f32, f64>(Parallelism::None);
}

#[cfg(feature = "rayon")]
#[test]
fn mixed_pairs_match_on_rayon_workers() {
    check_pair::<f32, f32>(Parallelism::Rayon(2));
    check_pair::<f32, f64>(Parallelism::Rayon(2));
}

fn check_stack_pair<Lhs: Scalar, Rhs: Scalar>()
where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    let (m, n, k) = (7, 5, 1031);
    let layout = (Layout::ROW, Layout::COL, Layout::ROW);
    let lhs = View::<Lhs>::new(m, k, layout.1, |i, j| value(i, j, 5));
    let rhs = View::<Rhs>::new(k, n, layout.2, |i, j| value(i, j, 7));
    let mut plain = View::<f64>::new(m, n, layout.0, |i, j| value(i, j, 3));
    let mut stacked = plain.clone();
    unsafe {
        call(
            m,
            n,
            k,
            &mut plain,
            true,
            &lhs,
            &rhs,
            0.75,
            1.25,
            Parallelism::None,
            None,
        );
    }
    let req = gemm_mixed_scratch::<Lhs, Rhs>(m, n, k, Parallelism::None);
    assert!(req.size_bytes() > 16);
    let mut buf = MemBuffer::new(req);
    unsafe {
        call(
            m,
            n,
            k,
            &mut stacked,
            true,
            &lhs,
            &rhs,
            0.75,
            1.25,
            Parallelism::None,
            Some(MemStack::new(&mut buf)),
        );
    }
    for (&got, &want) in stacked.buf.iter().zip(&plain.buf) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    let mut short_dst = plain.clone();
    let before = short_dst.clone();
    let mut short = MemBuffer::new(StackReq::new::<u8>(16));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        call(
            m,
            n,
            k,
            &mut short_dst,
            true,
            &lhs,
            &rhs,
            0.75,
            1.25,
            Parallelism::None,
            Some(MemStack::new(&mut short)),
        );
    }));
    assert!(result.is_err(), "a short stack must panic");
    for (&after, &before) in short_dst.buf.iter().zip(&before.buf) {
        assert_eq!(after.to_bits(), before.to_bits(), "short stack wrote dst");
    }
}

#[test]
fn caller_stack_matches_and_a_short_stack_is_atomic() {
    check_stack_pair::<f32, f32>();
    check_stack_pair::<f32, f64>();
}

fn assert_first_call_allocates_nothing<Lhs: Scalar, Rhs: Scalar>()
where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    let (m, n, k) = (7, 5, 1031);
    let lhs = View::<Lhs>::new(m, k, Layout::COL, |i, j| value(i, j, 5));
    let rhs = View::<Rhs>::new(k, n, Layout::COL, |i, j| value(i, j, 7));
    let mut dst = View::<f64>::new(m, n, Layout::COL, |i, j| value(i, j, 3));
    let req = gemm_mixed_scratch::<Lhs, Rhs>(m, n, k, Parallelism::None);
    let mut buf = MemBuffer::new(req);
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let before = ALLOCS.load(Ordering::Relaxed);
                COUNTED.with(|c| c.set(true));
                unsafe {
                    call(
                        m,
                        n,
                        k,
                        &mut dst,
                        true,
                        &lhs,
                        &rhs,
                        0.75,
                        1.25,
                        Parallelism::None,
                        Some(MemStack::new(&mut buf)),
                    );
                }
                let allocations = ALLOCS.load(Ordering::Relaxed) - before;
                drop(std::hint::black_box(Box::new(0_u64)));
                let with_control = ALLOCS.load(Ordering::Relaxed) - before;
                COUNTED.with(|c| c.set(false));
                assert_eq!(with_control, allocations + 1, "allocator positive control");
                assert_eq!(allocations, 0, "first caller-stack call allocated");
            })
            .join()
            .unwrap();
    });
}

#[test]
fn first_caller_stack_call_allocates_nothing() {
    check_pair::<f32, f32>(Parallelism::None);
    check_pair::<f32, f64>(Parallelism::None);
    assert_first_call_allocates_nothing::<f32, f32>();
    assert_first_call_allocates_nothing::<f32, f64>();
}
