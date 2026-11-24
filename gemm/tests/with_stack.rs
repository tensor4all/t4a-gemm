//! `gemm_with_stack` and `gemm_scratch` (#21): with the stack `gemm_scratch` asks for, a call
//! allocates nothing on a fresh thread, stays within that stack, and gives bit-for-bit the
//! result of `gemm`.
//!
//! A counting global allocator counts the allocations of the threads that mark themselves. The
//! process-wide one-time set-up (the cache sizes, the kernel selection, the SME probe) is done
//! once on the main thread first: it is not per-thread, so it is not what this test is about.

use dyn_stack::{MemBuffer, MemStack};
use gemm::{c32, c64, gemm, gemm_scratch, gemm_with_stack, Parallelism};
use std::alloc::{GlobalAlloc, Layout, System};
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
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if counted() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if counted() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if counted() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The tests count on one shared counter: one at a time.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A small deterministic generator (SplitMix64).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

trait Scalar: Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static {
    const COMPLEX: bool;
    fn sample(rng: &mut Rng) -> Self;
    /// The bits of the value, so that NaNs and signed zeros compare exactly.
    fn bits(self) -> (u64, u64);
}
impl Scalar for f64 {
    const COMPLEX: bool = false;
    fn sample(rng: &mut Rng) -> Self {
        rng.unit()
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}
impl Scalar for f32 {
    const COMPLEX: bool = false;
    fn sample(rng: &mut Rng) -> Self {
        rng.unit() as f32
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits() as u64, 0)
    }
}
impl Scalar for c64 {
    const COMPLEX: bool = true;
    fn sample(rng: &mut Rng) -> Self {
        c64::new(rng.unit(), rng.unit())
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}
impl Scalar for c32 {
    const COMPLEX: bool = true;
    fn sample(rng: &mut Rng) -> Self {
        c32::new(rng.unit() as f32, rng.unit() as f32)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits() as u64, self.im.to_bits() as u64)
    }
}

/// A strided `rows x cols` matrix in a buffer: column-major, row-major, or column-major with
/// padded columns, each possibly with reversed rows or columns.
#[derive(Clone, Copy, Debug)]
struct Layout2 {
    row_major: bool,
    pad: usize,
    rev_rows: bool,
    rev_cols: bool,
}

impl Layout2 {
    fn sample(rng: &mut Rng) -> Self {
        Self {
            row_major: rng.below(2) == 0,
            pad: [0, 0, 3][rng.below(3)],
            rev_rows: rng.below(4) == 0,
            rev_cols: rng.below(4) == 0,
        }
    }
    const PLAIN: Self = Self {
        row_major: false,
        pad: 0,
        rev_rows: false,
        rev_cols: false,
    };
    /// (buffer length, offset of element (0, 0), row stride, column stride)
    fn strides(self, rows: usize, cols: usize) -> (usize, isize, isize, isize) {
        let (mut rs, mut cs, len) = if self.row_major {
            let ld = cols + self.pad;
            (ld as isize, 1, rows.max(1) * ld)
        } else {
            let ld = rows + self.pad;
            (1, ld as isize, cols.max(1) * ld)
        };
        let mut off = 0isize;
        if self.rev_rows && rows > 0 {
            off += (rows - 1) as isize * rs;
            rs = -rs;
        }
        if self.rev_cols && cols > 0 {
            off += (cols - 1) as isize * cs;
            cs = -cs;
        }
        (len.max(1), off, rs, cs)
    }
}

struct Mat<T> {
    buf: Vec<T>,
    off: isize,
    rs: isize,
    cs: isize,
}

impl<T: Scalar> Mat<T> {
    fn new(rng: &mut Rng, rows: usize, cols: usize, layout: Layout2) -> Self {
        let (len, off, rs, cs) = layout.strides(rows, cols);
        Self {
            buf: (0..len).map(|_| T::sample(rng)).collect(),
            off,
            rs,
            cs,
        }
    }
    fn ptr(&self) -> *const T {
        self.buf.as_ptr().wrapping_offset(self.off)
    }
    fn ptr_mut(&mut self) -> *mut T {
        self.buf.as_mut_ptr().wrapping_offset(self.off)
    }
}

#[derive(Clone, Copy, Debug)]
struct Call {
    m: usize,
    n: usize,
    k: usize,
    dst: Layout2,
    lhs: Layout2,
    rhs: Layout2,
    read_dst: bool,
    conj: (bool, bool, bool),
}

/// Runs `call` with `gemm` (`stack` `None`) or `gemm_with_stack`, and returns `dst`'s buffer.
fn run<T: Scalar>(call: Call, seed: u64, stack: Option<&mut MemStack>) -> Vec<T> {
    let mut rng = Rng(seed);
    let Call { m, n, k, .. } = call;
    let mut dst = Mat::<T>::new(&mut rng, m, n, call.dst);
    let lhs = Mat::<T>::new(&mut rng, m, k, call.lhs);
    let rhs = Mat::<T>::new(&mut rng, k, n, call.rhs);
    let (alpha, beta) = (T::sample(&mut rng), T::sample(&mut rng));
    let (cd, cl, cr) = if T::COMPLEX {
        call.conj
    } else {
        (false, false, false)
    };
    unsafe {
        let args = (
            dst.ptr_mut(),
            dst.cs,
            dst.rs,
            lhs.ptr(),
            lhs.cs,
            lhs.rs,
            rhs.ptr(),
            rhs.cs,
            rhs.rs,
        );
        match stack {
            None => gemm(
                m,
                n,
                k,
                args.0,
                args.1,
                args.2,
                call.read_dst,
                args.3,
                args.4,
                args.5,
                args.6,
                args.7,
                args.8,
                alpha,
                beta,
                cd,
                cl,
                cr,
                Parallelism::None,
            ),
            Some(stack) => gemm_with_stack(
                m,
                n,
                k,
                args.0,
                args.1,
                args.2,
                call.read_dst,
                args.3,
                args.4,
                args.5,
                args.6,
                args.7,
                args.8,
                alpha,
                beta,
                cd,
                cl,
                cr,
                Parallelism::None,
                stack,
            ),
        }
    }
    dst.buf
}

/// Shapes on both sides of the SME gate (`sme_thresholds`: `16 x 64 x 128` stays on NEON,
/// `64 x 128 x 64` takes SME on hosts with it), of the small-product blocking (`m, n <= 64`),
/// of the lhs prepacking (`m` not a multiple of the register) and of the L2 size (checked in
/// `first_call_on_a_fresh_thread_allocates_nothing`), plus the early paths (`k <= 2`, vectors).
const SHAPES: &[(usize, usize, usize)] = &[
    (0, 5, 7),
    (5, 7, 0),
    (3, 4, 2),
    (1, 300, 200),
    (300, 1, 200),
    (16, 64, 128),
    (64, 128, 64),
    (37, 513, 300),
    (300, 200, 150),
    (513, 37, 700),
    (1024, 1024, 1024),
    (1500, 2000, 700),
];

fn fresh_thread_allocations<T: Scalar>(call: Call) -> usize {
    let req = gemm_scratch::<T>(call.m, call.n, call.k, Parallelism::None);
    let mut buf = MemBuffer::new(req);
    let mut rng = Rng(7);
    let Call { m, n, k, .. } = call;
    let mut dst = Mat::<T>::new(&mut rng, m, n, call.dst);
    let lhs = Mat::<T>::new(&mut rng, m, k, call.lhs);
    let rhs = Mat::<T>::new(&mut rng, k, n, call.rhs);
    let (alpha, beta) = (T::sample(&mut rng), T::sample(&mut rng));
    std::thread::scope(|s| {
        s.spawn(|| {
            let stack = MemStack::new(&mut buf);
            let before = ALLOCS.load(Ordering::Relaxed);
            COUNTED.with(|c| c.set(true));
            unsafe {
                gemm_with_stack(
                    m,
                    n,
                    k,
                    dst.ptr_mut(),
                    dst.cs,
                    dst.rs,
                    call.read_dst,
                    lhs.ptr(),
                    lhs.cs,
                    lhs.rs,
                    rhs.ptr(),
                    rhs.cs,
                    rhs.rs,
                    alpha,
                    beta,
                    false,
                    false,
                    false,
                    Parallelism::None,
                    stack,
                )
            };
            COUNTED.with(|c| c.set(false));
            ALLOCS.load(Ordering::Relaxed) - before
        })
        .join()
        .unwrap()
    })
}

fn first_call<T: Scalar>() {
    let l2 = gemm_common::cache::CACHE_INFO[1].cache_bytes;
    let (mut below_l2, mut above_l2) = (false, false);
    for &(m, n, k) in SHAPES {
        let req = gemm_scratch::<T>(m, n, k, Parallelism::None);
        if req.size_bytes() > l2 {
            above_l2 = true;
        } else if req.size_bytes() > 0 {
            below_l2 = true;
        }
        for (lhs_row_major, dst_row_major) in [(false, false), (true, false), (false, true)] {
            let call = Call {
                m,
                n,
                k,
                dst: Layout2 {
                    row_major: dst_row_major,
                    ..Layout2::PLAIN
                },
                lhs: Layout2 {
                    row_major: lhs_row_major,
                    ..Layout2::PLAIN
                },
                rhs: Layout2::PLAIN,
                read_dst: true,
                conj: (false, false, false),
            };
            let n_alloc = fresh_thread_allocations::<T>(call);
            assert_eq!(
                n_alloc,
                0,
                "{} {m}x{n}x{k} lhs row-major {lhs_row_major} dst row-major {dst_row_major}",
                core::any::type_name::<T>()
            );
        }
    }
    assert!(
        below_l2 && above_l2,
        "the shapes cover both sides of the L2 size ({l2} bytes)"
    );
}

/// The process-wide set-up, once, before any thread is counted.
fn warm_up_process() {
    let mut buf = MemBuffer::new(gemm_scratch::<f64>(64, 128, 64, Parallelism::None));
    let call = Call {
        m: 64,
        n: 128,
        k: 64,
        dst: Layout2::PLAIN,
        lhs: Layout2::PLAIN,
        rhs: Layout2::PLAIN,
        read_dst: false,
        conj: (false, false, false),
    };
    let _ = run::<f64>(call, 1, Some(MemStack::new(&mut buf)));
    let _ = run::<f32>(call, 1, None);
    let _ = run::<c64>(call, 1, None);
    let _ = run::<c32>(call, 1, None);
}

#[test]
fn first_call_on_a_fresh_thread_allocates_nothing() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    warm_up_process();
    first_call::<f64>();
    first_call::<f32>();
    first_call::<c64>();
    first_call::<c32>();
}

/// For random shapes and layouts: `gemm_with_stack` on a stack of exactly `gemm_scratch`
/// (a larger request would panic in `make_aligned_uninit`) gives bit-for-bit `gemm`'s result.
fn matches_gemm<T: Scalar>(seed: u64) {
    let mut rng = Rng(seed);
    let mut calls: Vec<Call> = Vec::new();
    for &(m, n, k) in SHAPES {
        calls.push(Call {
            m,
            n,
            k,
            dst: Layout2::sample(&mut rng),
            lhs: Layout2::sample(&mut rng),
            rhs: Layout2::sample(&mut rng),
            read_dst: rng.below(2) == 0,
            conj: (rng.below(2) == 0, rng.below(2) == 0, rng.below(2) == 0),
        });
    }
    for _ in 0..60 {
        let dim = |rng: &mut Rng| [1, 2, 3, 7, 16, 33, 64, 65, 129, 300][rng.below(10)];
        calls.push(Call {
            m: dim(&mut rng),
            n: dim(&mut rng),
            k: dim(&mut rng),
            dst: Layout2::sample(&mut rng),
            lhs: Layout2::sample(&mut rng),
            rhs: Layout2::sample(&mut rng),
            read_dst: rng.below(2) == 0,
            conj: (rng.below(2) == 0, rng.below(2) == 0, rng.below(2) == 0),
        });
    }
    for (i, call) in calls.into_iter().enumerate() {
        let seed = 1000 + i as u64;
        let want = run::<T>(call, seed, None);
        let mut buf = MemBuffer::new(gemm_scratch::<T>(call.m, call.n, call.k, Parallelism::None));
        let got = run::<T>(call, seed, Some(MemStack::new(&mut buf)));
        let same = want.iter().zip(&got).all(|(a, b)| a.bits() == b.bits());
        assert!(same, "{} {call:?}", core::any::type_name::<T>());
    }
}

#[test]
fn matches_gemm_bit_for_bit_within_the_bound() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    matches_gemm::<f64>(1);
    matches_gemm::<f32>(2);
    matches_gemm::<c64>(3);
    matches_gemm::<c32>(4);
}

#[test]
#[should_panic(expected = "cannot hold gemm_scratch")]
fn a_short_stack_panics() {
    let mut buf = MemBuffer::new(dyn_stack::StackReq::new::<u8>(16));
    let call = Call {
        m: 300,
        n: 200,
        k: 150,
        dst: Layout2::PLAIN,
        lhs: Layout2::PLAIN,
        rhs: Layout2::PLAIN,
        read_dst: false,
        conj: (false, false, false),
    };
    let _ = run::<f64>(call, 1, Some(MemStack::new(&mut buf)));
}
