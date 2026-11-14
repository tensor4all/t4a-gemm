//! Allocation count of the generic GEMM path once it is warm.
//!
//! tenferro-rs#2472 / #2448 measured two heap allocations per call on aarch64 for small
//! factorization-sized products (`MemBuffer::new` for the packed panels and the
//! `did_pack_lhs` flag vector). Once a thread has run a shape, running it again must not
//! touch the allocator.
//!
//! The counter is process-wide, so every check lives in one `#[test]` to keep the default
//! parallel test harness from mixing its own allocations into the count.

// Without `std` the generic path allocates its buffers per call by design.
#![cfg(feature = "std")]

use gemm_common::Parallelism;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct Case {
    m: usize,
    n: usize,
    k: usize,
    /// `false` stores `lhs` row-major (`lhs_rs = k`), which takes the prepacked-lhs branch.
    lhs_colmajor: bool,
    dst_colmajor: bool,
    /// `false` stores `rhs` row-major. A column-major `rhs` is what makes the generic path pack
    /// the RHS on every target (`do_pack_rhs`), so the shapes below use it.
    rhs_colmajor: bool,
}

struct Operands {
    lhs: Vec<f64>,
    rhs: Vec<f64>,
    dst: Vec<f64>,
}

fn operands(c: &Case) -> Operands {
    Operands {
        lhs: (0..c.m * c.k)
            .map(|i| (i % 17) as f64 / 8.0 - 1.0)
            .collect(),
        rhs: (0..c.k * c.n)
            .map(|i| (i % 13) as f64 / 8.0 - 0.5)
            .collect(),
        dst: vec![0.0; c.m * c.n],
    }
}

/// Runs one product and returns how many allocations it made, and how many bytes they were.
fn count(c: &Case, ops: &mut Operands, parallelism: Parallelism) -> (usize, usize) {
    let (lhs_cs, lhs_rs) = if c.lhs_colmajor {
        (c.m as isize, 1)
    } else {
        (1, c.k as isize)
    };
    let (dst_cs, dst_rs) = if c.dst_colmajor {
        (c.m as isize, 1)
    } else {
        (1, c.n as isize)
    };
    let (rhs_cs, rhs_rs) = if c.rhs_colmajor {
        (1, c.k as isize)
    } else {
        (c.k as isize, 1)
    };
    let gemm = gemm_f64::gemm::f64::get_gemm_fn();
    let before = ALLOCS.load(Ordering::Relaxed);
    let before_bytes = BYTES.load(Ordering::Relaxed);
    unsafe {
        gemm(
            c.m,
            c.n,
            c.k,
            ops.dst.as_mut_ptr(),
            dst_cs,
            dst_rs,
            true,
            ops.lhs.as_ptr(),
            lhs_cs,
            lhs_rs,
            ops.rhs.as_ptr(),
            rhs_cs,
            rhs_rs,
            1.0,
            1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
    (
        ALLOCS.load(Ordering::Relaxed) - before,
        BYTES.load(Ordering::Relaxed) - before_bytes,
    )
}

#[test]
fn warm_generic_gemm_does_not_allocate() {
    // 48 is the size tenferro measured (`faer/{svd,eig,qr}/48`); the rest are the panel
    // shapes a blocked factorization of that size issues, plus both packing branches.
    let cases = [
        Case {
            m: 48,
            n: 48,
            k: 48,
            lhs_colmajor: true,
            dst_colmajor: true,
            rhs_colmajor: false,
        },
        Case {
            m: 48,
            n: 48,
            k: 48,
            lhs_colmajor: true,
            dst_colmajor: false,
            rhs_colmajor: false,
        },
        Case {
            m: 48,
            n: 48,
            k: 48,
            lhs_colmajor: false,
            dst_colmajor: true,
            rhs_colmajor: false,
        },
        Case {
            m: 48,
            n: 32,
            k: 16,
            lhs_colmajor: true,
            dst_colmajor: true,
            rhs_colmajor: false,
        },
        Case {
            m: 32,
            n: 48,
            k: 16,
            lhs_colmajor: true,
            dst_colmajor: true,
            rhs_colmajor: false,
        },
        Case {
            m: 47,
            n: 33,
            k: 31,
            lhs_colmajor: true,
            dst_colmajor: true,
            rhs_colmajor: false,
        },
        Case {
            m: 96,
            n: 96,
            k: 96,
            lhs_colmajor: true,
            dst_colmajor: true,
            rhs_colmajor: false,
        },
    ];
    let mut parallelisms = vec![Parallelism::None];
    #[cfg(feature = "rayon")]
    parallelisms.push(Parallelism::Rayon(1));

    let mut report = Vec::new();
    for &par in &parallelisms {
        for c in &cases {
            let mut ops = operands(c);
            let first = count(c, &mut ops, par).0;
            let later: Vec<usize> = (0..3).map(|_| count(c, &mut ops, par).0).collect();
            report.push(format!(
                "{:?} m={} n={} k={} lhs_colmajor={} dst_colmajor={}: first={} later={:?}",
                par, c.m, c.n, c.k, c.lhs_colmajor, c.dst_colmajor, first, later
            ));
        }
    }
    // A thread that has seen a larger product must reuse its buffer for a smaller one.
    for &par in &parallelisms {
        for c in &cases {
            let mut ops = operands(c);
            let n = count(c, &mut ops, par).0;
            report.push(format!(
                "{:?} m={} n={} k={} lhs_colmajor={} dst_colmajor={}: after all shapes={}",
                par, c.m, c.n, c.k, c.lhs_colmajor, c.dst_colmajor, n
            ));
        }
    }
    // A product over the retention cap must not cost the *next* small product an allocation:
    // the cap decides what is kept, not whether a buffer that still fits is thrown away. The cap
    // is one `CACHE_INFO[1]` slice, so this check only bites where the cap lies between the
    // 18 KiB of `48 x 48 x 48` and the 683 KiB this asks for: on the x86 host measured here
    // (cap 512 KiB) it is red before the fix and green after, while on a host whose cap is
    // larger (an M-series perf core reports ~2.6 MB) the big request is kept and the small
    // product reuses it, which passes either way — there the byte bound below is the contract.
    let small = Case {
        m: 48,
        n: 48,
        k: 48,
        lhs_colmajor: true,
        dst_colmajor: true,
        rhs_colmajor: true,
    };
    // A row-major `rhs` keeps this off the packed-`rhs` path at `k = 2048` (see the PR note on
    // the pre-existing `300 x 300 x 512` crash), so the request is the prepacked `lhs`,
    // `kc * MR * ceil(m / MR)`, with `kc` bounded by the L2 panel fit.
    let big = Case {
        m: 126,
        n: 64,
        k: 2048,
        lhs_colmajor: true,
        dst_colmajor: false,
        rhs_colmajor: false,
    };
    for &par in &parallelisms {
        let mut small_ops = operands(&small);
        for _ in 0..3 {
            small_ops.dst.fill(0.0);
            count(&small, &mut small_ops, par);
        }
        let mut big_ops = operands(&big);
        for _ in 0..2 {
            big_ops.dst.fill(0.0);
            count(&big, &mut big_ops, par);
        }
        small_ops.dst.fill(0.0);
        let after = count(&small, &mut small_ops, par).0;
        report.push(format!("{par:?}: small after oversized={after}"));
    }

    // The packing request may only be as large as the matrix needs: `kc <= k`,
    // `nc <= ceil(n / NR) * NR <= n + 16`, prepacked `lhs <= k * (m + 31)` (`MR <= 32`). A host
    // without an L3 (Apple reports `L3 = 0`) used to ask for `kc * 128 * NR` columns instead,
    // 3.99 MB for `n = 300` where 0.6 MB is written.
    for (m, n, k) in [(60usize, 300usize, 256usize), (300, 300, 240)] {
        let c = Case {
            m,
            n,
            k,
            lhs_colmajor: true,
            dst_colmajor: true,
            rhs_colmajor: true,
        };
        let mut ops = operands(&c);
        count(&c, &mut ops, Parallelism::None);
        ops.dst.fill(0.0);
        let (_events, bytes) = count(&c, &mut ops, Parallelism::None);
        let bound = k * ((n + 16) + (m + 32)) * core::mem::size_of::<f64>() + 4096;
        report.push(format!(
            "m={m} n={n} k={k}: bytes={bytes} bound={bound} {}",
            if bytes <= bound {
                "bounded"
            } else {
                "OVERSIZED"
            }
        ));
    }

    let text = report.join("\n");
    println!("{text}");
    let bad: Vec<&str> = text
        .lines()
        .filter(|l| {
            !(l.ends_with("later=[0, 0, 0]")
                || l.ends_with("after all shapes=0")
                || l.ends_with("small after oversized=0")
                || l.ends_with("bounded"))
        })
        .collect();
    assert!(bad.is_empty(), "warm calls allocated:\n{}", bad.join("\n"));
}
