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

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
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

/// Runs one product and returns how many allocations it made.
fn count(c: &Case, ops: &mut Operands, parallelism: Parallelism) -> usize {
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
    let gemm = gemm_f64::gemm::f64::get_gemm_fn();
    let before = ALLOCS.load(Ordering::Relaxed);
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
            c.k as isize,
            1,
            1.0,
            1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
    ALLOCS.load(Ordering::Relaxed) - before
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
        },
        Case {
            m: 48,
            n: 48,
            k: 48,
            lhs_colmajor: true,
            dst_colmajor: false,
        },
        Case {
            m: 48,
            n: 48,
            k: 48,
            lhs_colmajor: false,
            dst_colmajor: true,
        },
        Case {
            m: 48,
            n: 32,
            k: 16,
            lhs_colmajor: true,
            dst_colmajor: true,
        },
        Case {
            m: 32,
            n: 48,
            k: 16,
            lhs_colmajor: true,
            dst_colmajor: true,
        },
        Case {
            m: 47,
            n: 33,
            k: 31,
            lhs_colmajor: true,
            dst_colmajor: true,
        },
        Case {
            m: 96,
            n: 96,
            k: 96,
            lhs_colmajor: true,
            dst_colmajor: true,
        },
    ];
    let mut parallelisms = vec![Parallelism::None];
    #[cfg(feature = "rayon")]
    parallelisms.push(Parallelism::Rayon(1));

    let mut report = Vec::new();
    for &par in &parallelisms {
        for c in &cases {
            let mut ops = operands(c);
            let first = count(c, &mut ops, par);
            let later: Vec<usize> = (0..3).map(|_| count(c, &mut ops, par)).collect();
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
            let n = count(c, &mut ops, par);
            report.push(format!(
                "{:?} m={} n={} k={} lhs_colmajor={} dst_colmajor={}: after all shapes={}",
                par, c.m, c.n, c.k, c.lhs_colmajor, c.dst_colmajor, n
            ));
        }
    }
    let text = report.join("\n");
    println!("{text}");
    let bad: Vec<&str> = text
        .lines()
        .filter(|l| !(l.ends_with("later=[0, 0, 0]") || l.ends_with("after all shapes=0")))
        .collect();
    assert!(bad.is_empty(), "warm calls allocated:\n{}", bad.join("\n"));
}
