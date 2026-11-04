//! Micro-benchmark for the `did_pack_lhs` sizing.
//!
//! Strides follow the crate's own tests (column-major everywhere), so a crash here
//! is the library and not the harness.
//!
//!   cargo run --release --example did_pack_sizing -p t4a-gemm-f64 -- [m n k reps]

use gemm_common::Parallelism;
use std::time::Instant;

fn bench(m: usize, n: usize, k: usize, reps: usize) -> f64 {
    let lhs: Vec<f64> = (0..m * k).map(|i| (i % 17) as f64 / 8.0 - 1.0).collect();
    let rhs: Vec<f64> = (0..k * n).map(|i| (i % 13) as f64 / 8.0 - 0.5).collect();
    let mut dst = vec![0.0f64; m * n];
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..reps {
            unsafe {
                gemm_f64::gemm::f64::get_gemm_fn()(
                    m, n, k,
                    dst.as_mut_ptr(), m as isize, 1, false,
                    lhs.as_ptr(), m as isize, 1,
                    rhs.as_ptr(), k as isize, 1,
                    1.0, 0.0, false, false, false,
                    Parallelism::None,
                );
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    std::hint::black_box(&dst);
    best * 1e6 / reps as f64
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (m, n, k) = if args.len() >= 3 { (args[0], args[1], args[2]) } else { (200_000, 32, 32) };
    let reps = if args.len() >= 4 { args[3] } else { if m > 100_000 { 3 } else { 500 } };
    let per_call = bench(m, n, k, reps);
    println!("m={m} n={n} k={k} reps={reps}: {per_call:.1} us/call");
}
