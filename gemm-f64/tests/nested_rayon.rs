//! Products issued from inside rayon tasks while each product is itself multi-threaded.
//!
//! `gemm_basic_generic` keeps its packing buffer across `par_for_each`; a worker that blocks
//! there can steal another task's product and run it on the same thread. Both products must
//! get their own buffer, so every result is compared against a reference. Stealing is up to
//! the scheduler: with a temporary probe in `PackBuffer::drop`, an M5 Max run of this test
//! nested about once, so a single run is a smoke test and not a proof.

#![cfg(feature = "rayon")]

use gemm_common::Parallelism;
use rayon::prelude::*;

#[test]
fn nested_products_on_rayon_workers_match_reference() {
    // Every size is above the default threading threshold (48 * 48 * 256), so each product forks.
    let sizes = [192usize, 160, 224, 96];
    (0..256usize).into_par_iter().for_each(|task| {
        let m = sizes[task % sizes.len()];
        let (n, k) = (m, m + task % 3);
        let lhs: Vec<f64> = (0..m * k)
            .map(|i| ((i + task) % 17) as f64 / 8.0 - 1.0)
            .collect();
        let rhs: Vec<f64> = (0..k * n)
            .map(|i| ((i * 3 + task) % 13) as f64 / 8.0 - 0.5)
            .collect();
        let mut got = vec![0.0f64; m * n];
        unsafe {
            gemm_f64::gemm::f64::get_gemm_fn()(
                m,
                n,
                k,
                got.as_mut_ptr(),
                m as isize,
                1,
                false,
                lhs.as_ptr(),
                m as isize,
                1,
                rhs.as_ptr(),
                k as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                Parallelism::Rayon(0),
            );
        }
        for j in 0..n {
            for i in 0..m {
                let want: f64 = (0..k).map(|p| lhs[i + p * m] * rhs[p + j * k]).sum();
                let g = got[i + j * m];
                assert!(
                    (g - want).abs() <= 1e-10 * want.abs().max(1.0),
                    "task {task} ({m}x{n}x{k}) at ({i}, {j}): {g} vs {want}"
                );
            }
        }
    });
}
