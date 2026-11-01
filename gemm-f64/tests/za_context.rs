//! ZA / streaming-mode context-switch safety for the `f64` SME microkernel.
//!
//! The kernel enables streaming mode, keeps live accumulators in the four `ZA`
//! tiles across the whole `k` loop and only reads them back at the end. That is
//! only safe because macOS 26 saves and restores `ZA` plus the streaming mode
//! state when it preempts a thread; if it did not, a thread that is descheduled
//! inside the `k` loop would come back with another thread's accumulators (or a
//! trapped `SMSTART`).
//!
//! The test runs more threads than there are performance cores and mixes
//! `yield_now()` with short sleeps into the loop, so preemption inside the
//! assembly is the common case rather than a rare one. Every iteration must
//! reproduce the first iteration's result bit for bit, and the first iteration is
//! checked against a scalar reference.

use std::time::Duration;

use gemm_common::Parallelism;

fn reference(m: usize, n: usize, k: usize, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) {
    for j in 0..n {
        for i in 0..m {
            let mut s = 0.0;
            for kk in 0..k {
                s += lhs[i + kk * m] * rhs[kk + j * k];
            }
            dst[i + j * m] = s;
        }
    }
}

fn val(i: usize) -> f64 {
    (i % 17) as f64 / 16.0 - 0.5
}

#[test]
fn za_survives_context_switches() {
    let n = 192;
    let threads = 8;
    let iters = 150;

    let lhs: Vec<f64> = (0..n * n).map(val).collect();
    let rhs: Vec<f64> = (0..n * n).map(|i| val(i + 7)).collect();
    let lhs = &lhs;
    let rhs = &rhs;

    let mut handles = Vec::new();
    for _ in 0..threads {
        let lhs = lhs.clone();
        let rhs = rhs.clone();
        handles.push(std::thread::spawn(move || {
            let mut want = vec![0.0; n * n];
            reference(n, n, n, &lhs, &rhs, &mut want);

            let mut got = vec![0.0; n * n];
            let mut first: Option<Vec<f64>> = None;

            for it in 0..iters {
                got.iter_mut().for_each(|x| *x = 0.0);
                unsafe {
                    gemm_f64::gemm::f64::get_gemm_fn()(
                        n,
                        n,
                        n,
                        got.as_mut_ptr(),
                        n as isize,
                        1,
                        false,
                        lhs.as_ptr(),
                        n as isize,
                        1,
                        rhs.as_ptr(),
                        n as isize,
                        1,
                        1.0,
                        1.0,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }

                if it == 0 {
                    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                            "first iteration differs from the scalar reference at {idx}: \
                             got {g:?}, want {w:?}"
                        );
                    }
                    first = Some(got.clone());
                } else {
                    let first = first.as_ref().unwrap();
                    if got != *first {
                        let idx = got
                            .iter()
                            .zip(first.iter())
                            .position(|(a, b)| a != b)
                            .unwrap();
                        panic!(
                            "iteration {it} differs from iteration 0 at {idx}: got {:?}, \
                             want {:?} (ZA was not preserved across a context switch?)",
                            got[idx], first[idx]
                        );
                    }
                }

                if it % 5 == 0 {
                    std::thread::yield_now();
                }
                if it % 50 == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
