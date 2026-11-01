//! Reference comparison for `gemm_f64`'s public entry point.
//!
//! Exercises the shapes that hit the trailing partial panels of the microkernel
//! (`m % 8`, `n % 32`), both destination layouts, and the `alpha`/`beta` cases the
//! microkernel has to special-case.

use gemm_common::Parallelism;

fn reference(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f64],
    rhs: &[f64],
    dst: &mut [f64],
    dst_cs: isize,
    dst_rs: isize,
    alpha: f64,
    beta: f64,
) {
    for j in 0..n {
        for i in 0..m {
            let mut s = 0.0;
            for kk in 0..k {
                s += lhs[i + kk * m] * rhs[kk + j * k];
            }
            let dst = &mut dst[(i as isize * dst_rs + j as isize * dst_cs) as usize];
            *dst = alpha * *dst + beta * s;
        }
    }
}

fn check(m: usize, n: usize, k: usize, alpha: f64, beta: f64, colmajor: bool) {
    let lhs: Vec<f64> = (0..m * k).map(|i| (i % 17) as f64 / 8.0 - 1.0).collect();
    let rhs: Vec<f64> = (0..k * n).map(|i| (i % 13) as f64 / 8.0 - 0.5).collect();
    let init: Vec<f64> = (0..m * n).map(|i| (i % 7) as f64 / 16.0).collect();

    let (dst_cs, dst_rs) = if colmajor { (m as isize, 1) } else { (1, n as isize) };

    let mut got = init.clone();
    let mut want = init.clone();

    unsafe {
        gemm_f64::gemm::f64::get_gemm_fn()(
            m,
            n,
            k,
            got.as_mut_ptr(),
            dst_cs,
            dst_rs,
            true,
            lhs.as_ptr(),
            m as isize,
            1,
            rhs.as_ptr(),
            k as isize,
            1,
            alpha,
            beta,
            false,
            false,
            false,
            Parallelism::None,
        );
    }
    reference(m, n, k, &lhs, &rhs, &mut want, dst_cs, dst_rs, alpha, beta);

    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1.0),
            "m={m} n={n} k={k} alpha={alpha} beta={beta} colmajor={colmajor} \
             at {idx}: got {g}, want {w}"
        );
    }
}

#[test]
fn matches_reference() {
    let shapes = [
        (1, 1, 1),
        (1, 1, 2),
        (3, 5, 7),
        (4, 4, 4),
        (7, 33, 5),
        (8, 8, 8),
        (9, 17, 10),
        (16, 32, 4),
        (17, 40, 9),
        (33, 33, 33),
        (40, 17, 9),
        (63, 2, 10),
        (64, 64, 64),
    ];
    let coeffs = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (2.3, 2.3)];

    for (m, n, k) in shapes {
        for (alpha, beta) in coeffs {
            for colmajor in [true, false] {
                check(m, n, k, alpha, beta, colmajor);
            }
        }
    }
}
