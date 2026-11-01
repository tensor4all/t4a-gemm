//! Apple SME microkernel tests for `f32`.
//!
//! Two layers:
//!  - `ukernel_block`: calls the `global_asm!` kernel directly on a packed
//!    `MR x NR` block, so a wrong outer product or spill shows up on its own
//!    rather than as a mismatch somewhere inside the blocking,
//!  - `matches_reference`: runs the dispatched entry point over shapes that hit
//!    the trailing partial panels (`m % 16`, `n % 64`), both destination layouts
//!    and the five `alpha`/`beta` cases the microkernel special-cases.
//!
//! Both are gated on the kernel actually being present; on a machine without
//! `FEAT_SME` the dispatch test still exercises the NEON path.

use gemm_common::Parallelism;

fn reference(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    rhs: &[f32],
    dst: &mut [f32],
    dst_cs: isize,
    dst_rs: isize,
    alpha: f32,
    beta: f32,
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

fn check(m: usize, n: usize, k: usize, alpha: f32, beta: f32, colmajor: bool) {
    let lhs: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 / 8.0 - 0.5).collect();
    let init: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 / 16.0).collect();

    let (dst_cs, dst_rs) = if colmajor {
        (m as isize, 1)
    } else {
        (1, n as isize)
    };

    let mut got = init.clone();
    let mut want = init.clone();

    unsafe {
        gemm_f32::gemm::f32::get_gemm_fn()(
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

    let tol = 2e-4;
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
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
        (15, 63, 6),
        (16, 64, 8),
        (17, 65, 9),
        (33, 129, 10),
        (40, 96, 9),
        (63, 2, 10),
        (100, 200, 70),
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

#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    feature = "apple-sme"
))]
mod ukr {
    use gemm_common::cache::HasAppleSme;
    use gemm_f32::microkernel::sme::f32::*;

    const MR: usize = MR_DIV_N * N;

    /// `lhs` and `rhs` are handed over in the layout `pack_operands` produces:
    /// `MR` / `NR` contiguous elements per `k` step.
    fn check_block(m: usize, n: usize, k: usize) {
        let mut lhs = vec![0.0f32; k * MR];
        let mut rhs = vec![0.0f32; k * NR];
        for kk in 0..k {
            for i in 0..MR {
                lhs[i + kk * MR] = ((i + 3 * kk) % 11) as f32 / 8.0 - 0.7;
            }
            for j in 0..NR {
                rhs[j + kk * NR] = ((2 * j + 5 * kk) % 9) as f32 / 8.0 - 0.5;
            }
        }

        let mut accum = vec![0.0f32; MR * NR];
        // `gemm_basic_generic` hands the microkernel a `[NR][MR]` temporary with
        // `dst_cs = MR`, `dst_rs = 1`, `alpha = 0`, `beta = 1`.
        unsafe {
            ukernel(
                m,
                n,
                k,
                accum.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                MR as isize,
                1,
                MR as isize,
                NR as isize,
                1,
                0.0,
                1.0,
                0,
                false,
                false,
                false,
                core::ptr::null(),
            );
        }

        for j in 0..n {
            for i in 0..m {
                let mut want = 0.0f32;
                for kk in 0..k {
                    want += lhs[i + kk * MR] * rhs[j + kk * NR];
                }
                let got = accum[i + j * MR];
                assert!(
                    (got - want).abs() <= 2e-4 * want.abs().max(1.0),
                    "ukernel m={m} n={n} k={k} at i={i} j={j}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn ukernel_block() {
        if !HasAppleSme::get_f32() {
            eprintln!("skipping: no FEAT_SME_F32F32 on this machine");
            return;
        }
        for (m, n, k) in [
            (MR, NR, 1),
            (MR, NR, 7),
            (MR, NR, 64),
            (1, 1, 5),
            (3, 9, 5),
            (MR - 1, NR - 1, 33),
            (5, 1, 3),
            (1, 17, 3),
        ] {
            check_block(m, n, k);
        }
    }
}
