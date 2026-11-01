//! Apple SME microkernel tests for `f32`.
//!
//! Two layers:
//!  - `ukernel_block`: calls the `global_asm!` kernel directly on a packed
//!    `MR x NR` complex block (including the conjugation flags), so a wrong
//!    outer product, unzip or spill shows up on its own rather than as a
//!    mismatch somewhere inside the blocking,
//!  - `matches_reference`: runs the dispatched entry point over shapes that hit
//!    the trailing partial panels, both destination layouts, the five
//!    `alpha`/`beta` cases and all four `conj_lhs`/`conj_rhs` combinations,
//!    comparing against a plain scalar reference.
//!
//! Both are gated on the kernel actually being present; on a machine without
//! `FEAT_SME_F32F32` the dispatch test still exercises the NEON `fcmla` path.

use gemm_common::Parallelism;

const TOL: f64 = 4e-3;

fn val(i: usize) -> num_complex::Complex32 {
    num_complex::Complex32::new((i % 17) as f32 / 8.0 - 1.0, (i % 11) as f32 / 8.0 - 0.5)
}

#[allow(clippy::too_many_arguments)]
fn reference(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[num_complex::Complex32],
    rhs: &[num_complex::Complex32],
    dst: &mut [num_complex::Complex32],
    dst_cs: isize,
    dst_rs: isize,
    alpha: num_complex::Complex32,
    beta: num_complex::Complex32,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) {
    for j in 0..n {
        for i in 0..m {
            let mut s = num_complex::Complex32::new(0.0, 0.0);
            for kk in 0..k {
                let a = lhs[i + kk * m];
                let b = rhs[kk + j * k];
                let a = if conj_lhs { a.conj() } else { a };
                let b = if conj_rhs { b.conj() } else { b };
                s += a * b;
            }
            let dst = &mut dst[(i as isize * dst_rs + j as isize * dst_cs) as usize];
            // the crate's convention: `conj_dst` conjugates the destination, it is not
            // a conjugation of the product
            let dst0 = if conj_dst { (*dst).conj() } else { *dst };
            *dst = alpha * dst0 + beta * s;
        }
    }
}

fn check(
    m: usize,
    n: usize,
    k: usize,
    alpha: num_complex::Complex32,
    beta: num_complex::Complex32,
    colmajor: bool,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) {
    let lhs: Vec<num_complex::Complex32> = (0..m * k).map(val).collect();
    let rhs: Vec<num_complex::Complex32> = (0..k * n).map(|i| val(i + 5)).collect();
    let init: Vec<num_complex::Complex32> = (0..m * n).map(|i| val(i + 9)).collect();

    let (dst_cs, dst_rs) = if colmajor {
        (m as isize, 1)
    } else {
        (1, n as isize)
    };

    let mut got = init.clone();
    let mut want = init.clone();

    unsafe {
        gemm_c32::gemm::f32::get_gemm_fn()(
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
            conj_dst,
            conj_lhs,
            conj_rhs,
            Parallelism::None,
        );
    }
    reference(
        m, n, k, &lhs, &rhs, &mut want, dst_cs, dst_rs, alpha, beta, conj_dst, conj_lhs, conj_rhs,
    );

    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let scale = (w.re.abs() + w.im.abs()).max(1.0) as f64;
        assert!(
            ((g.re - w.re).abs() as f64) <= TOL * scale
                && ((g.im - w.im).abs() as f64) <= TOL * scale,
            "m={m} n={n} k={k} alpha={alpha:?} beta={beta:?} colmajor={colmajor} \
             conj_dst={conj_dst} conj_lhs={conj_lhs} conj_rhs={conj_rhs} at {idx}: \
             got {g:?}, want {w:?}"
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
        (15, 31, 6),
        (16, 32, 8),
        (17, 33, 9),
        (33, 129, 10),
        (40, 96, 9),
        (63, 2, 10),
        (100, 200, 70),
    ];
    let coeffs = [
        (
            num_complex::Complex32::new(1.0, 0.0),
            num_complex::Complex32::new(0.0, 0.0),
        ),
        (
            num_complex::Complex32::new(1.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0),
        ),
        (
            num_complex::Complex32::new(0.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0),
        ),
        (
            num_complex::Complex32::new(0.0, 0.0),
            num_complex::Complex32::new(0.0, 0.0),
        ),
        (
            num_complex::Complex32::new(2.3, 4.1),
            num_complex::Complex32::new(2.3, 4.1),
        ),
    ];

    for (m, n, k) in shapes {
        for (alpha, beta) in coeffs {
            for colmajor in [true, false] {
                for conj_dst in [false, true] {
                    for conj_lhs in [false, true] {
                        for conj_rhs in [false, true] {
                            check(m, n, k, alpha, beta, colmajor, conj_dst, conj_lhs, conj_rhs);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple", feature = "sme"))]
mod ukr {
    use super::*;
    use gemm_c32::microkernel::sme::c32::*;
    use gemm_common::cache::HasSme;

    const MR: usize = MR_DIV_N * N;

    fn check_block(m: usize, n: usize, k: usize, conj_dst: bool, conj_lhs: bool, conj_rhs: bool) {
        let mut lhs = vec![num_complex::Complex32::new(0.0, 0.0); k * MR];
        let mut rhs = vec![num_complex::Complex32::new(0.0, 0.0); k * NR];
        for kk in 0..k {
            for i in 0..MR {
                lhs[i + kk * MR] = val(i + 3 * kk);
            }
            for j in 0..NR {
                rhs[j + kk * NR] = val(2 * j + 5 * kk + 1);
            }
        }

        let dst_init: Vec<num_complex::Complex32> = (0..MR * NR).map(|i| val(i + 2)).collect();
        let mut dst = dst_init.clone();
        unsafe {
            ukernel(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                lhs.as_ptr(),
                rhs.as_ptr(),
                MR as isize,
                1,
                MR as isize,
                NR as isize,
                1,
                num_complex::Complex32::new(1.0, 0.0),
                num_complex::Complex32::new(1.0, 0.0),
                1,
                conj_dst,
                conj_lhs,
                conj_rhs,
                core::ptr::null(),
            );
        }

        for j in 0..n {
            for i in 0..m {
                let mut want = num_complex::Complex32::new(0.0, 0.0);
                for kk in 0..k {
                    let a = lhs[i + kk * MR];
                    let b = rhs[j + kk * NR];
                    let a = if conj_lhs { a.conj() } else { a };
                    let b = if conj_rhs { b.conj() } else { b };
                    want += a * b;
                }
                let init = dst_init[i + j * MR];
                let init = if conj_dst { init.conj() } else { init };
                let want = init + want;
                let got = dst[i + j * MR];
                let scale = (want.re.abs() + want.im.abs()).max(1.0) as f64;
                assert!(
                    ((got.re - want.re).abs() as f64) <= TOL * scale
                        && ((got.im - want.im).abs() as f64) <= TOL * scale,
                    "ukernel m={m} n={n} k={k} conj_dst={conj_dst} conj_lhs={conj_lhs} \
                     conj_rhs={conj_rhs} at i={i} j={j}: got {got:?}, want {want:?}"
                );
            }
        }
    }

    #[test]
    fn ukernel_block() {
        if !HasSme::get_f32() {
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
            (1, NR / 2, 3),
        ] {
            for conj_dst in [false, true] {
                for conj_lhs in [false, true] {
                    for conj_rhs in [false, true] {
                        check_block(m, n, k, conj_dst, conj_lhs, conj_rhs);
                    }
                }
            }
        }
    }
}
