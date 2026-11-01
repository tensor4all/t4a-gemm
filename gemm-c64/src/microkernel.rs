pub mod scalar {
    pub mod f64 {
        type T = f64;
        const N: usize = 2;
        const CPLX_N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value, value]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0], lhs[1] + rhs[1]]
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            [a[0], -a[1]]
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            [a[1], a[0]]
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, _a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            [
                a_re_im[0] * b_re[0] - a_re_im[1] * b_im[0],
                a_re_im[1] * b_re[0] + a_re_im[0] * b_im[0],
            ]
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                add(
                    c_re_im,
                    mul_cplx(a_re_im, a_im_re, b_re, [-b_im[0], -b_im[1]]),
                )
            } else {
                add(c_re_im, mul_cplx(a_re_im, a_im_re, b_re, b_im))
            }
        }

        microkernel_cplx!(, 2, x1x1, 1, 1);
        microkernel_cplx!(, 2, x1x2, 1, 2);
        microkernel_cplx!(, 2, x1x3, 1, 3);
        microkernel_cplx!(, 2, x1x4, 1, 4);

        microkernel_cplx!(, 2, x2x1, 2, 1);
        microkernel_cplx!(, 2, x2x2, 2, 2);
        microkernel_cplx!(, 2, x2x3, 2, 3);
        microkernel_cplx!(, 2, x2x4, 2, 4);

        microkernel_cplx_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
        pub const H_CPLX_M: usize = 0;
        pub const H_CPLX_N: usize = 0;
        pub const H_CPLX_UKR: [[gemm_common::microkernel::HMicroKernelFn<num_complex::Complex<T>>;
            H_CPLX_N]; H_CPLX_M] = [];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod fma {
    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        use gemm_common::pulp::u64x4;

        type T = f64;
        const N: usize = 4;
        const CPLX_N: usize = 2;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m256d = unsafe { transmute([0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64]) };
            transmute(_mm256_xor_pd(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn neg_conj(a: Pack) -> Pack {
            const MASK: __m256d = unsafe { transmute([-0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64]) };
            transmute(_mm256_xor_pd(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn neg(a: Pack) -> Pack {
            const MASK: __m256d = unsafe { transmute([-0.0_f64, -0.0_f64, -0.0_f64, -0.0_f64]) };
            transmute(_mm256_xor_pd(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm256_permute_pd::<0b0101>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm256_fmaddsub_pd(
                transmute(a_re_im),
                transmute(b_re),
                _mm256_mul_pd(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx_step0(
            a_re_im: Pack,
            b_re: Pack,
            c_re_im: Pack,
            neg_conj_rhs: bool,
        ) -> Pack {
            if neg_conj_rhs {
                transmute(_mm256_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    transmute(c_re_im),
                ))
            } else {
                transmute(_mm256_fmsubadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    transmute(c_re_im),
                ))
            }
        }

        #[inline(always)]
        unsafe fn mul_add_cplx_step1(
            a_im_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            neg_conj_rhs: bool,
        ) -> Pack {
            if neg_conj_rhs {
                transmute(_mm256_fmaddsub_pd(
                    transmute(a_im_re),
                    transmute(b_im),
                    transmute(c_re_im),
                ))
            } else {
                transmute(_mm256_fmsubadd_pd(
                    transmute(a_im_re),
                    transmute(b_im),
                    transmute(c_re_im),
                ))
            }
        }

        microkernel_cplx_2step!(["fma"], 2, cplx_x1x1, 1, 1);
        microkernel_cplx_2step!(["fma"], 2, cplx_x1x2, 1, 2);
        microkernel_cplx_2step!(["fma"], 2, cplx_x1x3, 1, 3);
        microkernel_cplx_2step!(["fma"], 2, cplx_x1x4, 1, 4);
        microkernel_cplx_2step!(["fma"], 2, cplx_x1x5, 1, 5);
        microkernel_cplx_2step!(["fma"], 2, cplx_x1x6, 1, 6);

        microkernel_cplx_2step!(["fma"], 2, cplx_x2x1, 2, 1);
        microkernel_cplx_2step!(["fma"], 2, cplx_x2x2, 2, 2);
        microkernel_cplx_2step!(["fma"], 2, cplx_x2x3, 2, 3);
        microkernel_cplx_2step!(["fma"], 2, cplx_x2x4, 2, 4);
        microkernel_cplx_2step!(["fma"], 2, cplx_x2x5, 2, 5);
        microkernel_cplx_2step!(["fma"], 2, cplx_x2x6, 2, 6);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3, cplx_x1x4, cplx_x1x5, cplx_x1x6,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3, cplx_x2x4, cplx_x2x5, cplx_x2x6,],
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            let ab = transmute(a);
            let xy = transmute(b);

            let yx = _mm256_permute_pd::<0b0101>(xy);
            let aa = _mm256_unpacklo_pd(ab, ab);
            let bb = _mm256_unpackhi_pd(ab, ab);

            transmute(_mm256_fmaddsub_pd(
                aa,
                xy,
                _mm256_fmaddsub_pd(bb, yx, transmute(c)),
            ))
        }

        #[inline(always)]
        unsafe fn conj_mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            let ab = transmute(a);
            let xy = transmute(b);

            let yx = _mm256_permute_pd::<0b0101>(xy);
            let aa = _mm256_unpacklo_pd(ab, ab);
            let bb = _mm256_unpackhi_pd(ab, ab);

            transmute(_mm256_fmsubadd_pd(
                aa,
                xy,
                _mm256_fmsubadd_pd(bb, yx, transmute(c)),
            ))
        }

        static U64_MASKS: [u64x4; 3] = [
            u64x4(0, 0, 0, 0),
            u64x4(!0, !0, 0, 0),
            u64x4(!0, !0, !0, !0),
        ];

        #[inline(always)]
        unsafe fn partial_load(ptr: *const num_complex::Complex<T>, len: usize) -> Pack {
            transmute(_mm256_maskload_pd(
                ptr as *mut T,
                transmute(*(U64_MASKS.as_ptr().add(len))),
            ))
        }

        #[inline(always)]
        unsafe fn reduce_sum(x: Pack) -> num_complex::Complex<T> {
            let x: __m256d = transmute(x);
            let x = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd::<1>(x));
            transmute(x)
        }

        horizontal_cplx_kernel!(["fma"], hx1x1, 1, 1);
        horizontal_cplx_kernel!(["fma"], hx1x2, 1, 2);
        horizontal_cplx_kernel!(["fma"], hx2x1, 2, 1);
        horizontal_cplx_kernel!(["fma"], hx2x2, 2, 2);
        hmicrokernel_cplx_fn_array! {
            [hx1x1, hx1x2,],
            [hx2x1, hx2x2,],
        }
    }
}

#[cfg(all(feature = "x86-v4", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512f {
    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f64;
        const N: usize = 8;
        const CPLX_N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn neg(a: Pack) -> Pack {
            const MASK: __m512i = unsafe {
                transmute([
                    -0.0_f64, -0.0_f64, -0.0_f64, -0.0_f64, -0.0_f64, -0.0_f64, -0.0_f64, -0.0_f64,
                ])
            };
            transmute(_mm512_xor_si512(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m512i = unsafe {
                transmute([
                    0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64,
                ])
            };
            transmute(_mm512_xor_si512(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn neg_conj(a: Pack) -> Pack {
            const MASK: __m512i = unsafe {
                transmute([
                    -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64,
                ])
            };
            transmute(_mm512_xor_si512(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm512_permute_pd::<0b01010101>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm512_fmaddsub_pd(
                transmute(a_re_im),
                transmute(b_re),
                _mm512_mul_pd(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[target_feature(enable = "avx512f")]
        #[inline]
        unsafe fn subadd_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
            _mm512_fmaddsub_pd(
                a,
                b,
                transmute(_mm512_xor_si512(
                    transmute(c),
                    transmute(_mm512_set1_pd(-0.0)),
                )),
            )
        }

        #[inline(always)]
        unsafe fn mul_add_cplx_step0(
            a_re_im: Pack,
            b_re: Pack,
            c_re_im: Pack,
            neg_conj_rhs: bool,
        ) -> Pack {
            if neg_conj_rhs {
                transmute(_mm512_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    transmute(c_re_im),
                ))
            } else {
                transmute(subadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    transmute(c_re_im),
                ))
            }
        }

        #[inline(always)]
        unsafe fn mul_add_cplx_step1(
            a_im_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            neg_conj_rhs: bool,
        ) -> Pack {
            if neg_conj_rhs {
                transmute(_mm512_fmaddsub_pd(
                    transmute(a_im_re),
                    transmute(b_im),
                    transmute(c_re_im),
                ))
            } else {
                transmute(subadd_pd(
                    transmute(a_im_re),
                    transmute(b_im),
                    transmute(c_re_im),
                ))
            }
        }

        #[inline(always)]
        #[allow(dead_code)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(subadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    subadd_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm512_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmaddsub_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel_cplx_2step!(["avx512f"], 2, cplx_x1x1, 1, 1);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x1x2, 1, 2);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x1x3, 1, 3);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x1x4, 1, 4);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x1x5, 1, 5);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x1x6, 1, 6);

        microkernel_cplx_2step!(["avx512f"], 2, cplx_x2x1, 2, 1);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x2x2, 2, 2);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x2x3, 2, 3);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x2x4, 2, 4);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x2x5, 2, 5);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x2x6, 2, 6);

        microkernel_cplx_2step!(["avx512f"], 2, cplx_x3x1, 3, 1);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x3x2, 3, 2);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x3x3, 3, 3);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x3x4, 3, 4);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x3x5, 3, 5);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x3x6, 3, 6);

        microkernel_cplx_2step!(["avx512f"], 2, cplx_x4x1, 4, 1);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x4x2, 4, 2);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x4x3, 4, 3);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x4x4, 4, 4);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x4x5, 4, 5);
        microkernel_cplx_2step!(["avx512f"], 2, cplx_x4x6, 4, 6);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3, cplx_x1x4, cplx_x1x5, cplx_x1x6,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3, cplx_x2x4, cplx_x2x5, cplx_x2x6,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3, cplx_x3x4, cplx_x3x5, cplx_x3x6,],
            [cplx_x4x1, cplx_x4x2, cplx_x4x3, cplx_x4x4, cplx_x4x5, cplx_x4x6,],
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            let ab = transmute(a);
            let xy = transmute(b);

            let yx = _mm512_permute_pd::<0b01010101>(xy);
            let aa = _mm512_unpacklo_pd(ab, ab);
            let bb = _mm512_unpackhi_pd(ab, ab);

            transmute(_mm512_fmaddsub_pd(
                aa,
                xy,
                _mm512_fmaddsub_pd(bb, yx, transmute(c)),
            ))
        }

        #[inline(always)]
        unsafe fn conj_mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            let ab = transmute(a);
            let xy = transmute(b);

            let yx = _mm512_permute_pd::<0b01010101>(xy);
            let aa = _mm512_unpacklo_pd(ab, ab);
            let bb = _mm512_unpackhi_pd(ab, ab);

            transmute(subadd_pd(aa, xy, subadd_pd(bb, yx, transmute(c))))
        }

        static U64_MASKS: [u8; 5] = [
            0b00000000, //
            0b00000011, //
            0b00001111, //
            0b00111111, //
            0b11111111, //
        ];

        #[inline(always)]
        unsafe fn partial_load(ptr: *const num_complex::Complex<T>, len: usize) -> Pack {
            transmute(_mm512_maskz_loadu_pd(
                transmute(*(U64_MASKS.as_ptr().add(len))),
                ptr as *mut T,
            ))
        }

        #[inline(always)]
        unsafe fn reduce_sum(x: Pack) -> num_complex::Complex<T> {
            let x = transmute(x);
            let x = _mm256_add_pd(_mm512_castpd512_pd256(x), _mm512_extractf64x4_pd::<1>(x));
            let x = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd::<1>(x));
            transmute(x)
        }

        horizontal_cplx_kernel!(["avx512f"], hx1x1, 1, 1);
        horizontal_cplx_kernel!(["avx512f"], hx1x2, 1, 2);
        horizontal_cplx_kernel!(["avx512f"], hx1x3, 1, 3);
        horizontal_cplx_kernel!(["avx512f"], hx1x4, 1, 4);
        horizontal_cplx_kernel!(["avx512f"], hx2x1, 2, 1);
        horizontal_cplx_kernel!(["avx512f"], hx2x2, 2, 2);
        horizontal_cplx_kernel!(["avx512f"], hx2x3, 2, 3);
        horizontal_cplx_kernel!(["avx512f"], hx2x4, 2, 4);
        horizontal_cplx_kernel!(["avx512f"], hx3x1, 3, 1);
        horizontal_cplx_kernel!(["avx512f"], hx3x2, 3, 2);
        horizontal_cplx_kernel!(["avx512f"], hx3x3, 3, 3);
        horizontal_cplx_kernel!(["avx512f"], hx3x4, 3, 4);
        horizontal_cplx_kernel!(["avx512f"], hx4x1, 4, 1);
        horizontal_cplx_kernel!(["avx512f"], hx4x2, 4, 2);
        horizontal_cplx_kernel!(["avx512f"], hx4x3, 4, 3);
        horizontal_cplx_kernel!(["avx512f"], hx4x4, 4, 4);
        hmicrokernel_cplx_fn_array! {
            [hx1x1, hx1x2, hx1x3, hx1x4,],
            [hx2x1, hx2x2, hx2x3, hx2x4,],
            [hx3x1, hx3x2, hx3x3, hx3x4,],
            [hx4x1, hx4x2, hx4x3, hx4x4,],
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neonfcma {
    pub mod c64 {
        use core::arch::{aarch64::*, asm};
        use core::mem::transmute;

        type T = num_complex::Complex64;
        const N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [transmute(vaddq_f64(transmute(lhs), transmute(rhs)))]
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            [T::new(a[0].re, -a[0].im)]
        }

        #[inline(always)]
        unsafe fn mul_cplx(lhs: Pack, rhs: Pack) -> Pack {
            mul_add_cplx(lhs, rhs, core::mem::zeroed(), false)
        }

        #[inline]
        #[target_feature(enable = "neon,fcma")]
        unsafe fn vcmlaq_0_f64(
            mut acc: float64x2_t,
            lhs: float64x2_t,
            rhs: float64x2_t,
        ) -> float64x2_t {
            asm!(
                "fcmla {0:v}.2d, {1:v}.2d, {2:v}.2d, 0",
                inout(vreg) acc,
                in(vreg) lhs,
                in(vreg) rhs,
                options(pure, nomem, nostack));
            acc
        }

        #[inline]
        #[target_feature(enable = "neon,fcma")]
        unsafe fn vcmlaq_90_f64(
            mut acc: float64x2_t,
            lhs: float64x2_t,
            rhs: float64x2_t,
        ) -> float64x2_t {
            asm!(
                "fcmla {0:v}.2d, {1:v}.2d, {2:v}.2d, 90",
                inout(vreg) acc,
                in(vreg) lhs,
                in(vreg) rhs,
                options(pure, nomem, nostack));
            acc
        }

        #[inline]
        #[target_feature(enable = "neon,fcma")]
        unsafe fn vcmlaq_270_f64(
            mut acc: float64x2_t,
            lhs: float64x2_t,
            rhs: float64x2_t,
        ) -> float64x2_t {
            asm!(
                "fcmla {0:v}.2d, {1:v}.2d, {2:v}.2d, 270",
                inout(vreg) acc,
                in(vreg) lhs,
                in(vreg) rhs,
                options(pure, nomem, nostack));
            acc
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(lhs: Pack, rhs: Pack, acc: Pack, conj_rhs: bool) -> Pack {
            let _ = conj_rhs;
            let lhs = transmute(lhs);
            let rhs = transmute(rhs);
            let acc = transmute(acc);

            if conj_rhs {
                transmute(vcmlaq_270_f64(vcmlaq_0_f64(acc, rhs, lhs), rhs, lhs))
            } else {
                transmute(vcmlaq_90_f64(vcmlaq_0_f64(acc, lhs, rhs), lhs, rhs))
            }
        }

        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x1x1, 1, 1);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x1x2, 1, 2);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x1x3, 1, 3);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x1x4, 1, 4);

        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x2x1, 2, 1);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x2x2, 2, 2);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x2x3, 2, 3);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x2x4, 2, 4);

        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x3x1, 3, 1);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x3x2, 3, 2);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x3x3, 3, 3);
        microkernel_cplx_packed!(["neon,fcma"], 4, cplx_x3x4, 3, 4);

        microkernel_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3, cplx_x1x4,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3, cplx_x2x4,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3, cplx_x3x4,],
        }
        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];
    }
}

/// Apple SME (`FEAT_SME` + `FEAT_SME_F64F64`) microkernel for `c64`.
///
/// Apple Silicon only: this module exists only under
/// `all(target_vendor = "apple", target_arch = "aarch64", feature = "apple-sme")`.
/// SME itself is an Armv9 feature, but only Apple's implementation (`svl == 64`,
/// macOS 26 or newer) has been validated here, so generic aarch64 (Linux) SME is
/// not supported.
///
/// SME has no complex outer product: `FMOPA` takes two vectors of reals, and
/// there is no `za`-form `FCMLA` (the assembler rejects `fcmla za0.d, ...` and
/// there is no complex `*MOPA` mnemonic at all, in `FEAT_SME` or `FEAT_SME2`).
/// A complex block is therefore computed with four real outer products per
/// column block, exactly as Arm's own SME complex GEMM example does it:
///
/// ```text
/// Re += Lr * Rr        Im += Lr * Ri
/// Re += Li * (-Ri)     Im += Li * Rr
/// ```
///
/// Each output column block keeps two `ZA64` tiles, one for the real part and one
/// for the imaginary part, so four tiles hold `MR = 8` complex rows by
/// `NR = 16` complex columns. `Lr`/`Li`/`Rr`/`Ri` are obtained from the
/// interleaved (`re, im`) packed operands with `UZP1`/`UZP2` on 64-bit elements:
/// one `Z` register holds four complex, so two loads plus two unzips give the
/// eight reals and eight imaginaries a `ZA64` tile needs.
///
/// Conjugation is a sign flip of an imaginary part, applied with a predicated
/// `FNEG`: the predicate is all-true or all-false, so no branch is needed in the
/// `k` loop.
#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    feature = "std",
    feature = "apple-sme"
))]
pub mod sme {
    pub mod c64 {
        pub type T = num_complex::Complex64;

        /// Complex elements per `Z` register (four `c64` = 64 bytes).
        pub const N: usize = 4;
        /// `Z` register reads per microkernel row panel: `MR = MR_DIV_N * N = 8`.
        pub const MR_DIV_N: usize = 2;
        /// Complex columns per microkernel invocation: four `ZA64` tiles, two for
        /// the real part and two for the imaginary part.
        pub const NR: usize = 16;

        const MR: usize = MR_DIV_N * N;

        pub const UKR: [[gemm_common::microkernel::MicroKernelFn<T>; NR]; MR_DIV_N] =
            [[ukernel; NR]; MR_DIV_N];

        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];

        // `accum_re[i + j * MR]` / `accum_im[i + j * MR]` receive the real and
        // imaginary parts of `sum_kk lhs[kk * lhs_step + i] * rhs[kk * rhs_step + j]`.
        // `za0`/`za1` hold the real/imaginary parts of columns 0-7, `za2`/`za3` those
        // of columns 8-15; one tile column is 8 contiguous `f64` (= `MR`), so a
        // column spills with one `st1d` and a 64 byte stride.
        core::arch::global_asm!(
            r#"
            .arch_extension sme
            .arch_extension sme-f64f64
            .text
            .globl  _t4a_gemm_c64_sme_ukr
            .p2align 4
            _t4a_gemm_c64_sme_ukr:
                // x0 = k, x1 = lhs, x2 = rhs, x3 = lhs_step, x4 = rhs_step,
                // x5 = accum_re, x6 = accum_im, w7 = conj flags (bit 0 lhs, bit 1 rhs)
                smstart
                // p2 = all true iff conj(lhs), p3 = all true iff conj(rhs)
                pfalse  p2.b
                pfalse  p3.b
                tbz     w7, #0, 10f
                ptrue   p2.d
            10:
                tbz     w7, #1, 11f
                ptrue   p3.d
            11:
                zero    {{za}}
                ptrue   p0.d
                ptrue   p1.d
            1:
                ldr     z0, [x1]
                ldr     z1, [x1, #1, mul vl]
                ldr     z4, [x2]
                ldr     z5, [x2, #1, mul vl]
                uzp1    z2.d, z0.d, z1.d        // Lr: real part of rows 0-7
                uzp2    z3.d, z0.d, z1.d        // Li: imaginary part of rows 0-7
                uzp1    z6.d, z4.d, z5.d        // Rr: real part of columns 0-7
                uzp2    z7.d, z4.d, z5.d        // Ri: imaginary part of columns 0-7
                fneg    z3.d, p2/m, z3.d        // conj(lhs)
                fneg    z7.d, p3/m, z7.d        // conj(rhs)
                fneg    z16.d, p1/m, z7.d       // -Ri
                fmopa   za0.d, p0/m, p1/m, z2.d, z6.d   // Re += Lr * Rr
                fmopa   za0.d, p0/m, p1/m, z3.d, z16.d  // Re += Li * -Ri
                fmopa   za1.d, p0/m, p1/m, z2.d, z7.d   // Im += Lr * Ri
                fmopa   za1.d, p0/m, p1/m, z3.d, z6.d   // Im += Li * Rr
                ldr     z4, [x2, #2, mul vl]
                ldr     z5, [x2, #3, mul vl]
                uzp1    z6.d, z4.d, z5.d        // Rr: columns 8-15
                uzp2    z7.d, z4.d, z5.d        // Ri: columns 8-15
                fneg    z7.d, p3/m, z7.d
                fneg    z16.d, p1/m, z7.d
                fmopa   za2.d, p0/m, p1/m, z2.d, z6.d
                fmopa   za2.d, p0/m, p1/m, z3.d, z16.d
                fmopa   za3.d, p0/m, p1/m, z2.d, z7.d
                fmopa   za3.d, p0/m, p1/m, z3.d, z6.d
                add     x1, x1, x3
                add     x2, x2, x4
                subs    x0, x0, #1
                b.ne    1b
                // spill: x5 walks the real parts, x6 the imaginary parts, x8 the
                // 64 byte column stride, w12 the tile column, x10 counts the 8 columns
                mov     x8, #64
                mov     w12, #0
                mov     x10, #8
            20:
                st1d    {{za0v.d[w12, 0]}}, p0, [x5]
                add     x5, x5, x8
                add     w12, w12, #1
                subs    x10, x10, #1
                b.ne    20b
                mov     w12, #0
                mov     x10, #8
            21:
                st1d    {{za1v.d[w12, 0]}}, p0, [x6]
                add     x6, x6, x8
                add     w12, w12, #1
                subs    x10, x10, #1
                b.ne    21b
                mov     w12, #0
                mov     x10, #8
            22:
                st1d    {{za2v.d[w12, 0]}}, p0, [x5]
                add     x5, x5, x8
                add     w12, w12, #1
                subs    x10, x10, #1
                b.ne    22b
                mov     w12, #0
                mov     x10, #8
            23:
                st1d    {{za3v.d[w12, 0]}}, p0, [x6]
                add     x6, x6, x8
                add     w12, w12, #1
                subs    x10, x10, #1
                b.ne    23b
                smstop
                ret
            "#
        );

        extern "C" {
            /// Computes the `MR x NR` complex accumulator block into `accum_re` and
            /// `accum_im`.
            fn t4a_gemm_c64_sme_ukr(
                k: usize,
                lhs: *const T,
                rhs: *const T,
                lhs_step: usize,
                rhs_step: usize,
                accum_re: *mut f64,
                accum_im: *mut f64,
                conj: usize,
            );
        }

        /// `dst := alpha * dst + beta * (lhs * rhs)` for a `MR x NR` complex block.
        ///
        /// `n` may be anything in `1..=NR` and `m` anything in `1..=MR`; the
        /// dispatcher only passes the trailing partial panels and `pack_operands`
        /// zero-fills the padding, so the accumulators of the unused rows/columns are
        /// computed and spilled but never read back.
        pub unsafe fn ukernel(
            m: usize,
            n: usize,
            k: usize,
            dst: *mut T,
            packed_lhs: *const T,
            packed_rhs: *const T,
            dst_cs: isize,
            dst_rs: isize,
            lhs_cs: isize,
            rhs_rs: isize,
            rhs_cs: isize,
            alpha: T,
            beta: T,
            alpha_status: u8,
            conj_dst: bool,
            conj_lhs: bool,
            conj_rhs: bool,
            _next_lhs: *const T,
        ) {
            debug_assert_eq!(rhs_cs, 1);

            let mut accum_re = [0.0f64; MR * NR];
            let mut accum_im = [0.0f64; MR * NR];

            // `z0`-`z7` and `z16` are clobbered by the callee, which the caller may
            // be using for `alpha`/`beta` (`v0`-`v7` are volatile, `v16`/`z16` is
            // spelled out because it is not part of `clobber_abi("C")`).
            core::arch::asm!(
                "bl {ukr}",
                ukr = sym t4a_gemm_c64_sme_ukr,
                inlateout("x0") k => _,
                in("x1") packed_lhs,
                in("x2") packed_rhs,
                in("x3") (lhs_cs * core::mem::size_of::<T>() as isize) as usize,
                in("x4") (rhs_rs * core::mem::size_of::<T>() as isize) as usize,
                in("x5") accum_re.as_mut_ptr(),
                in("x6") accum_im.as_mut_ptr(),
                in("x7") (conj_lhs as usize) | ((conj_rhs as usize) << 1),
                out("z0") _,
                out("z1") _,
                out("z2") _,
                out("z3") _,
                out("z4") _,
                out("z5") _,
                out("z6") _,
                out("z7") _,
                out("z16") _,
                out("p0") _,
                out("p1") _,
                out("p2") _,
                out("p3") _,
                clobber_abi("C"),
            );

            // dst := alpha * dst + beta * accum
            for j in 0..n {
                for i in 0..m {
                    let dst = dst.offset(i as isize * dst_rs + j as isize * dst_cs);
                    let accum = T::new(accum_re[i + j * MR], accum_im[i + j * MR]);
                    // `conj_dst` applies to the destination, not the product: the
                    // crate's convention is `dst = alpha * conj(dst) + beta * lhs * rhs`.
                    let dst0 = if conj_dst { (*dst).conj() } else { *dst };
                    *dst = match alpha_status {
                        2 => alpha * dst0 + beta * accum,
                        1 => dst0 + beta * accum,
                        _ => beta * accum,
                    };
                }
            }
        }
    }
}
