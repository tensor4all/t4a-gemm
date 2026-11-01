pub mod scalar {
    pub mod f64 {
        type T = f64;
        const N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value]
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] * rhs[0]]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0]]
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        #[inline(always)]
        pub unsafe fn scalar_mul(lhs: T, rhs: T) -> T {
            lhs * rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_add(lhs: T, rhs: T) -> T {
            lhs + rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_mul_add(a: T, b: T, c: T) -> T {
            a * b + c
        }

        microkernel!(, 2, x1x1, 1, 1);
        microkernel!(, 2, x1x2, 1, 2);
        microkernel!(, 2, x1x3, 1, 3);
        microkernel!(, 2, x1x4, 1, 4);

        microkernel!(, 2, x2x1, 2, 1);
        microkernel!(, 2, x2x2, 2, 2);
        microkernel!(, 2, x2x3, 2, 3);
        microkernel!(, 2, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }

        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];
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
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm256_fmadd_pd(transmute(a), transmute(b), transmute(c)))
        }

        #[inline(always)]
        pub unsafe fn scalar_mul(lhs: T, rhs: T) -> T {
            lhs * rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_add(lhs: T, rhs: T) -> T {
            lhs + rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_mul_add(a: T, b: T, c: T) -> T {
            gemm_common::simd::v3_fma(a, b, c)
        }

        static U64_MASKS: [u64x4; 5] = [
            u64x4(0, 0, 0, 0),
            u64x4(!0, 0, 0, 0),
            u64x4(!0, !0, 0, 0),
            u64x4(!0, !0, !0, 0),
            u64x4(!0, !0, !0, !0),
        ];

        #[inline(always)]
        pub unsafe fn partial_load(ptr: *const T, len: usize) -> [T; N] {
            transmute(_mm256_maskload_pd(
                ptr,
                transmute(*(U64_MASKS.as_ptr().add(len))),
            ))
        }

        #[inline(always)]
        pub unsafe fn reduce_sum(x: [T; N]) -> T {
            let x: __m256d = transmute(x);
            let x = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd::<1>(x));
            let hi = transmute(_mm_movehl_ps(transmute(x), transmute(x)));
            let r = _mm_add_sd(x, hi);
            _mm_cvtsd_f64(r)
        }

        microkernel!(["fma"], 2, x1x1, 1, 1);
        microkernel!(["fma"], 2, x1x2, 1, 2);
        microkernel!(["fma"], 2, x1x3, 1, 3);
        microkernel!(["fma"], 2, x1x4, 1, 4);
        microkernel!(["fma"], 2, x1x5, 1, 5);
        microkernel!(["fma"], 2, x1x6, 1, 6);

        microkernel!(["fma"], 2, x2x1, 2, 1);
        microkernel!(["fma"], 2, x2x2, 2, 2);
        microkernel!(["fma"], 2, x2x3, 2, 3);
        microkernel!(["fma"], 2, x2x4, 2, 4);
        microkernel!(["fma"], 2, x2x5, 2, 5);
        microkernel!(["fma"], 2, x2x6, 2, 6);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6,],
        }

        horizontal_kernel!(["fma"], hx1x1, 1, 1);
        horizontal_kernel!(["fma"], hx1x2, 1, 2);
        horizontal_kernel!(["fma"], hx2x1, 2, 1);
        horizontal_kernel!(["fma"], hx2x2, 2, 2);
        hmicrokernel_fn_array! {
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
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm512_fmadd_pd(transmute(a), transmute(b), transmute(c)))
        }

        #[inline(always)]
        pub unsafe fn scalar_mul(lhs: T, rhs: T) -> T {
            lhs * rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_add(lhs: T, rhs: T) -> T {
            lhs + rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_mul_add(a: T, b: T, c: T) -> T {
            gemm_common::simd::v3_fma(a, b, c)
        }

        microkernel!(["avx512f"], 4, x1x1, 1, 1);
        microkernel!(["avx512f"], 4, x1x2, 1, 2);
        microkernel!(["avx512f"], 4, x1x3, 1, 3);
        microkernel!(["avx512f"], 4, x1x4, 1, 4);
        microkernel!(["avx512f"], 4, x1x5, 1, 5);
        microkernel!(["avx512f"], 4, x1x6, 1, 6);

        microkernel!(["avx512f"], 4, x2x1, 2, 1);
        microkernel!(["avx512f"], 4, x2x2, 2, 2);
        microkernel!(["avx512f"], 4, x2x3, 2, 3);
        microkernel!(["avx512f"], 4, x2x4, 2, 4);
        microkernel!(["avx512f"], 4, x2x5, 2, 5);
        microkernel!(["avx512f"], 4, x2x6, 2, 6);

        microkernel!(["avx512f"], 4, x3x1, 3, 1);
        microkernel!(["avx512f"], 4, x3x2, 3, 2);
        microkernel!(["avx512f"], 4, x3x3, 3, 3);
        microkernel!(["avx512f"], 4, x3x4, 3, 4);
        microkernel!(["avx512f"], 4, x3x5, 3, 5);
        microkernel!(["avx512f"], 4, x3x6, 3, 6);

        microkernel!(["avx512f"], 4, x4x1, 4, 1);
        microkernel!(["avx512f"], 4, x4x2, 4, 2);
        microkernel!(["avx512f"], 4, x4x3, 4, 3);
        microkernel!(["avx512f"], 4, x4x4, 4, 4);
        microkernel!(["avx512f"], 4, x4x5, 4, 5);
        microkernel!(["avx512f"], 4, x4x6, 4, 6);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6,],
            [x4x1, x4x2, x4x3, x4x4, x4x5, x4x6,],
        }

        static U64_MASKS: [u8; 9] = [
            0b00000000, //
            0b00000001, //
            0b00000011, //
            0b00000111, //
            0b00001111, //
            0b00011111, //
            0b00111111, //
            0b01111111, //
            0b11111111, //
        ];

        #[inline(always)]
        pub unsafe fn partial_load(ptr: *const T, len: usize) -> [T; N] {
            transmute(_mm512_maskz_loadu_pd(*(U64_MASKS.as_ptr().add(len)), ptr))
        }

        #[inline(always)]
        pub unsafe fn reduce_sum(x: [T; N]) -> T {
            let x = transmute(x);
            let x = _mm256_add_pd(_mm512_castpd512_pd256(x), _mm512_extractf64x4_pd::<1>(x));
            let x = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd::<1>(x));
            let hi = transmute(_mm_movehl_ps(transmute(x), transmute(x)));
            let r = _mm_add_sd(x, hi);
            _mm_cvtsd_f64(r)
        }

        horizontal_kernel!(["avx512f"], hx1x1, 1, 1);
        horizontal_kernel!(["avx512f"], hx1x2, 1, 2);
        horizontal_kernel!(["avx512f"], hx1x3, 1, 3);
        horizontal_kernel!(["avx512f"], hx1x4, 1, 4);
        horizontal_kernel!(["avx512f"], hx2x1, 2, 1);
        horizontal_kernel!(["avx512f"], hx2x2, 2, 2);
        horizontal_kernel!(["avx512f"], hx2x3, 2, 3);
        horizontal_kernel!(["avx512f"], hx2x4, 2, 4);
        horizontal_kernel!(["avx512f"], hx3x1, 3, 1);
        horizontal_kernel!(["avx512f"], hx3x2, 3, 2);
        horizontal_kernel!(["avx512f"], hx3x3, 3, 3);
        horizontal_kernel!(["avx512f"], hx3x4, 3, 4);
        horizontal_kernel!(["avx512f"], hx4x1, 4, 1);
        horizontal_kernel!(["avx512f"], hx4x2, 4, 2);
        horizontal_kernel!(["avx512f"], hx4x3, 4, 3);
        horizontal_kernel!(["avx512f"], hx4x4, 4, 4);
        hmicrokernel_fn_array! {
            [hx1x1, hx1x2, hx1x3, hx1x4,],
            [hx2x1, hx2x2, hx2x3, hx2x4,],
            [hx3x1, hx3x2, hx3x3, hx3x4,],
            [hx4x1, hx4x2, hx4x3, hx4x4,],
        }
    }
}

#[allow(dead_code)]
mod v128_common {
    pub mod f64 {
        pub type T = f64;
        pub const N: usize = 2;
        pub type Pack = [T; N];

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value]
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    pub mod f64 {
        use super::super::v128_common::f64::*;
        use core::arch::aarch64::*;
        use core::mem::transmute;

        #[cfg(miri)]
        unsafe fn vfmaq_f64(c: float64x2_t, a: float64x2_t, b: float64x2_t) -> float64x2_t {
            let c: [f64; 2] = transmute(c);
            let a: [f64; 2] = transmute(a);
            let b: [f64; 2] = transmute(b);

            transmute([
                f64::mul_add(a[0], b[0], c[0]),
                f64::mul_add(a[1], b[1], c[1]),
            ])
        }

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f64(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f64(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_f64(transmute(c), transmute(a), transmute(b)))
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f64::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        #[inline(always)]
        pub unsafe fn scalar_mul(lhs: T, rhs: T) -> T {
            lhs * rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_add(lhs: T, rhs: T) -> T {
            lhs + rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_mul_add(a: T, b: T, c: T) -> T {
            gemm_common::simd::neon_fma(a, b, c)
        }

        microkernel!(["neon"], 2, x1x1, 1, 1);
        microkernel!(["neon"], 2, x1x2, 1, 2, 1, 2);
        microkernel!(["neon"], 2, x1x3, 1, 3);
        microkernel!(["neon"], 2, x1x4, 1, 4, 2, 2);

        microkernel!(["neon"], 2, x2x1, 2, 1);
        microkernel!(["neon"], 2, x2x2, 2, 2, 1, 2);
        microkernel!(["neon"], 2, x2x3, 2, 3);
        microkernel!(["neon"], 2, x2x4, 2, 4, 2, 2);

        microkernel!(["neon"], 2, x3x1, 3, 1);
        microkernel!(["neon"], 2, x3x2, 3, 2, 1, 2);
        microkernel!(["neon"], 2, x3x3, 3, 3);
        microkernel!(["neon"], 2, x3x4, 3, 4, 2, 2);

        microkernel!(["neon"], 2, x4x1, 4, 1);
        microkernel!(["neon"], 2, x4x2, 4, 2, 1, 2);
        microkernel!(["neon"], 2, x4x3, 4, 3);
        microkernel!(["neon"], 2, x4x4, 4, 4, 2, 2);

        #[inline(always)]
        pub unsafe fn load<const MR_DIV_N: usize>(dst: *mut Pack, ptr: *const f64) {
            match MR_DIV_N {
                1 => *(dst as *mut [Pack; 1]) = transmute(vld1q_f64(ptr)),
                2 => *(dst as *mut [Pack; 2]) = transmute(vld1q_f64_x2(ptr)),
                3 => *(dst as *mut [Pack; 3]) = transmute(vld1q_f64_x3(ptr)),
                4 => *(dst as *mut [Pack; 4]) = transmute(vld1q_f64_x4(ptr)),
                _ => unreachable!(),
            }
        }

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, ],
            [x2x1, x2x2, x2x3, x2x4, ],
            [x3x1, x3x2, x3x3, x3x4, ],
            [x4x1, x4x2, x4x3, x4x4, ],
        }

        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];
    }
}

#[cfg(target_arch = "aarch64")]
pub mod amx {
    pub mod f64 {
        pub type T = f64;
        pub const N: usize = 8;

        microkernel_amx!(f64, ["neon"], 4, x1x8, 1, 8, 1, 8);
        microkernel_amx!(f64, ["neon"], 4, x1x16, 1, 16, 2, 8);
        microkernel_amx!(f64, ["neon"], 4, x2x8, 2, 8, 1, 8);
        microkernel_amx!(f64, ["neon"], 4, x2x16, 2, 16, 2, 8);

        microkernel_fn_array! {
            [
                x1x8,x1x8,x1x8,x1x8,x1x8,x1x8,x1x8,x1x8,
                x1x16,x1x16,x1x16,x1x16,x1x16,x1x16,x1x16,x1x16,
            ],
            [
                x2x8,x2x8,x2x8,x2x8,x2x8,x2x8,x2x8,x2x8,
                x2x16,x2x16,x2x16,x2x16,x2x16,x2x16,x2x16,x2x16,
            ],
        }

        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];
    }
}

/// Apple `SME` microkernel for `f64`.
///
/// `fmopa za_t.d, p0/m, p1/m, z0.d, z_m.d` computes
/// `ZA_t[i][j] += z0[i] * z_m[j]`, i.e. with `z0 = A[..][kk]` and `z_m = B[kk][8t..]`
/// it accumulates `ZA_t[i][j] += A[i][kk] * B[kk][8t + j]`, which is a transposed
/// outer product: `ZA_t = A_panel^T @ B_panel`. Since `ZA` holds `svl/64 x svl/64`
/// elements and `svl == 64` bytes here, one tile is exactly an `8 x 8` block of
/// `dst`, with `i` running over `MR = 8` rows and `j` over 8 of the `NR = 32`
/// columns.
///
/// Layout produced by `pack_operands` matches the instructions exactly:
/// `MR = 8` contiguous `f64` per `k` step for the lhs (= one `Z` register) and
/// `NR = 32` contiguous `f64` per `k` step for the rhs (= four `Z` registers), so
/// the inner loop is four loads and four outer products per `k`.
///
/// The instruction words are emitted with `.inst` because
/// `#[target_feature(enable = "sme")]` and the SME intrinsics are still unstable
/// (rust-lang/rust#150244); every word is annotated with the instruction it encodes.
/// `ZA` and streaming mode are enabled per call with `smstart`/`smstop`, which is
/// what makes this safe to use from arbitrary threads: macOS saves the `ZA` and
/// streaming mode state on context switch (verified on macOS 26 / M5 Max).
#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    feature = "std",
    feature = "sme"
))]
pub mod sme {
    pub mod f64 {
        pub type T = f64;

        /// `f64` lanes per `Z` register (as wide as one `ZA64` tile row).
        pub const N: usize = 8;
        /// `Z` register writes per microkernel row panel: `MR = MR_DIV_N * N = 8`.
        pub const MR_DIV_N: usize = 1;
        /// Columns per microkernel invocation: four `ZA64` tiles.
        pub const NR: usize = 32;

        const MR: usize = MR_DIV_N * N;

        pub const UKR: [[gemm_common::microkernel::MicroKernelFn<T>; NR]; MR_DIV_N] =
            [[ukernel; NR]; MR_DIV_N];

        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];

        // The accumulator block: `accum[i + j * MR]` receives
        // `sum_kk lhs[kk * lhs_step + i] * rhs[kk * rhs_step + j]`, computed by four
        // `za64` outer products.
        //
        // Defined in assembly (outside the inline-asm constraints) because the body
        // relies on ZA, on the streaming mode state `smstart` just enabled, on
        // all-active predicate registers and on registers `x0`-`x7`/`x12` only; a
        // real function keeps that out of the compiler's register allocation.
        core::arch::global_asm!(
            r#"
            .globl  _t4a_gemm_f64_sme_ukr
            .p2align 4
            _t4a_gemm_f64_sme_ukr:
                // x0 = k, x1 = lhs, x2 = rhs, x3 = lhs_step, x4 = rhs_step, x5 = accum
                .inst 0xd503477f            // smstart
                .inst 0xc00800ff            // zero {{za}}
                .inst 0x25d8e3e0            // ptrue p0.d
                .inst 0x25d8e3e1            // ptrue p1.d
            1:
                .inst 0x85804020            // ldr z0, [x1]
                .inst 0x85804041            // ldr z1, [x2]
                .inst 0x85804442            // ldr z2, [x2, #1, mul vl]
                .inst 0x85804843            // ldr z3, [x2, #2, mul vl]
                .inst 0x85804c44            // ldr z4, [x2, #3, mul vl]
                .inst 0x80c12000            // fmopa za0.d, p0/m, p1/m, z0.d, z1.d
                .inst 0x80c22001            // fmopa za1.d, p0/m, p1/m, z0.d, z2.d
                .inst 0x80c32002            // fmopa za2.d, p0/m, p1/m, z0.d, z3.d
                .inst 0x80c42003            // fmopa za3.d, p0/m, p1/m, z0.d, z4.d
                add     x1, x1, x3
                add     x2, x2, x4
                subs    x0, x0, #1
                b.ne    1b
                // spill one column (= MR f64) at a time; x5 walks the tile,
                // x6 is the column stride, x7 counts the 8 columns, w12 is the slice
                mov     x6, #64
                mov     w12, #0
                mov     x7, #8
            2:
                .inst 0xe0ff80a0            // st1d {{za0v.d[w12, 0]}}, p0, [x5]
                add     x5, x5, x6
                add     w12, w12, #1
                subs    x7, x7, #1
                b.ne    2b
                mov     w12, #0
                mov     x7, #8
            3:
                .inst 0xe0ff80a2            // st1d {{za1v.d[w12, 0]}}, p0, [x5]
                add     x5, x5, x6
                add     w12, w12, #1
                subs    x7, x7, #1
                b.ne    3b
                mov     w12, #0
                mov     x7, #8
            4:
                .inst 0xe0ff80a4            // st1d {{za2v.d[w12, 0]}}, p0, [x5]
                add     x5, x5, x6
                add     w12, w12, #1
                subs    x7, x7, #1
                b.ne    4b
                mov     w12, #0
                mov     x7, #8
            5:
                .inst 0xe0ff80a6            // st1d {{za3v.d[w12, 0]}}, p0, [x5]
                add     x5, x5, x6
                add     w12, w12, #1
                subs    x7, x7, #1
                b.ne    5b
                .inst 0xd503467f            // smstop
                ret
            "#
        );

        extern "C" {
            /// Computes the `MR x NR` accumulator block into `accum`.
            fn t4a_gemm_f64_sme_ukr(
                k: usize,
                lhs: *const T,
                rhs: *const T,
                lhs_step: usize,
                rhs_step: usize,
                accum: *mut T,
            );
        }

        /// `dst := alpha * dst + beta * (lhs * rhs)` for a `MR x NR` block.
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
            _conj_dst: bool,
            _conj_lhs: bool,
            _conj_rhs: bool,
            _next_lhs: *const T,
        ) {
            debug_assert_eq!(rhs_cs, 1);

            let mut accum = [0.0 as T; MR * NR];

            // The callee clobbers the volatile SIMD registers (`z0`-`z4` alias
            // `v0`-`v4`, which the caller uses for `alpha`/`beta`), so the call goes
            // through `asm!` with an explicit clobber list instead of a plain
            // `extern "C"` call.
            core::arch::asm!(
                "bl {ukr}",
                ukr = sym t4a_gemm_f64_sme_ukr,
                inlateout("x0") k => _,
                in("x1") packed_lhs,
                in("x2") packed_rhs,
                in("x3") (lhs_cs * core::mem::size_of::<T>() as isize) as usize,
                in("x4") (rhs_rs * core::mem::size_of::<T>() as isize) as usize,
                in("x5") accum.as_mut_ptr(),
                out("z0") _,
                out("z1") _,
                out("z2") _,
                out("z3") _,
                out("z4") _,
                out("p0") _,
                out("p1") _,
                clobber_abi("C"),
            );

            // dst := alpha * dst + beta * accum
            match alpha_status {
                2 => {
                    for j in 0..n {
                        for i in 0..m {
                            let dst = dst.offset(i as isize * dst_rs + j as isize * dst_cs);
                            *dst = alpha * *dst + beta * accum[i + j * MR];
                        }
                    }
                }
                1 => {
                    for j in 0..n {
                        for i in 0..m {
                            let dst = dst.offset(i as isize * dst_rs + j as isize * dst_cs);
                            *dst += beta * accum[i + j * MR];
                        }
                    }
                }
                _ => {
                    for j in 0..n {
                        for i in 0..m {
                            let dst = dst.offset(i as isize * dst_rs + j as isize * dst_cs);
                            *dst = beta * accum[i + j * MR];
                        }
                    }
                }
            }
        }

    }
}

#[cfg(target_arch = "wasm32")]
pub mod simd128 {
    pub mod f64 {
        use super::super::v128_common::f64::*;
        use core::arch::wasm32::*;
        use core::mem::transmute;

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(f64x2_mul(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(f64x2_add(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(c, mul(a, b))
        }

        #[inline(always)]
        pub unsafe fn scalar_mul(lhs: T, rhs: T) -> T {
            lhs * rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_add(lhs: T, rhs: T) -> T {
            lhs + rhs
        }

        #[inline(always)]
        pub unsafe fn scalar_mul_add(a: T, b: T, c: T) -> T {
            a * b + c
        }

        microkernel!(["simd128"], 2, x1x1, 1, 1);
        microkernel!(["simd128"], 2, x1x2, 1, 2);
        microkernel!(["simd128"], 2, x1x3, 1, 3);
        microkernel!(["simd128"], 2, x1x4, 1, 4);

        microkernel!(["simd128"], 2, x2x1, 2, 1);
        microkernel!(["simd128"], 2, x2x2, 2, 2);
        microkernel!(["simd128"], 2, x2x3, 2, 3);
        microkernel!(["simd128"], 2, x2x4, 2, 4);

        microkernel!(["simd128"], 2, x3x1, 3, 1);
        microkernel!(["simd128"], 2, x3x2, 3, 2);
        microkernel!(["simd128"], 2, x3x3, 3, 3);
        microkernel!(["simd128"], 2, x3x4, 3, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
            [x3x1, x3x2, x3x3, x3x4,],
        }

        pub const H_M: usize = 0;
        pub const H_N: usize = 0;
        pub const H_UKR: [[gemm_common::microkernel::HMicroKernelFn<T>; H_N]; H_M] = [];
    }
}
