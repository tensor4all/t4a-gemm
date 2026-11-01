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
/// Apple Silicon only: this module exists only under
/// `all(target_vendor = "apple", target_arch = "aarch64", feature = "apple-sme")`.
/// SME itself is an Armv9 feature, but only Apple's implementation (`svl == 64`,
/// macOS 26 or newer) has been validated here, so generic aarch64 (Linux) SME is
/// not supported.
///
/// `fmopa za_t.d, p0/m, p1/m, z0.d, z_m.d` computes
/// `ZA_t[i][j] += z0[i] * z_m[j]`, i.e. with `z0 = A[..][kk]` and `z_m = B[kk][8c..]`
/// it accumulates `ZA_t[i][j] += A[i][kk] * B[kk][8c + j]`, which is a transposed
/// outer product: `ZA_t = A_panel^T @ B_panel`. Since `ZA` holds `svl/64 x svl/64`
/// elements and `svl == 64` bytes here, one tile is exactly an `8 x 8` block of
/// `dst`. The four `za64` tiles are laid out as a `2 x 2` grid of tiles, so one
/// invocation computes an `MR = 16` by `NR = 16` block: `za0`/`za1` hold rows
/// `0..8` (columns `0..8` and `8..16`), `za2`/`za3` rows `8..16`.
///
/// `MR = 16` (not 8) is deliberate: one column of the accumulator block is then
/// `16 * 8 = 128` bytes, i.e. exactly one cache line, and the two 64 byte `st1d`
/// halves of it are adjacent. With `MR = 8` every `st1d` writes only the low half
/// of a line.
///
/// Layout produced by `pack_operands` matches the instructions exactly:
/// `MR = 16` contiguous `f64` per `k` step for the lhs (= two `Z` registers) and
/// `NR = 16` contiguous `f64` per `k` step for the rhs (= two `Z` registers), so
/// the inner loop is four loads and four outer products per `k`, and per `k` it
/// loads `16 + 16 = 32` `f64` for `16 x 16` multiply-adds instead of the
/// `8 + 32 = 40` `f64` an `8 x 32` block needs.
///
/// The epilogue (`dst := alpha * dst + beta * accum`) for a full block whose rows
/// are contiguous in `dst` (`dst_rs == 1`) is fused into the same streaming-mode
/// region and done with SVE `ld1d`/`fmul`/`fmla`/`st1d` on whole 8 element
/// vectors, instead of the scalar read-modify-write loop. That is what makes small
/// `k` fast: with `kc == k` the epilogue runs once per block and costs a large
/// fraction of the `k` loop when `k` is short. Trailing panels (`m < MR`,
/// `n < NR`) and a `dst` that is contiguous along the other axis keep the scalar
/// loop, which needs no extra code path for partial vectors.
///
/// The instruction mnemonics are only accepted because of the `.arch` line: the
/// SME intrinsics and `#[target_feature(enable = "sme")]` are still unstable
/// (rust-lang/rust#150244), so the kernel is written as assembly. `ZA` and
/// streaming mode are enabled per call with `smstart`/`smstop`, which is what
/// makes this safe to use from arbitrary threads: macOS saves the `ZA` and
/// streaming mode state on context switch (verified on macOS 26 / M5 Max).
#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    feature = "std",
    feature = "apple-sme"
))]
pub mod sme {
    pub mod f64 {
        pub type T = f64;

        /// `f64` lanes per `Z` register (as wide as one `ZA64` tile row).
        pub const N: usize = 8;
        /// `Z` register writes per microkernel row panel: `MR = MR_DIV_N * N = 16`.
        pub const MR_DIV_N: usize = 2;
        /// Columns per microkernel invocation: four `ZA64` tiles.
        pub const NR: usize = 16;

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
        // all-active predicate registers and on registers `x0`-`x19` only; a
        // real function keeps that out of the compiler's register allocation.
        //
        // `x8` is the `alpha`/`beta` status (`0`: `dst = beta * accum`,
        // `1`: `dst += beta * accum`, `2`: `dst = alpha * dst + beta * accum`) and
        // `x9` tells the kernel whether the Rust side still has to run the epilogue
        // (`0`) or the fused vector epilogue already did (`1`).
        core::arch::global_asm!(
            r#"
            .arch   armv9-a+sme+sme-f64f64
            .globl  _t4a_gemm_f64_sme_ukr
            .p2align 4
            _t4a_gemm_f64_sme_ukr:
                // x0 = k, x1 = lhs, x2 = rhs, x3 = lhs_step, x4 = rhs_step,
                // x5 = accum, x6 = dst, x7 = dst_cs (bytes), x8 = alpha status,
                // x9 = fused epilogue, x10 = alpha bits, x11 = beta bits
                smstart
                zero    {{za}}
                ptrue   p0.d
                ptrue   p1.d
                mov     x17, x5
            1:
                ldr     z0, [x1]                // A rows 0..8
                ldr     z1, [x1, #1, mul vl]    // A rows 8..16
                ldr     z2, [x2]                // B cols 0..8
                ldr     z3, [x2, #1, mul vl]    // B cols 8..16
                fmopa   za0.d, p0/m, p1/m, z0.d, z2.d
                fmopa   za1.d, p0/m, p1/m, z0.d, z3.d
                fmopa   za2.d, p0/m, p1/m, z1.d, z2.d
                fmopa   za3.d, p0/m, p1/m, z1.d, z3.d
                add     x1, x1, x3
                add     x2, x2, x4
                subs    x0, x0, #1
                b.ne    1b
                // spill one column (= MR = 16 f64 = 128 B = one cache line) at a
                // time: za0/za2 fill the two 64 B halves of that line, za1/za3 the
                // line 1024 B further (columns 8..16 of the block)
                add     x12, x5, #64
                add     x13, x5, #1024
                add     x14, x13, #64
                mov     w15, #0
                mov     w16, #8
            2:
                st1d    {{za0v.d[w15, 0]}}, p0, [x5]
                st1d    {{za2v.d[w15, 0]}}, p0, [x12]
                st1d    {{za1v.d[w15, 0]}}, p0, [x13]
                st1d    {{za3v.d[w15, 0]}}, p0, [x14]
                add     x5, x5, #128
                add     x12, x12, #128
                add     x13, x13, #128
                add     x14, x14, #128
                add     w15, w15, #1
                subs    w16, w16, #1
                b.ne    2b
                cbz     x9, 9f                  // no fused epilogue: Rust does it
                fmov    d0, x10
                fmov    d1, x11
                dup     z14.d, z0.d[0]          // alpha
                dup     z15.d, z1.d[0]          // beta
                mov     w16, #16                // NR columns of the block
                cmp     x8, #0
                b.eq    6f
                cmp     x8, #1
                b.eq    5f
                // status 2: dst = alpha * dst + beta * accum
            3:
                ld1d    z4.d, p0/z, [x17]
                ld1d    z5.d, p0/z, [x17, #1, mul vl]
                ld1d    z6.d, p0/z, [x6]
                ld1d    z7.d, p0/z, [x6, #1, mul vl]
                fmul    z6.d, p0/m, z6.d, z14.d
                fmul    z7.d, p0/m, z7.d, z14.d
                fmla    z6.d, p0/m, z4.d, z15.d
                fmla    z7.d, p0/m, z5.d, z15.d
                st1d    z6.d, p0, [x6]
                st1d    z7.d, p0, [x6, #1, mul vl]
                add     x17, x17, #128
                add     x6, x6, x7
                subs    w16, w16, #1
                b.ne    3b
                b       9f
                // status 1: dst += beta * accum
            5:
                ld1d    z4.d, p0/z, [x17]
                ld1d    z5.d, p0/z, [x17, #1, mul vl]
                ld1d    z6.d, p0/z, [x6]
                ld1d    z7.d, p0/z, [x6, #1, mul vl]
                fmla    z6.d, p0/m, z4.d, z15.d
                fmla    z7.d, p0/m, z5.d, z15.d
                st1d    z6.d, p0, [x6]
                st1d    z7.d, p0, [x6, #1, mul vl]
                add     x17, x17, #128
                add     x6, x6, x7
                subs    w16, w16, #1
                b.ne    5b
                b       9f
                // status 0: dst = beta * accum (dst is not read)
            6:
                ld1d    z4.d, p0/z, [x17]
                ld1d    z5.d, p0/z, [x17, #1, mul vl]
                fmul    z4.d, p0/m, z4.d, z15.d
                fmul    z5.d, p0/m, z5.d, z15.d
                st1d    z4.d, p0, [x6]
                st1d    z5.d, p0, [x6, #1, mul vl]
                add     x17, x17, #128
                add     x6, x6, x7
                subs    w16, w16, #1
                b.ne    6b
            9:
                smstop
                ret
            "#
        );

        extern "C" {
            /// Computes the `MR x NR` accumulator block into `accum` and, when
            /// `fused != 0`, also applies `dst := alpha * dst + beta * accum`.
            fn t4a_gemm_f64_sme_ukr(
                k: usize,
                lhs: *const T,
                rhs: *const T,
                lhs_step: usize,
                rhs_step: usize,
                accum: *mut T,
                dst: *mut T,
                dst_cs: usize,
                alpha_status: usize,
                fused: usize,
                alpha: u64,
                beta: u64,
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

            // The fused epilogue only handles a whole block whose rows are
            // contiguous in `dst`; everything else falls back to the scalar loop
            // below, which needs no per-element bounds handling.
            let fused = (m == MR && n == NR && dst_rs == 1) as usize;

            // The callee clobbers the volatile SIMD registers (`z0`-`z7` alias
            // `v0`-`v7`, which the caller uses for `alpha`/`beta`), so the call goes
            // through `asm!` with an explicit clobber list instead of a plain
            // `extern "C"` call. `alpha`/`beta` travel as bit patterns in `x10`/`x11`
            // because `z0` is both a clobber and the register the broadcast reads.
            core::arch::asm!(
                "bl {ukr}",
                ukr = sym t4a_gemm_f64_sme_ukr,
                inlateout("x0") k => _,
                in("x1") packed_lhs,
                in("x2") packed_rhs,
                in("x3") (lhs_cs * core::mem::size_of::<T>() as isize) as usize,
                in("x4") (rhs_rs * core::mem::size_of::<T>() as isize) as usize,
                in("x5") accum.as_mut_ptr(),
                in("x6") dst,
                in("x7") (dst_cs * core::mem::size_of::<T>() as isize) as usize,
                in("x8") alpha_status as usize,
                in("x9") fused,
                in("x10") alpha.to_bits(),
                in("x11") beta.to_bits(),
                out("z0") _,
                out("z1") _,
                out("z2") _,
                out("z3") _,
                out("z4") _,
                out("z5") _,
                out("z6") _,
                out("z7") _,
                out("z14") _,
                out("z15") _,
                out("p0") _,
                out("p1") _,
                clobber_abi("C"),
            );

            if fused != 0 {
                return;
            }

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
