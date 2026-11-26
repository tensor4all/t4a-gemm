pub mod f64 {
    type T = f64;
    gemm_common::gemm_def!(f64, 1, get, gemm_common::sme_thresholds::F64);

    #[doc(hidden)]
    pub mod mixed {
        use gemm_common::{dyn_stack::MemStack, pack_operands::PackOperand, simd::*, Parallelism};

        type GemmTy<Lhs, Rhs> = unsafe fn(
            usize,
            usize,
            usize,
            *mut f64,
            isize,
            isize,
            *const Lhs,
            isize,
            isize,
            *const Rhs,
            isize,
            isize,
            f64,
            f64,
            Parallelism,
            Option<&mut MemStack>,
        );

        macro_rules! mixed_kernel {
            ($module:ident, $simd:ident, $N:expr, $requires_packed_rhs:expr) => {
                mod $module {
                    use super::*;
                    use crate::microkernel::$module::f64::*;
                    const N: usize = $N;

                    #[allow(clippy::too_many_arguments)]
                    pub unsafe fn gemm_basic<Lhs: PackOperand<f64>, Rhs: PackOperand<f64>>(
                        m: usize,
                        n: usize,
                        k: usize,
                        dst: *mut f64,
                        dst_cs: isize,
                        dst_rs: isize,
                        lhs: *const Lhs,
                        lhs_cs: isize,
                        lhs_rs: isize,
                        rhs: *const Rhs,
                        rhs_cs: isize,
                        rhs_rs: isize,
                        alpha: f64,
                        beta: f64,
                        parallelism: Parallelism,
                        stack: Option<&mut MemStack>,
                    ) {
                        gemm_common::gemm::gemm_blocked_generic_with_stack::<
                            _,
                            Lhs,
                            Rhs,
                            f64,
                            N,
                            { MR_DIV_N * N },
                            NR,
                            MR_DIV_N,
                        >(
                            <$simd as MixedSimd<f64, f64, f64, f64>>::try_new().unwrap(),
                            m,
                            n,
                            k,
                            dst,
                            dst_cs,
                            dst_rs,
                            lhs,
                            lhs_cs,
                            lhs_rs,
                            rhs,
                            rhs_cs,
                            rhs_rs,
                            alpha,
                            beta,
                            false,
                            false,
                            false,
                            &UKR,
                            $requires_packed_rhs,
                            parallelism,
                            stack,
                        )
                    }
                }
            };
        }

        mixed_kernel!(scalar, Scalar, 1, false);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mixed_kernel!(fma, V3, 4, false);
        #[cfg(all(feature = "x86-v4", any(target_arch = "x86", target_arch = "x86_64")))]
        mixed_kernel!(avx512f, V4, 8, false);
        #[cfg(target_arch = "aarch64")]
        mixed_kernel!(neon, Scalar, 2, false);
        #[cfg(all(
            target_arch = "aarch64",
            target_vendor = "apple",
            feature = "apple-sme"
        ))]
        mixed_kernel!(sme, Scalar, 8, true);
        #[cfg(all(target_arch = "aarch64", feature = "experimental-apple-amx"))]
        mixed_kernel!(amx, Scalar, 8, true);
        #[cfg(target_arch = "wasm32")]
        mixed_kernel!(simd128, Scalar, 2, false);

        #[cfg(all(
            target_arch = "aarch64",
            target_vendor = "apple",
            feature = "apple-sme"
        ))]
        mod sme_gate {
            use super::*;

            #[allow(clippy::too_many_arguments)]
            pub unsafe fn gemm_basic<Lhs: PackOperand<f64>, Rhs: PackOperand<f64>>(
                m: usize,
                n: usize,
                k: usize,
                dst: *mut f64,
                dst_cs: isize,
                dst_rs: isize,
                lhs: *const Lhs,
                lhs_cs: isize,
                lhs_rs: isize,
                rhs: *const Rhs,
                rhs_cs: isize,
                rhs_rs: isize,
                alpha: f64,
                beta: f64,
                parallelism: Parallelism,
                stack: Option<&mut MemStack>,
            ) {
                if gemm_common::sme_thresholds::F64.wants_sme(m, n, k) {
                    super::sme::gemm_basic(
                        m,
                        n,
                        k,
                        dst,
                        dst_cs,
                        dst_rs,
                        lhs,
                        lhs_cs,
                        lhs_rs,
                        rhs,
                        rhs_cs,
                        rhs_rs,
                        alpha,
                        beta,
                        parallelism,
                        stack,
                    )
                } else {
                    super::neon::gemm_basic(
                        m,
                        n,
                        k,
                        dst,
                        dst_cs,
                        dst_rs,
                        lhs,
                        lhs_cs,
                        lhs_rs,
                        rhs,
                        rhs_cs,
                        rhs_rs,
                        alpha,
                        beta,
                        parallelism,
                        stack,
                    )
                }
            }
        }

        macro_rules! select_mixed {
            () => {{
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    #[cfg(feature = "x86-v4")]
                    if gemm_common::feature_detected!("avx512f") {
                        return avx512f::gemm_basic::<Lhs, Rhs>;
                    }
                    if gemm_common::feature_detected!("fma") {
                        fma::gemm_basic::<Lhs, Rhs>
                    } else {
                        scalar::gemm_basic::<Lhs, Rhs>
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if gemm_common::feature_detected!("neon") {
                        #[cfg(all(
                            target_arch = "aarch64",
                            target_vendor = "apple",
                            feature = "apple-sme"
                        ))]
                        if gemm_common::cache::HasAppleSme::get() {
                            return sme_gate::gemm_basic::<Lhs, Rhs>;
                        }
                        #[cfg(feature = "experimental-apple-amx")]
                        if gemm_common::cache::HasAmx::get() {
                            return amx::gemm_basic::<Lhs, Rhs>;
                        }
                        neon::gemm_basic::<Lhs, Rhs>
                    } else {
                        scalar::gemm_basic::<Lhs, Rhs>
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if gemm_common::feature_detected!("simd128") {
                        simd128::gemm_basic::<Lhs, Rhs>
                    } else {
                        scalar::gemm_basic::<Lhs, Rhs>
                    }
                }
                #[cfg(not(any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "wasm32",
                )))]
                {
                    scalar::gemm_basic::<Lhs, Rhs>
                }
            }};
        }

        // INVARIANT: no-std feature detection mirrors `gemm_def!`'s cfg-based FMA fallback;
        // the destination crate cannot declare that implementation detail in check-cfg.
        #[allow(unexpected_cfgs)]
        fn init<Lhs: PackOperand<f64>, Rhs: PackOperand<f64>>() -> GemmTy<Lhs, Rhs> {
            select_mixed!()
        }

        fn get<Lhs: PackOperand<f64>, Rhs: PackOperand<f64>>(
            slot: &core::sync::atomic::AtomicPtr<()>,
        ) -> GemmTy<Lhs, Rhs> {
            let mut f = slot.load(core::sync::atomic::Ordering::Relaxed);
            if f.is_null() {
                f = init::<Lhs, Rhs>() as *mut ();
                slot.store(f, core::sync::atomic::Ordering::Relaxed);
            }
            unsafe { core::mem::transmute(f) }
        }

        static F32_F32_PTR: core::sync::atomic::AtomicPtr<()> =
            core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());
        static F32_F64_PTR: core::sync::atomic::AtomicPtr<()> =
            core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

        pub fn get_f32_f32_with_stack_fn() -> GemmTy<f32, f32> {
            get(&F32_F32_PTR)
        }

        pub fn get_f32_f64_with_stack_fn() -> GemmTy<f32, f64> {
            get(&F32_F64_PTR)
        }

        pub fn gemm_scratch(m: usize, n: usize, k: usize) -> gemm_common::dyn_stack::StackReq {
            super::gemm_scratch(m, n, k)
        }
    }
}
