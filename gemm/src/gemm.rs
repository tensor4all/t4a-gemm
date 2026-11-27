use crate::Parallelism;
use core::any::TypeId;

#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex32;
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;
#[cfg(feature = "f16")]
#[allow(non_camel_case_types)]
pub type f16 = gemm_f16::f16;

mod mixed_sealed {
    pub trait Sealed {}
    impl Sealed for (f32, f32) {}
    impl Sealed for (f32, f64) {}
}

/// Input storage type pairs supported by [`gemm_mixed`], with `f64` destination and arithmetic.
///
/// The trait is sealed. An unsupported pair is rejected at compile time.
///
/// # Examples
///
/// ```
/// use gemm::MixedRealGemmToF64;
/// fn accepts_supported_pair<T: MixedRealGemmToF64>() {}
/// accepts_supported_pair::<(f32, f32)>();
/// accepts_supported_pair::<(f32, f64)>();
/// ```
///
/// An unsupported pair fails during type checking and reports the supported alternatives:
///
/// ```compile_fail
/// use gemm::MixedRealGemmToF64;
/// fn accepts_supported_pair<T: MixedRealGemmToF64>() {}
/// accepts_supported_pair::<(f64, f32)>();
/// ```
#[diagnostic::on_unimplemented(
    message = "unsupported mixed GEMM operand pair",
    note = "supported pairs are `f32 × f32 → f64` and `f32 × f64 → f64`; see `gemm_mixed` documentation"
)]
pub trait MixedRealGemmToF64: mixed_sealed::Sealed + Copy + Send + Sync + 'static {}
impl MixedRealGemmToF64 for (f32, f32) {}
impl MixedRealGemmToF64 for (f32, f64) {}

unsafe fn gemm_dispatch<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
    stack: Option<&mut dyn_stack::MemStack>,
) {
    #[cfg(feature = "f16")]
    if TypeId::of::<T>() == TypeId::of::<f16>() {
        assert!(
            stack.is_none(),
            "gemm_with_stack does not support f16: its kernels allocate their own buffers"
        );
        return gemm_f16::gemm::f16::get_gemm_fn()(
            m,
            n,
            k,
            dst as *mut f16,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f16,
            lhs_cs,
            lhs_rs,
            rhs as *mut f16,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f16),
            *(&beta as *const T as *const f16),
            false,
            false,
            false,
            parallelism,
        );
    }

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        gemm_f64::gemm::f64::get_gemm_with_stack_fn()(
            m,
            n,
            k,
            dst as *mut f64,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f64,
            lhs_cs,
            lhs_rs,
            rhs as *mut f64,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f64),
            *(&beta as *const T as *const f64),
            false,
            false,
            false,
            parallelism,
            stack,
        )
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        gemm_f32::gemm::f32::get_gemm_with_stack_fn()(
            m,
            n,
            k,
            dst as *mut f32,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f32,
            lhs_cs,
            lhs_rs,
            rhs as *mut f32,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f32),
            *(&beta as *const T as *const f32),
            false,
            false,
            false,
            parallelism,
            stack,
        )
    } else if TypeId::of::<T>() == TypeId::of::<c64>() {
        gemm_c64::gemm::f64::get_gemm_with_stack_fn()(
            m,
            n,
            k,
            dst as *mut c64,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut c64,
            lhs_cs,
            lhs_rs,
            rhs as *mut c64,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const c64),
            *(&beta as *const T as *const c64),
            conj_dst,
            conj_lhs,
            conj_rhs,
            parallelism,
            stack,
        )
    } else if TypeId::of::<T>() == TypeId::of::<c32>() {
        gemm_c32::gemm::f32::get_gemm_with_stack_fn()(
            m,
            n,
            k,
            dst as *mut c32,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut c32,
            lhs_cs,
            lhs_rs,
            rhs as *mut c32,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const c32),
            *(&beta as *const T as *const c32),
            conj_dst,
            conj_lhs,
            conj_rhs,
            parallelism,
            stack,
        )
    } else {
        panic!();
    }
}

/// dst := alpha×dst + beta×lhs×rhs
///
/// Under `std` the packing buffers are thread-locals of `gemm-common`: a call allocates on a
/// thread's first call and when a product needs more than the buffer that thread holds.
/// [`gemm_with_stack`] takes them from the caller instead.
///
/// # Panics
///
/// Panics if `T` is not `f32`, `f64`, `gemm::f16`, `gemm::c32`, or `gemm::c64`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    gemm_imp(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
        None,
    )
}

/// An upper bound on the scratch [`gemm_with_stack::<T>`](gemm_with_stack) needs for an
/// `m x n x k` product with `parallelism`, whatever the strides and whichever kernel the
/// dispatch picks (both orientations, a packed rhs and a prepacked or packed lhs, and every
/// kernel compiled for the target, the SME kernels included). It depends on `T`, the shape and
/// the host's cache sizes only.
///
/// It is empty for `Parallelism::Rayon`, whose calls still use the thread-local buffers (see
/// [`gemm_with_stack`]).
///
/// # Panics
///
/// Panics if `T` is not `f32`, `f64`, `gemm::c32`, or `gemm::c64`.
pub fn gemm_scratch<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    parallelism: Parallelism,
) -> dyn_stack::StackReq {
    #[cfg(feature = "rayon")]
    if let Parallelism::Rayon(_) = parallelism {
        return dyn_stack::StackReq::EMPTY;
    }
    let _ = parallelism;
    let one = |m, n| {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            gemm_f64::gemm::f64::gemm_scratch(m, n, k)
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            gemm_f32::gemm::f32::gemm_scratch(m, n, k)
        } else if TypeId::of::<T>() == TypeId::of::<c64>() {
            gemm_c64::gemm::f64::gemm_scratch(m, n, k)
        } else if TypeId::of::<T>() == TypeId::of::<c32>() {
            gemm_c32::gemm::f32::gemm_scratch(m, n, k)
        } else {
            panic!("gemm_scratch supports f32, f64, c32 and c64");
        }
    };
    // `gemm` transposes the product when `dst` is closer to row-major
    one(m, n).or(one(n, m))
}

/// [`gemm`], with its working memory taken from `stack` for `Parallelism::None`: the call
/// neither touches the thread-local buffers of `gemm-common` nor allocates, from a thread's
/// first call on, provided `stack` holds [`gemm_scratch::<T>(m, n, k,
/// parallelism)`](gemm_scratch). The result is bit-for-bit that of [`gemm`]: the kernel, the
/// blocking and the packing are the same; only where the packed operands live differs.
///
/// With `Parallelism::Rayon` the call is [`gemm`]: it still uses the thread-local buffers of
/// each worker, and `stack` is not used.
///
/// # Safety
///
/// The same as [`gemm`].
///
/// # Panics
///
/// Panics if `T` is not `f32`, `f64`, `gemm::c32`, or `gemm::c64`, or if `stack` is too short
/// for the packing buffers the call takes, before `dst` is written. Those buffers are never
/// more than [`gemm_scratch::<T>(m, n, k, parallelism)`](gemm_scratch); a call that takes none
/// (an empty product, `k == 0`, a matrix-vector product, some small products) does not look at
/// `stack`. The length of `stack` is not checked against `gemm_scratch` up front, as that
/// would cost a second blocking computation on every call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_with_stack<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
    stack: &mut dyn_stack::MemStack,
) {
    // No `can_hold(gemm_scratch(..))` check here: it would cost a second blocking computation
    // on every call, which made an `f64` 8 x 8 x 8 call about 1.5 times as long. A short
    // `stack` makes `MemStack::make_aligned_uninit` panic instead, when the buffers are taken,
    // before `dst` is written.
    gemm_imp(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
        Some(stack),
    )
}

/// `gemm` and `gemm_with_stack`: one code path, with or without a caller's stack.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_imp<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    mut dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
    stack: Option<&mut dyn_stack::MemStack>,
) {
    // we want to transpose if the destination is column-oriented, since the microkernel prefers
    // column major matrices.
    let do_transpose = dst_cs.abs() < dst_rs.abs();

    let (
        m,
        n,
        mut dst_cs,
        mut dst_rs,
        mut lhs,
        mut lhs_cs,
        mut lhs_rs,
        mut rhs,
        mut rhs_cs,
        mut rhs_rs,
        conj_lhs,
        conj_rhs,
    ) = if do_transpose {
        (
            n, m, dst_rs, dst_cs, rhs, rhs_rs, rhs_cs, lhs, lhs_rs, lhs_cs, conj_rhs, conj_lhs,
        )
    } else {
        (
            m, n, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, conj_lhs, conj_rhs,
        )
    };

    if dst_rs < 0 && m > 0 {
        dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
        dst_rs = -dst_rs;
        lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
        lhs_rs = -lhs_rs;
    }

    if dst_cs < 0 && n > 0 {
        dst = dst.wrapping_offset((n - 1) as isize * dst_cs);
        dst_cs = -dst_cs;
        rhs = rhs.wrapping_offset((n - 1) as isize * rhs_cs);
        rhs_cs = -rhs_cs;
    }

    if lhs_cs < 0 && k > 0 {
        lhs = lhs.wrapping_offset((k - 1) as isize * lhs_cs);
        lhs_cs = -lhs_cs;
        rhs = rhs.wrapping_offset((k - 1) as isize * rhs_rs);
        rhs_rs = -rhs_rs;
    }

    gemm_dispatch(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
        stack,
    )
}

/// `dst := alpha * dst + beta * lhs * rhs` for the sealed real input pairs in
/// [`MixedRealGemmToF64`]. Inputs are converted exactly into packed `f64` panels, and products and
/// partial sums use `f64`. With `read_dst == false`, old destination values are not read and
/// `alpha` is ignored. Empty output returns without access; zero depth applies only the old-output
/// rule. Mixed calls always enter the packed `f64` driver, including vector and small products.
///
/// Packing uses the per-thread buffers of `gemm-common`, reused across calls. Use
/// [`gemm_mixed_with_stack`] to provide the packing storage directly.
///
/// # Safety
///
/// The pointers and strides must describe valid `m x n`, `m x k`, and `k x n` matrices. `dst`
/// must not alias either operand, and every negative-stride origin must address the logical
/// `(0, 0)` element.
///
/// # Examples
///
/// ```
/// use gemm::{Parallelism, gemm_mixed};
/// let a = [1.0_f32 + f32::EPSILON];
/// let b = [a[0]];
/// let mut c = [0.0_f64];
/// unsafe {
///     gemm_mixed(
///         1, 1, 1,
///         c.as_mut_ptr(), 1, 1, false,
///         a.as_ptr(), 1, 1,
///         b.as_ptr(), 1, 1,
///         0.0, 1.0, Parallelism::None,
///     );
/// }
/// assert_eq!(c[0], f64::from(a[0]) * f64::from(a[0]));
/// ```
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_mixed<Lhs, Rhs>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut f64,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const Lhs,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const Rhs,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: f64,
    beta: f64,
    parallelism: Parallelism,
) where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    gemm_mixed_imp(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        parallelism,
        None,
    )
    .expect("the sealed mixed GEMM type pair must be supported")
}

/// An upper bound on the packed `f64` scratch [`gemm_mixed_with_stack`] needs for an
/// `m x n x k` product with `parallelism`, whichever supported kernel is selected. The bound
/// assumes both operands are packed and does not depend on strides. It is empty for empty
/// products, zero depth, and `Parallelism::Rayon` (which uses per-thread buffers).
///
/// # Examples
///
/// ```
/// use gemm::{Parallelism, gemm_mixed_scratch};
/// let req = gemm_mixed_scratch::<f32, f64>(7, 5, 1031, Parallelism::None);
/// assert!(req.size_bytes() > 0);
/// ```
pub fn gemm_mixed_scratch<Lhs, Rhs>(
    m: usize,
    n: usize,
    k: usize,
    parallelism: Parallelism,
) -> dyn_stack::StackReq
where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    #[cfg(feature = "rayon")]
    if let Parallelism::Rayon(_) = parallelism {
        return dyn_stack::StackReq::EMPTY;
    }
    let _ = parallelism;
    if m == 0 || n == 0 || k == 0 {
        return dyn_stack::StackReq::EMPTY;
    }
    gemm_f64::gemm::f64::mixed::gemm_scratch(m, n, k)
}

/// [`gemm_mixed`], with packed `f64` storage taken from `stack` for `Parallelism::None`.
/// With a stack sized by [`gemm_mixed_scratch`], the call allocates no per-call working memory
/// after process-wide kernel and cache initialization. The first mixed call may allocate for that
/// initialization; supplying a stack does not suppress it. The arithmetic and blocking are the same
/// as [`gemm_mixed`].
///
/// # Safety
///
/// The same as [`gemm_mixed`]. With `Parallelism::None`, `stack` must be large enough for the
/// packed buffers selected by the layout and kernel.
///
/// # Panics
///
/// With `Parallelism::None`, a short stack panics before any destination write. The length is
/// checked when each buffer is acquired, not by an up-front comparison with
/// [`gemm_mixed_scratch`]. With `Parallelism::Rayon`, `stack` is ignored and this does not apply.
///
/// # Examples
///
/// ```
/// use gemm::{Parallelism, gemm_mixed_scratch, gemm_mixed_with_stack};
/// use gemm::dyn_stack::{MemBuffer, MemStack};
/// let a = [2.0_f32];
/// let b = [3.0_f64];
/// let mut c = [0.0_f64];
/// let req = gemm_mixed_scratch::<f32, f64>(1, 1, 1, Parallelism::None);
/// let mut buf = MemBuffer::new(req);
/// unsafe {
///     gemm_mixed_with_stack(
///         1, 1, 1,
///         c.as_mut_ptr(), 1, 1, false,
///         a.as_ptr(), 1, 1,
///         b.as_ptr(), 1, 1,
///         0.0, 1.0, Parallelism::None, MemStack::new(&mut buf),
///     );
/// }
/// assert_eq!(c[0], 6.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_mixed_with_stack<Lhs, Rhs>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut f64,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const Lhs,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const Rhs,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: f64,
    beta: f64,
    parallelism: Parallelism,
    stack: &mut dyn_stack::MemStack,
) where
    (Lhs, Rhs): MixedRealGemmToF64,
{
    gemm_mixed_imp(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        parallelism,
        Some(stack),
    )
    .expect("the sealed mixed GEMM type pair must be supported")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UnsupportedMixedGemm;

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_mixed_imp<Lhs: 'static, Rhs: 'static>(
    m: usize,
    n: usize,
    k: usize,
    mut dst: *mut f64,
    mut dst_cs: isize,
    mut dst_rs: isize,
    read_dst: bool,
    mut lhs: *const Lhs,
    mut lhs_cs: isize,
    mut lhs_rs: isize,
    mut rhs: *const Rhs,
    mut rhs_cs: isize,
    mut rhs_rs: isize,
    mut alpha: f64,
    beta: f64,
    parallelism: Parallelism,
    stack: Option<&mut dyn_stack::MemStack>,
) -> Result<(), UnsupportedMixedGemm> {
    let lhs_is_f32 = TypeId::of::<Lhs>() == TypeId::of::<f32>();
    let rhs_is_f32 = TypeId::of::<Rhs>() == TypeId::of::<f32>();
    let rhs_is_f64 = TypeId::of::<Rhs>() == TypeId::of::<f64>();
    if !lhs_is_f32 || (!rhs_is_f32 && !rhs_is_f64) {
        return Err(UnsupportedMixedGemm);
    }

    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        if !read_dst || alpha == 0.0 {
            for j in 0..n {
                for i in 0..m {
                    *dst.wrapping_offset(i as isize * dst_rs + j as isize * dst_cs) = 0.0;
                }
            }
        } else if alpha != 1.0 {
            for j in 0..n {
                for i in 0..m {
                    let dst = dst.wrapping_offset(i as isize * dst_rs + j as isize * dst_cs);
                    *dst *= alpha;
                }
            }
        }
        return Ok(());
    }
    if !read_dst {
        alpha = 0.0;
    }

    if dst_rs < 0 {
        dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
        dst_rs = -dst_rs;
        lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
        lhs_rs = -lhs_rs;
    }
    if dst_cs < 0 {
        dst = dst.wrapping_offset((n - 1) as isize * dst_cs);
        dst_cs = -dst_cs;
        rhs = rhs.wrapping_offset((n - 1) as isize * rhs_cs);
        rhs_cs = -rhs_cs;
    }
    if lhs_cs < 0 {
        lhs = lhs.wrapping_offset((k - 1) as isize * lhs_cs);
        lhs_cs = -lhs_cs;
        rhs = rhs.wrapping_offset((k - 1) as isize * rhs_rs);
        rhs_rs = -rhs_rs;
    }

    macro_rules! dispatch {
        ($lhs_ty:ty, $rhs_ty:ty, $get:path) => {{
            // SAFETY: the entry check has established both type IDs and rejected every
            // unsupported pair before any destination access.
            $get()(
                m,
                n,
                k,
                dst,
                dst_cs,
                dst_rs,
                lhs as *const $lhs_ty,
                lhs_cs,
                lhs_rs,
                rhs as *const $rhs_ty,
                rhs_cs,
                rhs_rs,
                alpha,
                beta,
                parallelism,
                stack,
            )
        }};
    }

    if rhs_is_f32 {
        dispatch!(
            f32,
            f32,
            gemm_f64::gemm::f64::mixed::get_f32_f32_with_stack_fn
        );
        Ok(())
    } else {
        dispatch!(
            f32,
            f64,
            gemm_f64::gemm::f64::mixed::get_f32_f64_with_stack_fn
        );
        Ok(())
    }
}

#[inline(never)]
#[cfg(test)]
pub unsafe fn gemm_fallback<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
) where
    T: num_traits::Zero + Send + Sync,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    (0..m).for_each(|row| {
        (0..n).for_each(|col| {
            let mut accum = <T as num_traits::Zero>::zero();
            for depth in 0..k {
                let lhs = &*lhs.wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs);

                let rhs = &*rhs.wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs);

                accum = &accum + &(lhs * rhs);
            }
            accum = &accum * &beta;

            let dst = dst.wrapping_offset(row as isize * dst_rs + col as isize * dst_cs);
            if read_dst {
                accum = &accum + &(&alpha * &*dst);
            }
            *dst = accum
        });
    });
    return;
}

#[inline(never)]
#[cfg(test)]
pub(crate) unsafe fn gemm_cplx_fallback<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut num_complex::Complex<T>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const num_complex::Complex<T>,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const num_complex::Complex<T>,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: num_complex::Complex<T>,
    beta: num_complex::Complex<T>,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) where
    T: num_traits::Zero
        + Send
        + Sync
        + Copy
        + num_traits::Num
        + core::ops::Neg<Output = T>
        + core::fmt::Debug,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Sub<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    (0..m).for_each(|row| {
        (0..n).for_each(|col| {
            let mut accum = <num_complex::Complex<T> as num_traits::Zero>::zero();
            for depth in 0..k {
                let lhs = &*lhs.wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs);
                let rhs = &*rhs.wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs);

                match (conj_lhs, conj_rhs) {
                    (true, true) => accum = &accum + &(lhs.conj() * rhs.conj()),
                    (true, false) => accum = &accum + &(lhs.conj() * rhs),
                    (false, true) => accum = &accum + &(lhs * rhs.conj()),
                    (false, false) => accum = &accum + &(lhs * rhs),
                }
            }
            accum = &accum * &beta;

            let dst = dst.wrapping_offset(row as isize * dst_rs + col as isize * dst_cs);
            if read_dst {
                match conj_dst {
                    true => accum = &accum + &(&alpha * (*dst).conj()),
                    false => accum = &accum + &(&alpha * &*dst),
                }
            }
            *dst = accum
        });
    });
    return;
}
