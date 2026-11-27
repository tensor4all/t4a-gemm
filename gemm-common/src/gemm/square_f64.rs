//! Single-thread column-major DGEMM for square 511, 512 and 513 matrices.
//! Compact input panels span the full depth. Twelve YMM accumulators cover each 8 x 6 tile.
//! Each result accumulates depth terms in input order. The active row block produces
//! the next block's packed A panels while processing complete six-column tiles.

use super::{DynStack, PackBuffer, StackReq};
use core::arch::x86_64::*;

/// Try a serial, compact column-major square product with unit scale and overwrite semantics.
/// Returns true after writing the product. A false return leaves the destination untouched.
/// The supported sizes are 511 through 513. Each output accumulates in increasing depth order.
/// Requires AVX2, FMA and gradual underflow. Scratch is reused on the calling thread.
///
/// # Safety
///
/// For a supported size, both inputs must be readable for `n * n` consecutive doubles
/// and the destination writable for that many doubles. The destination must not overlap
/// either input. Its initial contents may be uninitialized.
pub unsafe fn try_gemm_square_f64(
    n: usize,
    lhs: *const f64,
    rhs: *const f64,
    dst: *mut f64,
) -> bool {
    if !(511..=513).contains(&n)
        || !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("fma")
    {
        return false;
    }
    // MXCSR bits 15 and 6 enable flush-to-zero and denormals-are-zero, respectively.
    let mut mxcsr = 0u32;
    core::arch::asm!("stmxcsr [{ptr}]", ptr = in(reg) &mut mxcsr, options(nostack, preserves_flags));
    if mxcsr & 0x8040 != 0 {
        return false;
    }
    match n {
        511 => square::<511>(lhs, rhs, dst),
        512 => square::<512>(lhs, rhs, dst),
        513 => square::<513>(lhs, rhs, dst),
        _ => unreachable!(),
    }
    true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn square<const K: usize>(a: *const f64, b: *const f64, c: *mut f64) {
    #[cfg(debug_assertions)]
    debug_assert!((511..=513).contains(&K));
    // Eight doubles per rank. At 512 the extra cache line separates adjacent panels.
    let sa = 8 * K + usize::from(K == 512) * 8;
    let packed_len = K.div_ceil(8) * sa;
    let req = StackReq::new_aligned::<f64>(packed_len, 64);
    let mut mem = PackBuffer::take(req);
    let packed = DynStack::new(&mut mem)
        .make_aligned_uninit::<f64>(packed_len, 64)
        .0
        .as_mut_ptr()
        .cast::<f64>();
    let shift = if K == 512 {
        ((64 - c.addr() % 64) % 64) / 8
    } else {
        0
    };
    let a0 = a.add(shift);
    let c0 = c.add(shift);
    let panels = if K == 513 {
        62
    } else if K == 511 || shift > 0 {
        63
    } else {
        64
    };
    // The first six columns seed these panels. Each subsequent row block packs its successor.
    let seed = if K == 511 { 16 } else { 8 };
    let mc = match K {
        511 => 24,
        512 => 28,
        _ => 27,
    };
    let edge = packed.add(panels * sa);
    if K == 511 {
        for d in 0..K {
            core::ptr::copy_nonoverlapping(a.add(d * K + 504), edge.add(d * 8), 7);
            *edge.add(d * 8 + 7) = *a.add(d * K + 510);
        }
    }
    let step = if K == 511 { 64 } else { 16 };
    let blocks = (K + step / 2) / step;
    for block in 0..blocks {
        let lo = block * step;
        let hi = if block + 1 == blocks { K } else { lo + step };
        for p in 0..seed {
            pack_and_multiply_first_panel(
                a0.add(p * 8),
                K as isize,
                packed.add(p * sa),
                b,
                K as isize,
                c0.add(p * 8),
                K as isize,
                1.,
                lo,
                hi,
                hi == K,
            );
        }
    }
    let mut first = 0;
    while first < panels {
        let block = |begin: usize| {
            let size = if begin == 0 { seed } else { mc };
            size.min(panels - begin)
        };
        let count = block(first);
        let next = first + count;
        let end = if K == 511 {
            if first == 0 {
                panels
            } else {
                next
            }
        } else {
            next + block(next)
        };
        if next == panels {
            if K == 513 {
                pack_bottom17(a.add(496), K as isize, edge);
            } else if shift > 0 {
                for d in 0..512 {
                    core::ptr::copy_nonoverlapping(
                        a.add(d * 512 + 504 + shift),
                        edge.add(d * 8),
                        8 - shift,
                    );
                    core::ptr::copy_nonoverlapping(
                        a.add(d * 512),
                        edge.add(d * 8 + 8 - shift),
                        shift,
                    );
                }
            }
        }
        let groups = (end - next).div_ceil(count);
        let spacing = if K == 511 {
            1
        } else {
            2usize
                .min(83 / (groups * 8).saturating_sub(1).max(1))
                .max(1)
        };
        let slots = groups * 8;
        let begin = 85
            - if slots == 0 {
                0
            } else {
                (slots - 1) * spacing + 1
            };
        let slab = packed.add(first * sa);
        let dst = c0.add(first * 8);
        for j in usize::from(first == 0)..85 {
            let right = b.add(6 * j * K);
            let out = dst.add(6 * j * K);
            let mut copied = 0;
            if j >= begin && (j - begin) % spacing == 0 {
                let ordinal = (j - begin) / spacing;
                let (g, part) = if K == 511 {
                    (ordinal / 8, ordinal % 8)
                } else {
                    (ordinal % groups, ordinal / groups)
                };
                let g = if K == 512 { groups - 1 - g } else { g };
                let future = next + g * count;
                copied = count.min(end - future);
                let depth = part * (K / 8);
                let source = a0.add(future * 8 + depth * K);
                let target = packed.add(future * sa + depth * 8);
                let siblings = if K == 512 {
                    copy_sibling_tiles::<2, false>(
                        K, slab, right, out, sa, K as isize, K as isize, copied, source, target,
                    )
                } else {
                    copy_sibling_tiles::<4, true>(
                        K, slab, right, out, sa, K as isize, K as isize, copied, source, target,
                    )
                };
                if siblings < copied {
                    let pa = slab.add(siblings * sa);
                    let pd = out.add(siblings * 8);
                    let src = source.add(siblings * 8);
                    let to = target.add(siblings * sa);
                    copy_sibling_tiles::<1, false>(
                        K,
                        pa,
                        right,
                        pd,
                        sa,
                        K as isize,
                        K as isize,
                        copied - siblings,
                        src,
                        to,
                    );
                }
                if part == 7 {
                    for d in K / 8 * 8..K {
                        for r in 0..copied {
                            core::ptr::copy_nonoverlapping(
                                a0.add((future + r) * 8 + d * K),
                                packed.add((future + r) * sa + d * 8),
                                8,
                            );
                        }
                    }
                }
            }
            if copied < count {
                let pa = slab.add(copied * sa);
                let dst = out.add(copied * 8);
                let qty = count - copied;
                if K == 511 {
                    prefetched_511(pa, right, dst, sa, qty);
                } else {
                    depth_loop_tile(K, pa, right, dst, sa, K as isize, K as isize, qty);
                }
            }
            if next == panels {
                if K == 513 {
                    bottom17_range::<2>(
                        edge,
                        right,
                        K as isize,
                        c.add(j * 6 * K + 496),
                        K as isize,
                        1.,
                        64,
                        32,
                        2,
                    );
                } else if shift > 0 {
                    wrapped_panel_512(edge, right, c.add(j * 6 * K), shift, 6);
                } else if K == 511 {
                    let mut temp = [0.; 48];
                    depth_loop_tile(K, edge, right, temp.as_mut_ptr(), sa, K as isize, 8, 1);
                    for col in 0..6 {
                        core::ptr::copy_nonoverlapping(
                            temp.as_ptr().add(col * 8),
                            c.add((j * 6 + col) * K + 504),
                            7,
                        );
                    }
                }
            }
        }
        let right = b.add(510 * K);
        let tail_out = dst.add(510 * K);
        let div = match K {
            511 => 6,
            512 => 3,
            _ => 2,
        };
        let full = count / div * div;
        if K == 511 {
            for p in (0..full).step_by(6) {
                edge_511_48x1(slab.add(p * sa), sa, right, tail_out.add(p * 8), 1.);
            }
        } else if K == 512 {
            narrow_24x2(slab, right, tail_out, sa, K as isize, K as isize, full, 1.);
        } else {
            narrow_16x3(slab, right, tail_out, sa, K as isize, K as isize, full, 1.);
        }
        for p in full..count {
            small::<K>(slab.add(p * sa), right, tail_out.add(p * 8), 8);
        }
        if next == panels {
            if K == 513 {
                bottom17_range::<2>(
                    edge,
                    right,
                    K as isize,
                    c.add(510 * K + 496),
                    K as isize,
                    1.,
                    64,
                    32,
                    1,
                );
            } else if shift > 0 {
                wrapped_panel_512(edge, right, c.add(510 * K), shift, 2);
            } else if K == 511 {
                small::<K>(edge, right, c.add(510 * K + 504), 7);
            }
        }
        first = next;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn small<const K: usize>(a: *const f64, b: *const f64, c: *mut f64, rows: usize) {
    for j in 0..K - 510 {
        let mut x = _mm256_setzero_pd();
        let mut y = x;
        for d in 0..K {
            let rhs = _mm256_set1_pd(*b.add(j * K + d));
            x = _mm256_fmadd_pd(_mm256_loadu_pd(a.add(d * 8)), rhs, x);
            y = _mm256_fmadd_pd(_mm256_loadu_pd(a.add(d * 8 + 4)), rhs, y);
        }
        _mm256_storeu_pd(c.add(j * K), x);
        if rows == 8 {
            _mm256_storeu_pd(c.add(j * K + 4), y);
        } else {
            let mut high = [0.; 4];
            _mm256_storeu_pd(high.as_mut_ptr(), y);
            core::ptr::copy_nonoverlapping(high.as_ptr(), c.add(j * K + 4), 3);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn pack_and_multiply_first_panel(
    lhs: *const f64,
    lhs_cs: isize,
    packed: *mut f64,
    rhs: *const f64,
    rhs_cs: isize,
    dst: *mut f64,
    dst_cs: isize,
    scale: f64,
    lo: usize,
    hi: usize,
    last: bool,
) {
    use core::arch::x86_64::*;
    let mut low = [_mm256_setzero_pd(); 6];
    let mut high = low;
    if lo != 0 {
        for col in 0..6 {
            let out = dst.offset(col as isize * dst_cs);
            low[col] = _mm256_loadu_pd(out);
            high[col] = _mm256_loadu_pd(out.add(4));
        }
    }
    for depth in lo..hi {
        let source = lhs.offset(depth as isize * lhs_cs);
        let first = _mm256_loadu_pd(source);
        let second = _mm256_loadu_pd(source.add(4));
        _mm256_storeu_pd(packed.add(depth * 8), first);
        _mm256_storeu_pd(packed.add(depth * 8 + 4), second);
        for col in 0..6 {
            let b = _mm256_set1_pd(*rhs.offset(depth as isize + col as isize * rhs_cs));
            low[col] = _mm256_fmadd_pd(first, b, low[col]);
            high[col] = _mm256_fmadd_pd(second, b, high[col]);
        }
    }
    let scale = _mm256_set1_pd(scale);
    for col in 0..6 {
        let out = dst.offset(col as isize * dst_cs);
        _mm256_storeu_pd(
            out,
            if last {
                _mm256_mul_pd(scale, low[col])
            } else {
                low[col]
            },
        );
        _mm256_storeu_pd(
            out.add(4),
            if last {
                _mm256_mul_pd(scale, high[col])
            } else {
                high[col]
            },
        );
    }
}

// One increasing-depth FMA step, shared by computing and packing tiles.
macro_rules! accumulate_8x6 {
    ($depth:literal) => {
        concat!(
            "vmovupd ymm12, [{a} + 64*(",
            $depth,
            ")]\n",
            "vmovupd ymm13, [{a} + 64*(",
            $depth,
            ") + 32]\n",
            "vbroadcastsd ymm14, [{b0} + {idx} + 8*(",
            $depth,
            ")]\n",
            "vbroadcastsd ymm15, [{b1} + {idx} + 8*(",
            $depth,
            ")]\n",
            "vfmadd231pd ymm0, ymm12, ymm14\n",
            "vfmadd231pd ymm1, ymm13, ymm14\n",
            "vbroadcastsd ymm14, [{b2} + {idx} + 8*(",
            $depth,
            ")]\n",
            "vfmadd231pd ymm2, ymm12, ymm15\n",
            "vfmadd231pd ymm3, ymm13, ymm15\n",
            "vbroadcastsd ymm15, [{b3} + {idx} + 8*(",
            $depth,
            ")]\n",
            "vfmadd231pd ymm4, ymm12, ymm14\n",
            "vfmadd231pd ymm5, ymm13, ymm14\n",
            "vbroadcastsd ymm14, [{b4} + {idx} + 8*(",
            $depth,
            ")]\n",
            "vfmadd231pd ymm6, ymm12, ymm15\n",
            "vfmadd231pd ymm7, ymm13, ymm15\n",
            "vbroadcastsd ymm15, [{b5} + {idx} + 8*(",
            $depth,
            ")]\n",
            "vfmadd231pd ymm8, ymm12, ymm14\n",
            "vfmadd231pd ymm9, ymm13, ymm14\n",
            "vfmadd231pd ymm10, ymm12, ymm15\n",
            "vfmadd231pd ymm11, ymm13, ymm15\n",
        )
    };
}

#[target_feature(enable = "avx2,fma")]
unsafe fn depth_loop_tile(
    k: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    sa: usize,
    sb: isize,
    sc: isize,
    panels: usize,
) {
    use core::arch::x86_64::*;
    let bulk = k / 8 * 8;
    let b = b.wrapping_add(bulk);
    for panel in 0..panels {
        let a = a.add(panel * sa);
        let c = c.add(panel * 8);
        let v0: __m256d;
        let v1: __m256d;
        let v2: __m256d;
        let v3: __m256d;
        let v4: __m256d;
        let v5: __m256d;
        let v6: __m256d;
        let v7: __m256d;
        let v8: __m256d;
        let v9: __m256d;
        let v10: __m256d;
        let v11: __m256d;

        core::arch::asm!(
            "vxorpd ymm0, ymm0, ymm0",
            "vxorpd ymm1, ymm1, ymm1",
            "vxorpd ymm2, ymm2, ymm2",
            "vxorpd ymm3, ymm3, ymm3",
            "vxorpd ymm4, ymm4, ymm4",
            "vxorpd ymm5, ymm5, ymm5",
            "vxorpd ymm6, ymm6, ymm6",
            "vxorpd ymm7, ymm7, ymm7",
            "vxorpd ymm8, ymm8, ymm8",
            "vxorpd ymm9, ymm9, ymm9",
            "vxorpd ymm10, ymm10, ymm10",
            "vxorpd ymm11, ymm11, ymm11",
            ".p2align 5",
            "2:",
            ".irp q,0,1,2,3,4,5,6,7",
            accumulate_8x6!("\\q"),
            ".endr",
            "add {a}, 512",
            "add {idx}, 64",
            "jnz 2b",
            "test {tail}, {tail}",
            "jz 4f",
            "3:",
            accumulate_8x6!("0"),
            "add {a},64",
            "add {idx},8",
            "dec {tail}",
            "jnz 3b",
            "4:",
            a=inout(reg) a=>_,idx=inout(reg) (-(bulk as isize)*8)=>_,tail=inout(reg) (k-bulk)=>_,
            b0=in(reg) b.wrapping_offset(0*sb),
            b1=in(reg) b.wrapping_offset(1*sb),
            b2=in(reg) b.wrapping_offset(2*sb),
            b3=in(reg) b.wrapping_offset(3*sb),
            b4=in(reg) b.wrapping_offset(4*sb),
            b5=in(reg) b.wrapping_offset(5*sb),
            out("ymm0") v0, out("ymm1") v1, out("ymm2") v2, out("ymm3") v3, out("ymm4") v4, out("ymm5") v5, out("ymm6") v6, out("ymm7") v7, out("ymm8") v8, out("ymm9") v9, out("ymm10") v10, out("ymm11") v11,
            out("ymm12") _,out("ymm13") _,out("ymm14") _,out("ymm15") _,options(nostack,readonly),
        );
        _mm256_storeu_pd(c.offset(0 * sc).add(0), v0);
        _mm256_storeu_pd(c.offset(0 * sc).add(4), v1);
        _mm256_storeu_pd(c.offset(1 * sc).add(0), v2);
        _mm256_storeu_pd(c.offset(1 * sc).add(4), v3);
        _mm256_storeu_pd(c.offset(2 * sc).add(0), v4);
        _mm256_storeu_pd(c.offset(2 * sc).add(4), v5);
        _mm256_storeu_pd(c.offset(3 * sc).add(0), v6);
        _mm256_storeu_pd(c.offset(3 * sc).add(4), v7);
        _mm256_storeu_pd(c.offset(4 * sc).add(0), v8);
        _mm256_storeu_pd(c.offset(4 * sc).add(4), v9);
        _mm256_storeu_pd(c.offset(5 * sc).add(0), v10);
        _mm256_storeu_pd(c.offset(5 * sc).add(4), v11);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn copy_sibling_tiles<const W: usize, const EARLY: bool>(
    k: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    sa: usize,
    sb: isize,
    sc: isize,
    count: usize,
    source: *const f64,
    target: *mut f64,
) -> usize {
    use core::arch::x86_64::*;
    let copied = count / W * W;
    let bulk = k / 8 * 8;
    let span = k / 8 / W;
    let end = b.wrapping_add(bulk);
    for panel in 0..copied {
        let start = panel / W * W;
        let depth = panel % W * span;
        let src = source.add(start * 8 + depth * k);
        let wr = target.add(start * sa + depth * 8);
        let left = a.add(panel * sa);
        let dst = c.add(panel * 8);
        let v0: __m256d;
        let v1: __m256d;
        let v2: __m256d;
        let v3: __m256d;
        let v4: __m256d;
        let v5: __m256d;
        let v6: __m256d;
        let v7: __m256d;
        let v8: __m256d;
        let v9: __m256d;
        let v10: __m256d;
        let v11: __m256d;
        core::arch::asm!(
            "vxorpd ymm0,ymm0,ymm0",
            "vxorpd ymm1,ymm1,ymm1",
            "vxorpd ymm2,ymm2,ymm2",
            "vxorpd ymm3,ymm3,ymm3",
            "vxorpd ymm4,ymm4,ymm4",
            "vxorpd ymm5,ymm5,ymm5",
            "vxorpd ymm6,ymm6,ymm6",
            "vxorpd ymm7,ymm7,ymm7",
            "vxorpd ymm8,ymm8,ymm8",
            "vxorpd ymm9,ymm9,ymm9",
            "vxorpd ymm10,ymm10,ymm10",
            "vxorpd ymm11,ymm11,ymm11",
            ".p2align 5",
            "2:",
            ".irp q,0,1,2,3,4,5,6,7",
            accumulate_8x6!("\\q"),
            ".if (\\q==2 && {early}) || (\\q==7 && !{early})",
            "test {idx}, {mask}",
            "jnz 3f",
            ".irp v,0,1,2,3,4,5,6,7",
            ".if \\v < {width}",
            "vmovupd ymm12,[{src}+64*\\v]",
            "vmovupd ymm13,[{src}+64*\\v+32]",
            "vmovupd [{wr}],ymm12",
            "vmovupd [{wr}+32],ymm13",
            "add {wr},{stride}",
            ".endif",
            ".endr",
            "add {wr},{back}",
            "add {src},{srcstride}",
            "3:",
            ".endif",
            ".endr",
            "add {a},512",
            "add {idx},64",
            "jnz 2b",
            "test {tail},{tail}",
            "jz 5f",
            "4:",
            accumulate_8x6!("0"),
            "add {a},64",
            "add {idx},8",
            "dec {tail}",
            "jnz 4b",
            "5:",
            width=const W,early=const (EARLY as usize),mask=const (64*(W-1)),
            a=inout(reg) left=>_,idx=inout(reg) (-(bulk as isize)*8)=>_,tail=inout(reg) (k-bulk)=>_,
            src=inout(reg) src=>_,wr=inout(reg) wr=>_,stride=in(reg) (sa*8),
            back=in(reg) (64isize-(W*sa*8) as isize),srcstride=in(reg) (k*8),
            b0=in(reg) end.wrapping_offset(0*sb),
            b1=in(reg) end.wrapping_offset(1*sb),
            b2=in(reg) end.wrapping_offset(2*sb),
            b3=in(reg) end.wrapping_offset(3*sb),
            b4=in(reg) end.wrapping_offset(4*sb),
            b5=in(reg) end.wrapping_offset(5*sb),
            out("ymm0") v0, out("ymm1") v1, out("ymm2") v2, out("ymm3") v3, out("ymm4") v4, out("ymm5") v5, out("ymm6") v6, out("ymm7") v7, out("ymm8") v8, out("ymm9") v9, out("ymm10") v10, out("ymm11") v11,
            out("ymm12") _,out("ymm13") _,out("ymm14") _,out("ymm15") _,options(nostack),
        );
        _mm256_storeu_pd(dst.offset(0 * sc).add(0), v0);
        _mm256_storeu_pd(dst.offset(0 * sc).add(4), v1);
        _mm256_storeu_pd(dst.offset(1 * sc).add(0), v2);
        _mm256_storeu_pd(dst.offset(1 * sc).add(4), v3);
        _mm256_storeu_pd(dst.offset(2 * sc).add(0), v4);
        _mm256_storeu_pd(dst.offset(2 * sc).add(4), v5);
        _mm256_storeu_pd(dst.offset(3 * sc).add(0), v6);
        _mm256_storeu_pd(dst.offset(3 * sc).add(4), v7);
        _mm256_storeu_pd(dst.offset(4 * sc).add(0), v8);
        _mm256_storeu_pd(dst.offset(4 * sc).add(4), v9);
        _mm256_storeu_pd(dst.offset(5 * sc).add(0), v10);
        _mm256_storeu_pd(dst.offset(5 * sc).add(4), v11);
        for rem in (span * W)..k / 8 {
            core::ptr::copy_nonoverlapping(
                source.add(panel * 8 + rem * k),
                target.add(panel * sa + rem * 8),
                8,
            );
        }
    }
    copied
}

#[target_feature(enable = "avx2,fma")]
unsafe fn edge_511_48x1(a: *const f64, stride: usize, b: *const f64, dst: *mut f64, scale: f64) {
    use core::arch::x86_64::*;
    let mut sum = [_mm256_setzero_pd(); 12];
    for k in 0..511 {
        let v = _mm256_set1_pd(*b.add(k));
        for panel in 0..6 {
            let src = a.add(panel * stride + k * 8);
            let low = _mm256_loadu_pd(src);
            let high = _mm256_loadu_pd(src.add(4));
            sum[2 * panel] = _mm256_fmadd_pd(low, v, sum[2 * panel]);
            sum[2 * panel + 1] = _mm256_fmadd_pd(high, v, sum[2 * panel + 1]);
        }
    }
    let factor = _mm256_set1_pd(scale);
    for i in 0..12 {
        _mm256_storeu_pd(dst.add(i * 4), _mm256_mul_pd(factor, sum[i]));
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn narrow_24x2(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    sa: usize,
    sb: isize,
    sc: isize,
    panels: usize,
    beta: f64,
) {
    use core::arch::x86_64::*;
    let scale = _mm256_set1_pd(beta);
    let b0 = b.offset(0 * sb);
    let b1 = b.offset(1 * sb);
    for panel in (0..panels).step_by(3) {
        let c = c.add(panel * 8);
        let a0 = a.add((panel + 0) * sa);
        let a1 = a.add((panel + 1) * sa);
        let a2 = a.add((panel + 2) * sa);
        let out0: __m256d;
        let out1: __m256d;
        let out2: __m256d;
        let out3: __m256d;
        let out4: __m256d;
        let out5: __m256d;
        let out6: __m256d;
        let out7: __m256d;
        let out8: __m256d;
        let out9: __m256d;
        let out10: __m256d;
        let out11: __m256d;
        core::arch::asm!(
            "vxorpd ymm0, ymm0, ymm0",
            "vxorpd ymm1, ymm1, ymm1",
            "vxorpd ymm2, ymm2, ymm2",
            "vxorpd ymm3, ymm3, ymm3",
            "vxorpd ymm4, ymm4, ymm4",
            "vxorpd ymm5, ymm5, ymm5",
            "vxorpd ymm6, ymm6, ymm6",
            "vxorpd ymm7, ymm7, ymm7",
            "vxorpd ymm8, ymm8, ymm8",
            "vxorpd ymm9, ymm9, ymm9",
            "vxorpd ymm10, ymm10, ymm10",
            "vxorpd ymm11, ymm11, ymm11",
            ".p2align 5",
            "2:",
            ".irp q,0,1,2,3",
            "vmovupd ymm12, [{a0}+0+64*(\\q)]",
            "vmovupd ymm13, [{a0}+32+64*(\\q)]",
            "vbroadcastsd ymm14, [{b0}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm0, ymm12, ymm14",
            "vfmadd231pd ymm1, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b1}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm6, ymm12, ymm14",
            "vfmadd231pd ymm7, ymm13, ymm14",
            "vmovupd ymm12, [{a1}+0+64*(\\q)]",
            "vmovupd ymm13, [{a1}+32+64*(\\q)]",
            "vbroadcastsd ymm14, [{b0}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm2, ymm12, ymm14",
            "vfmadd231pd ymm3, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b1}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm8, ymm12, ymm14",
            "vfmadd231pd ymm9, ymm13, ymm14",
            "vmovupd ymm12, [{a2}+0+64*(\\q)]",
            "vmovupd ymm13, [{a2}+32+64*(\\q)]",
            "vbroadcastsd ymm14, [{b0}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm4, ymm12, ymm14",
            "vfmadd231pd ymm5, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b1}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm10, ymm12, ymm14",
            "vfmadd231pd ymm11, ymm13, ymm14",
            ".endr",
            "add {a0}, 256",
            "add {a1}, 256",
            "add {a2}, 256",
            "add {ix}, 4",
            "cmp {ix}, 512",
            "jb 2b",
            ix=inout(reg) 0usize=>_,
            a0=inout(reg) a0=>_,
            a1=inout(reg) a1=>_,
            a2=inout(reg) a2=>_,
            b0=in(reg) b0,
            b1=in(reg) b1,
            out("ymm0") out0,
            out("ymm1") out1,
            out("ymm2") out2,
            out("ymm3") out3,
            out("ymm4") out4,
            out("ymm5") out5,
            out("ymm6") out6,
            out("ymm7") out7,
            out("ymm8") out8,
            out("ymm9") out9,
            out("ymm10") out10,
            out("ymm11") out11,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, options(nostack,readonly),
        );
        _mm256_storeu_pd(c.offset(0 * sc).add(0), _mm256_mul_pd(scale, out0));
        _mm256_storeu_pd(c.offset(0 * sc).add(4), _mm256_mul_pd(scale, out1));
        _mm256_storeu_pd(c.offset(0 * sc).add(8), _mm256_mul_pd(scale, out2));
        _mm256_storeu_pd(c.offset(0 * sc).add(12), _mm256_mul_pd(scale, out3));
        _mm256_storeu_pd(c.offset(0 * sc).add(16), _mm256_mul_pd(scale, out4));
        _mm256_storeu_pd(c.offset(0 * sc).add(20), _mm256_mul_pd(scale, out5));
        _mm256_storeu_pd(c.offset(1 * sc).add(0), _mm256_mul_pd(scale, out6));
        _mm256_storeu_pd(c.offset(1 * sc).add(4), _mm256_mul_pd(scale, out7));
        _mm256_storeu_pd(c.offset(1 * sc).add(8), _mm256_mul_pd(scale, out8));
        _mm256_storeu_pd(c.offset(1 * sc).add(12), _mm256_mul_pd(scale, out9));
        _mm256_storeu_pd(c.offset(1 * sc).add(16), _mm256_mul_pd(scale, out10));
        _mm256_storeu_pd(c.offset(1 * sc).add(20), _mm256_mul_pd(scale, out11));
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn narrow_16x3(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    sa: usize,
    sb: isize,
    sc: isize,
    panels: usize,
    beta: f64,
) {
    use core::arch::x86_64::*;
    let scale = _mm256_set1_pd(beta);
    let b0 = b.offset(0 * sb);
    let b1 = b.offset(1 * sb);
    let b2 = b.offset(2 * sb);
    for panel in (0..panels).step_by(2) {
        let c = c.add(panel * 8);
        let a0 = a.add((panel + 0) * sa);
        let a1 = a.add((panel + 1) * sa);
        let out0: __m256d;
        let out1: __m256d;
        let out2: __m256d;
        let out3: __m256d;
        let out4: __m256d;
        let out5: __m256d;
        let out6: __m256d;
        let out7: __m256d;
        let out8: __m256d;
        let out9: __m256d;
        let out10: __m256d;
        let out11: __m256d;
        core::arch::asm!(
            "vxorpd ymm0, ymm0, ymm0",
            "vxorpd ymm1, ymm1, ymm1",
            "vxorpd ymm2, ymm2, ymm2",
            "vxorpd ymm3, ymm3, ymm3",
            "vxorpd ymm4, ymm4, ymm4",
            "vxorpd ymm5, ymm5, ymm5",
            "vxorpd ymm6, ymm6, ymm6",
            "vxorpd ymm7, ymm7, ymm7",
            "vxorpd ymm8, ymm8, ymm8",
            "vxorpd ymm9, ymm9, ymm9",
            "vxorpd ymm10, ymm10, ymm10",
            "vxorpd ymm11, ymm11, ymm11",
            ".p2align 5",
            "2:",
            ".irp q,0,1,2",
            "vmovupd ymm12, [{a0}+0+64*(\\q)]",
            "vmovupd ymm13, [{a0}+32+64*(\\q)]",
            "vbroadcastsd ymm14, [{b0}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm0, ymm12, ymm14",
            "vfmadd231pd ymm1, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b1}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm4, ymm12, ymm14",
            "vfmadd231pd ymm5, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b2}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm8, ymm12, ymm14",
            "vfmadd231pd ymm9, ymm13, ymm14",
            "vmovupd ymm12, [{a1}+0+64*(\\q)]",
            "vmovupd ymm13, [{a1}+32+64*(\\q)]",
            "vbroadcastsd ymm14, [{b0}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm2, ymm12, ymm14",
            "vfmadd231pd ymm3, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b1}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm6, ymm12, ymm14",
            "vfmadd231pd ymm7, ymm13, ymm14",
            "vbroadcastsd ymm14, [{b2}+{ix}*8+0+8*(\\q)]",
            "vfmadd231pd ymm10, ymm12, ymm14",
            "vfmadd231pd ymm11, ymm13, ymm14",
            ".endr",
            "add {a0}, 192",
            "add {a1}, 192",
            "add {ix}, 3",
            "cmp {ix}, 513",
            "jb 2b",
            ix=inout(reg) 0usize=>_,
            a0=inout(reg) a0=>_,
            a1=inout(reg) a1=>_,
            b0=in(reg) b0,
            b1=in(reg) b1,
            b2=in(reg) b2,
            out("ymm0") out0,
            out("ymm1") out1,
            out("ymm2") out2,
            out("ymm3") out3,
            out("ymm4") out4,
            out("ymm5") out5,
            out("ymm6") out6,
            out("ymm7") out7,
            out("ymm8") out8,
            out("ymm9") out9,
            out("ymm10") out10,
            out("ymm11") out11,
            out("ymm12") _, out("ymm13") _, out("ymm14") _, options(nostack,readonly),
        );
        _mm256_storeu_pd(c.offset(0 * sc).add(0), _mm256_mul_pd(scale, out0));
        _mm256_storeu_pd(c.offset(0 * sc).add(4), _mm256_mul_pd(scale, out1));
        _mm256_storeu_pd(c.offset(0 * sc).add(8), _mm256_mul_pd(scale, out2));
        _mm256_storeu_pd(c.offset(0 * sc).add(12), _mm256_mul_pd(scale, out3));
        _mm256_storeu_pd(c.offset(1 * sc).add(0), _mm256_mul_pd(scale, out4));
        _mm256_storeu_pd(c.offset(1 * sc).add(4), _mm256_mul_pd(scale, out5));
        _mm256_storeu_pd(c.offset(1 * sc).add(8), _mm256_mul_pd(scale, out6));
        _mm256_storeu_pd(c.offset(1 * sc).add(12), _mm256_mul_pd(scale, out7));
        _mm256_storeu_pd(c.offset(2 * sc).add(0), _mm256_mul_pd(scale, out8));
        _mm256_storeu_pd(c.offset(2 * sc).add(4), _mm256_mul_pd(scale, out9));
        _mm256_storeu_pd(c.offset(2 * sc).add(8), _mm256_mul_pd(scale, out10));
        _mm256_storeu_pd(c.offset(2 * sc).add(12), _mm256_mul_pd(scale, out11));
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn pack_bottom17(a: *const f64, lda: isize, panel: *mut f64) {
    use core::arch::x86_64::*;
    for k in 0..513 {
        let src = a.offset(k as isize * lda);
        let dst = panel.add(k * 20);
        core::ptr::copy_nonoverlapping(src, dst, 16);
        _mm256_store_pd(dst.add(16), _mm256_set1_pd(*src.add(16)));
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn bottom17_range<const CAP: usize>(
    panel: *const f64,
    b: *const f64,
    ldb: isize,
    c: *mut f64,
    ldc: isize,
    scale: f64,
    slab: usize,
    pair_groups: usize,
    groups: usize,
) {
    use core::arch::x86_64::*;
    debug_assert!(groups <= CAP);
    let mut storage = core::mem::MaybeUninit::<[[__m256d; 13]; CAP]>::uninit();
    let ptr = storage.as_mut_ptr().cast::<[__m256d; 13]>();
    core::ptr::write_bytes(ptr, 0, groups);
    let sums = core::slice::from_raw_parts_mut(ptr, groups);
    for (chunk_index, chunk) in sums.chunks_mut(pair_groups).enumerate() {
        for k0 in (0..512).step_by(slab) {
            for (local, acc) in chunk.iter_mut().enumerate() {
                let group = chunk_index * pair_groups + local;
                let a = panel.add(k0 * 20);
                let rhs = b.offset(3 * group as isize * ldb).add(k0);
                core::arch::asm!(
                    ".p2align 4",
                    "2:",
                    ".irp q,0,1,2,3",
                    "vbroadcastsd ymm13, [{b0} + {i} + 0+8*(\\q)]",
                    "vbroadcastsd ymm14, [{b1} + {i} + 0+8*(\\q)]",
                    "vmovapd ymm15, [{a} + 0+160*(\\q)]",
                    "vfmadd231pd ymm0, ymm13, ymm15",
                    "vfmadd231pd ymm4, ymm14, ymm15",
                    "vmovapd ymm15, [{a} + 32+160*(\\q)]",
                    "vfmadd231pd ymm1, ymm13, ymm15",
                    "vfmadd231pd ymm5, ymm14, ymm15",
                    "vmovapd ymm15, [{a} + 64+160*(\\q)]",
                    "vfmadd231pd ymm2, ymm13, ymm15",
                    "vfmadd231pd ymm6, ymm14, ymm15",
                    "vmovapd ymm15, [{a} + 96+160*(\\q)]",
                    "vfmadd231pd ymm3, ymm13, ymm15",
                    "vfmadd231pd ymm7, ymm14, ymm15",
                    "vunpcklpd xmm13, xmm13, xmm14",
                    "vbroadcastsd ymm14, [{b2} + {i} + 0+8*(\\q)]",
                    "vfmadd231pd ymm8, ymm14, [{a} + 0+160*(\\q)]",
                    "vfmadd231pd ymm9, ymm14, [{a} + 32+160*(\\q)]",
                    "vfmadd231pd ymm10, ymm14, [{a} + 64+160*(\\q)]",
                    "vfmadd231pd ymm11, ymm14, [{a} + 96+160*(\\q)]",
                    "vinsertf128 ymm13, ymm13, xmm14, 1",
                    "vfmadd231pd ymm12, ymm13, [{a} + 128+160*(\\q)]",
                    ".endr",
                    "add {a}, 640","add {i}, 32","cmp {i}, {bytes}","jne 2b",
                    a=inout(reg) a=>_, i=inout(reg) 0usize=>_, bytes=in(reg) (slab*8),
                    b0=in(reg) rhs,b1=in(reg) rhs.offset(ldb),b2=in(reg) rhs.offset(2*ldb),
                    inout("ymm0") acc[0],
                    inout("ymm1") acc[1],
                    inout("ymm2") acc[2],
                    inout("ymm3") acc[3],
                    inout("ymm4") acc[4],
                    inout("ymm5") acc[5],
                    inout("ymm6") acc[6],
                    inout("ymm7") acc[7],
                    inout("ymm8") acc[8],
                    inout("ymm9") acc[9],
                    inout("ymm10") acc[10],
                    inout("ymm11") acc[11],
                    inout("ymm12") acc[12],
                    out("ymm13") _,out("ymm14") _,out("ymm15") _,options(nostack,readonly)
                );
            }
        }
    }
    let factor = _mm256_set1_pd(scale);
    let tail = panel.add(512 * 20);
    for (group, acc) in sums.iter().enumerate() {
        let rhs = b.offset(3 * group as isize * ldb).add(512);
        let last_b = _mm256_set_pd(
            *rhs.offset(2 * ldb),
            *rhs.offset(2 * ldb),
            *rhs.offset(ldb),
            *rhs,
        );
        let last = _mm256_mul_pd(
            factor,
            _mm256_fmadd_pd(_mm256_load_pd(tail.add(16)), last_b, acc[12]),
        );
        let mut values = [0.; 4];
        _mm256_storeu_pd(values.as_mut_ptr(), last);
        for j in 0..3 {
            let dst = c.offset((3 * group + j) as isize * ldc);
            let right = _mm256_set1_pd(*rhs.offset(j as isize * ldb));
            for i in 0..4 {
                let sum = _mm256_fmadd_pd(_mm256_load_pd(tail.add(4 * i)), right, acc[4 * j + i]);
                _mm256_storeu_pd(dst.add(4 * i), _mm256_mul_pd(factor, sum));
            }
            *dst.add(16) = values[j];
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn wrapped_panel_512(
    panel: *const f64,
    b: *const f64,
    c: *mut f64,
    shift: usize,
    cols: usize,
) {
    use core::arch::x86_64::*;
    let mut out = [0.0; 48];
    if cols == 6 {
        depth_loop_tile(512, panel, b, out.as_mut_ptr(), 8 * 512, 512, 8, 1);
    } else {
        for j in 0..cols {
            let rhs = b.add(j * 512);
            let mut lo = _mm256_setzero_pd();
            let mut hi = _mm256_setzero_pd();
            for k in 0..512 {
                let val = _mm256_set1_pd(*rhs.add(k));
                let a = panel.add(8 * k);
                lo = _mm256_fmadd_pd(_mm256_load_pd(a), val, lo);
                hi = _mm256_fmadd_pd(_mm256_load_pd(a.add(4)), val, hi);
            }
            _mm256_storeu_pd(out.as_mut_ptr().add(j * 8), lo);
            _mm256_storeu_pd(out.as_mut_ptr().add(j * 8 + 4), hi);
        }
    }
    for j in 0..cols {
        let dst = c.add(j * 512);
        let src = out.as_ptr().add(j * 8);
        core::ptr::copy_nonoverlapping(src, dst.add(504 + shift), 8 - shift);
        core::ptr::copy_nonoverlapping(src.add(8 - shift), dst, shift);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn prefetched_511(packed: *const f64, b: *const f64, c: *mut f64, sa: usize, panels: usize) {
    const K: usize = 511;
    for p in 0..panels {
        let a = packed.add(p * sa);
        let next = packed.add(if p + 1 < panels { (p + 1) * sa } else { p * sa });
        let c = c.add(8 * p);
        let v0: __m256d;
        let v1: __m256d;
        let v2: __m256d;
        let v3: __m256d;
        let v4: __m256d;
        let v5: __m256d;
        let v6: __m256d;
        let v7: __m256d;
        let v8: __m256d;
        let v9: __m256d;
        let v10: __m256d;
        let v11: __m256d;
        core::arch::asm!(
            "vxorpd ymm0,ymm0,ymm0",
            "vxorpd ymm1,ymm1,ymm1",
            "vxorpd ymm2,ymm2,ymm2",
            "vxorpd ymm3,ymm3,ymm3",
            "vxorpd ymm4,ymm4,ymm4",
            "vxorpd ymm5,ymm5,ymm5",
            "vxorpd ymm6,ymm6,ymm6",
            "vxorpd ymm7,ymm7,ymm7",
            "vxorpd ymm8,ymm8,ymm8",
            "vxorpd ymm9,ymm9,ymm9",
            "vxorpd ymm10,ymm10,ymm10",
            "vxorpd ymm11,ymm11,ymm11",
            ".p2align 5",
            "2:",
            "lea {pf},[{idx}+8*{ahead}]",
            "and {pf},4095",
            "prefetcht0 [{b}+{pf}+0*{ld}]",
            "prefetcht0 [{b}+{pf}+1*{ld}]",
            "prefetcht0 [{b}+{pf}+2*{ld}]",
            "prefetcht0 [{b}+{pf}+3*{ld}]",
            "prefetcht0 [{b}+{pf}+4*{ld}]",
            "prefetcht0 [{b}+{pf}+5*{ld}]",
            "mov {base},{orig}",
            "cmp {pf},{idx}",
            "cmovb {base},{next}",
            "prefetcht0 [{base}+{pf}*8+0]",
            "prefetcht0 [{base}+{pf}*8+128]",
            "prefetcht1 [{base}+{pf}*8+256]",
            "prefetcht1 [{base}+{pf}*8+384]",
            ".irp q,0,1,2,3,4,5,6,7",
            "vmovupd ymm12,[{a}+64*(\\q)]",
            "vmovupd ymm13,[{a}+64*(\\q)+32]",
            "vbroadcastsd ymm14,[{b}+{idx}+0*{ld}+8*(\\q)]",
            "vbroadcastsd ymm15,[{b}+{idx}+1*{ld}+8*(\\q)]",
            "vfmadd231pd ymm0,ymm12,ymm14",
            "vfmadd231pd ymm1,ymm13,ymm14",
            "vbroadcastsd ymm14,[{b}+{idx}+2*{ld}+8*(\\q)]",
            "vfmadd231pd ymm2,ymm12,ymm15",
            "vfmadd231pd ymm3,ymm13,ymm15",
            "vbroadcastsd ymm15,[{b}+{idx}+3*{ld}+8*(\\q)]",
            "vfmadd231pd ymm4,ymm12,ymm14",
            "vfmadd231pd ymm5,ymm13,ymm14",
            "vbroadcastsd ymm14,[{b}+{idx}+4*{ld}+8*(\\q)]",
            "vfmadd231pd ymm6,ymm12,ymm15",
            "vfmadd231pd ymm7,ymm13,ymm15",
            "vbroadcastsd ymm15,[{b}+{idx}+5*{ld}+8*(\\q)]",
            "vfmadd231pd ymm8,ymm12,ymm14",
            "vfmadd231pd ymm9,ymm13,ymm14",
            "vfmadd231pd ymm10,ymm12,ymm15",
            "vfmadd231pd ymm11,ymm13,ymm15",
            ".endr",
            "add {a},512",
            "add {idx},64",
            "cmp {idx},{bulk}",
            "jne 2b",
            ".irp t,0,1,2,3,4,5,6",
            ".if \\t < {tail}",
            "vmovupd ymm12,[{a}+64*(\\t)]",
            "vmovupd ymm13,[{a}+64*(\\t)+32]",
            "vbroadcastsd ymm14,[{b}+{idx}+0*{ld}+8*(\\t)]",
            "vbroadcastsd ymm15,[{b}+{idx}+1*{ld}+8*(\\t)]",
            "vfmadd231pd ymm0,ymm12,ymm14",
            "vfmadd231pd ymm1,ymm13,ymm14",
            "vbroadcastsd ymm14,[{b}+{idx}+2*{ld}+8*(\\t)]",
            "vfmadd231pd ymm2,ymm12,ymm15",
            "vfmadd231pd ymm3,ymm13,ymm15",
            "vbroadcastsd ymm15,[{b}+{idx}+3*{ld}+8*(\\t)]",
            "vfmadd231pd ymm4,ymm12,ymm14",
            "vfmadd231pd ymm5,ymm13,ymm14",
            "vbroadcastsd ymm14,[{b}+{idx}+4*{ld}+8*(\\t)]",
            "vfmadd231pd ymm6,ymm12,ymm15",
            "vfmadd231pd ymm7,ymm13,ymm15",
            "vbroadcastsd ymm15,[{b}+{idx}+5*{ld}+8*(\\t)]",
            "vfmadd231pd ymm8,ymm12,ymm14",
            "vfmadd231pd ymm9,ymm13,ymm14",
            "vfmadd231pd ymm10,ymm12,ymm15",
            "vfmadd231pd ymm11,ymm13,ymm15",
            ".endif",
            ".endr",
            ld=const (8*K),bulk=const (K/8*64),tail=const (K%8),
            ahead=const 32,
            a=inout(reg) a=>_,idx=inout(reg) 0usize=>_,b=in(reg) b,next=in(reg) next,
            orig=in(reg) a,pf=out(reg) _,base=out(reg) _,
            out("ymm0") v0,out("ymm1") v1,out("ymm2") v2,out("ymm3") v3,out("ymm4") v4,out("ymm5") v5,out("ymm6") v6,out("ymm7") v7,out("ymm8") v8,out("ymm9") v9,out("ymm10") v10,out("ymm11") v11,
            out("ymm12") _,out("ymm13") _,out("ymm14") _,out("ymm15") _,options(nostack,readonly),
        );
        _mm256_storeu_pd(c.add(0 * K + 0), v0);
        _mm256_storeu_pd(c.add(0 * K + 4), v1);
        _mm256_storeu_pd(c.add(1 * K + 0), v2);
        _mm256_storeu_pd(c.add(1 * K + 4), v3);
        _mm256_storeu_pd(c.add(2 * K + 0), v4);
        _mm256_storeu_pd(c.add(2 * K + 4), v5);
        _mm256_storeu_pd(c.add(3 * K + 0), v6);
        _mm256_storeu_pd(c.add(3 * K + 4), v7);
        _mm256_storeu_pd(c.add(4 * K + 0), v8);
        _mm256_storeu_pd(c.add(4 * K + 4), v9);
        _mm256_storeu_pd(c.add(5 * K + 0), v10);
        _mm256_storeu_pd(c.add(5 * K + 4), v11);
    }
}
