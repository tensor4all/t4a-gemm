//! Check the ordered FMA result through the opt-in API, including edge tiles and unaligned C.
#![cfg(all(feature = "std", target_arch = "x86_64"))]

use gemm::try_gemm_square_f64;

#[test]
fn compact_squares_and_output_guards() {
    for n in [510, 514] {
        let mut canary = 17.0;
        assert!(!unsafe { try_gemm_square_f64(n, &0.0, &0.0, &mut canary) });
        assert_eq!(canary, 17.0);
    }
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
        return;
    }
    for n in [511usize, 512, 513] {
        let a: Vec<f64> = (0..n * n)
            .map(|i| ((i * 7 % 101) as f64 - 49.0) / 13.0)
            .collect();
        let b: Vec<f64> = (0..n * n)
            .map(|i| ((i * 13 % 89) as f64 - 41.0) / 19.0)
            .collect();
        let mut expected = vec![0.0f64; n * n];
        for (column, rhs) in expected.chunks_exact_mut(n).zip(b.chunks_exact(n)) {
            for (&v, lhs) in rhs.iter().zip(a.chunks_exact(n)) {
                for (sum, &u) in column.iter_mut().zip(lhs) {
                    *sum = u.mul_add(v, *sum);
                }
            }
        }
        let canary = f64::from_bits(0x7ff8_0000_0000_3137);
        let mut storage = vec![canary; n * n + 10];
        let offsets = if n == 512 { 8 } else { 2 };
        for start in 1..=offsets {
            storage.fill(canary);
            let end = start + n * n;
            let dst = &mut storage[start..end];
            assert!(unsafe { try_gemm_square_f64(n, a.as_ptr(), b.as_ptr(), dst.as_mut_ptr()) });
            assert!(
                dst.iter()
                    .zip(&expected)
                    .all(|(u, v)| u.to_bits() == v.to_bits()),
                "n={n}, offset={start}"
            );
            assert!(storage[..start]
                .iter()
                .chain(&storage[end..])
                .all(|v| v.to_bits() == canary.to_bits()));
        }
    }
}
