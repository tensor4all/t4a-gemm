pub mod f64 {
    type T = f64;
    gemm_common::gemm_def!(f64, 1, get, gemm_common::sme_thresholds::F64);
}
