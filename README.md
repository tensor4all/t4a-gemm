tensor4all fork of sarah-ek/gemm, MIT, upstream commit 86102c5

Playground for testing high performance matrix multiplication.

## What this fork changes

Upstream (`sarah-ek/gemm`) code, republished by tensor4all under the `t4a-*`
crate names. The upstream algorithms, the NEON/x86 dispatch and every benchmark
are unchanged; the only functional addition is a hand-written **Apple Silicon
SME** microkernel family for `f64`, `f32`, `c32` and `c64`, plus the `t4a-*`
rename.

### Apple Silicon SME path (fork-only)

Selected at runtime by `HasAppleSme` and **compiled only** under

    all(target_vendor = "apple", target_arch = "aarch64", feature = "apple-sme")

The `apple-sme` feature is on by default and is a no-op elsewhere: on any other
target, or with `--no-default-features`, the kernels are not built at all (no
`_t4a_gemm_*_sme_ukr` symbol ends up in the rlib).

The runtime probe additionally requires `FEAT_SME`, `FEAT_SME_F64F64`
(`f64`/`c64`) or `FEAT_SME_F32F32` (`f32`/`c32`), `svl == 64` and macOS 26 or
newer; otherwise dispatch stays on NEON.

SME is an Armv9 feature, not an Apple one, but this implementation is
Apple-Silicon-only on purpose: Linux SME and any future non-Apple
implementation have not been validated here, and the `ZA` / streaming-mode
save-restore on a context switch that the `k` loop depends on is an OS
responsibility that has only been measured on Apple. Generic `aarch64`
therefore always takes the NEON path.

### Verified hosts

- `aarch64-apple-darwin`, macOS 26.7, Apple M5 Max, rustc 1.96.0: build and the
  full test suite pass with default features (SME kernels executed) and with
  `--no-default-features --features std,rayon` (SME path compiled out).
- `aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu`: `cargo check`
  only, never run.
