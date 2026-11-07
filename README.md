tensor4all fork of sarah-ek/gemm, MIT, upstream commit 86102c5

Playground for testing high performance matrix multiplication.

## What this fork changes

Upstream (`sarah-ek/gemm`) code, republished by tensor4all under the `t4a-*`
crate names. Beyond the rename, this fork carries:

- a hand-written **Apple Silicon SME** microkernel family for `f64`, `f32`,
  `c32` and `c64`, with a measured per-element-type size gate: large calls take
  SME, small ones stay on NEON (real types) or `neonfcma` (complex types);
- a per-thread packing buffer on the generic path, shared across calls once
  warm instead of allocated per call (kept up to the L2 size);
- `did_pack_lhs` storage sized by `min(m, mc)` and kept inline up to 64
  entries, so a warm call does not allocate.

The upstream algorithms and benchmark harness are otherwise unchanged.

### Apple Silicon SME path (fork-only)

Selected at runtime by `HasAppleSme` and **compiled only** under

    all(target_vendor = "apple", target_arch = "aarch64", feature = "apple-sme")

The `apple-sme` feature is on by default and is a no-op elsewhere: on any other
target, or with `--no-default-features`, the kernels are not built at all (no
`_t4a_gemm_*_sme_ukr` symbol ends up in the rlib).

The runtime probe additionally requires `FEAT_SME`, `FEAT_SME_F64F64`
(`f64`/`c64`) or `FEAT_SME_F32F32` (`f32`/`c32`), `svl == 64` and macOS 26 or
newer; otherwise the gate keeps the call on NEON for real types and `neonfcma`
for complex types, the same kernels used below the size thresholds.

SME is an Armv9 feature, not an Apple one, but this implementation is
Apple-Silicon-only on purpose: Linux SME and any future non-Apple
implementation have not been validated here, and the `ZA` / streaming-mode
save-restore on a context switch that the `k` loop depends on is an OS
responsibility that has only been measured on Apple. Generic `aarch64`
therefore never takes SME: real types dispatch to NEON, complex types to
`neonfcma` when the CPU has FCMA (and to the scalar kernel when NEON is not
detected).

### Verified hosts

- `aarch64-apple-darwin`, macOS 26.7, Apple M5 Max, rustc 1.96.0: build and the
  full test suite pass with default features (SME kernels executed) and with
  `--no-default-features --features std,rayon` (SME path compiled out).
- `aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu`: `cargo check`
  only, never run.
