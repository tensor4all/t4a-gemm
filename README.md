tensor4all fork of sarah-ek/gemm, MIT, upstream commit 86102c5

Playground for testing high performance matrix multiplication.

## What this fork changes

Upstream (`sarah-ek/gemm`) code, republished by tensor4all under the `t4a-*`
crate names. Beyond the rename, this fork carries:

- a hand-written **Apple Silicon SME** microkernel family for `f64`, `f32`,
  `c32` and `c64`, behind a measured size gate: a threshold table per element
  type, on the shape `(m, n, k)` and not on the element type alone
  (`gemm-common/src/sme_thresholds.rs:47-67`, called at
  `gemm-common/src/gemm.rs:1153`), so a large call takes SME and a small one
  falls back as described below;
- a per-thread packing buffer on the generic path, shared across calls once
  warm instead of allocated per call (kept up to the L2 size);
- `did_pack_lhs` storage sized by `min(m, mc)` and kept inline up to 64
  entries, so a warm call does not allocate.

The benchmark harness (`gemm/benches/bench.rs`) is unchanged from upstream
`86102c5`. Beyond the SME kernels and their tests, the SME size gate, the
packing changes listed above and the crate rename, the code is upstream's.

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

The fallback is not "NEON" in general. On `aarch64` the AMX kernel is tested
before NEON when it is compiled in (`experimental-apple-amx`) and the probe
finds an M1/M2/M3 (`gemm-common/src/cache.rs:198-216`), and once selected it
takes every size, not only the large ones (`gemm-common/src/gemm.rs:1261-1265`);
on an SME host the gate comes first, so a small call there stays on NEON even
with AMX compiled in. Away from the gate the complex kernels also need
**both** `neon` and `fcma` for `neonfcma`, otherwise they take the scalar
kernel (`gemm-common/src/gemm.rs:1385-1390`); inside the gate, the fallback is
`neonfcma` whenever the SME probe passes, with no separate FCMA test (`:1423`).
Off `aarch64` the kernels are upstream's:
`avx512f`/`fma`/`scalar` on `x86`/`x86_64` (`:1234-1245`), `simd128`/`scalar`
on `wasm32` for the real types (`:1271-1278`) and `scalar` on any other
target (`:1280-1288`). The complex dispatch has no `wasm32` branch, so
complex types take the scalar kernel there whatever `simd128` says
(`:1362-1390`).

### Verified hosts

- `aarch64-apple-darwin`, macOS 26.7, Apple M5 Max, rustc 1.96.0: build and the
  full test suite pass with default features (SME kernels executed) and with
  `--no-default-features --features std,rayon` (SME path compiled out).
- `aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu`: `cargo check`
  only, never run.
