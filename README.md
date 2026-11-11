tensor4all fork of sarah-ek/gemm, MIT, upstream commit 86102c5

Playground for testing high performance matrix multiplication.

## What this fork changes

These crates are upstream `gemm` at commit `86102c5` (0.19.0) plus the changes
below, republished under `t4a-*` names. If you are deciding whether to depend on
this fork rather than upstream, this section lists every difference in behaviour
and public API as of `t4a-gemm-common` 0.20.4. `git diff 86102c5 HEAD` is the
authority; code not listed here is upstream's. Upstream has no commits after
`86102c5`, so the fork does not lag behind it.

Three of the changes apply on every target, not only on Apple Silicon:

### 1. A larger `kc` on every target

`kernel_params` (`gemm-common/src/cache.rs:721-729`) chooses how deep each `k`
block is. Upstream takes `kc` from an L1 set-conflict formula with a floor of
512. The fork also raises it to `l2_cache_bytes / (sizeof * (8 * mr + nr))`, the
largest depth at which one `A` macropanel (at most `8 * mr` rows) and one `B`
micropanel fit in L2, still capped at `k`. The change is not behind any `cfg`.
It changes the blocking of every kernel on every target — x86 `avx512f`/`fma`,
NEON, scalar and SME, and `t4a-gemm-f16`, which calls the same function
(`gemm-f16/src/gemm.rs:380`) — whenever that bound exceeds upstream's `kc` and
`k` is deeper than upstream's `kc`.

- Why: every extra `k` chunk costs a microkernel call plus a read-modify-write
  pass over the block. On an Apple M5 Max that was about 0.16 us, roughly 20% of
  a `kc = 512` block with the `f64` SME kernel; `n = k = 2048` at 1 thread went
  from 352 to 422 GFLOP/s (commit `5615de0`).
- Not promised: it was measured on that host only. The same commit records
  `f32`/`c32`/`c64` (NEON kernels there) within +-2%; no x86 or non-Apple
  `aarch64` measurement of this change is recorded. Because the `k` sum is
  grouped into different chunks, results can differ from upstream's in rounding.

### 2. A per-thread packing buffer (with `std`)

Upstream allocates the packed-`rhs` / prepacked-`lhs` buffer on every call of
`gemm_basic_generic` that packs. With `std`, the fork moves it out of a thread-local slot
and hands it back on drop, so the next call on the same thread reuses it when it
is large enough (`PACK_BUF` and `PackBuffer`, `gemm-common/src/gemm.rs:66-129`).

- Why: tenferro-rs#2448 and #2472 measured two heap allocations per call for
  small factorization-sized products on `aarch64` (`gemm-f64/tests/alloc_count.rs`).
- Not promised: a buffer larger than the L2 size (`CACHE_INFO[1]`) is not kept,
  so such products still allocate per call. A nested call on the same thread (a
  rayon worker that takes another product while it waits) allocates its own.
  Without `std` nothing changes. Each thread that has run a product keeps up to
  one L2-sized buffer alive afterwards, in addition to the `L2_SLAB` upstream
  already keeps per thread.

### 3. `did_pack_lhs` sized by `ceil(min(m, mc) / MR)`, inline up to 64

Upstream allocates `mc / MR` flags per call, and again per worker thread. The
fork allocates `ceil(min(m, mc) / MR)` and stores up to 64 inline
(`DidPackLhs`, `gemm-common/src/gemm.rs:138-171`, used at `:609` and `:778`). No
arithmetic, packing decision or loop bound changes (commits `cb2d33c`, `71c9848`).

- Why: a small `k` makes the cache-derived `mc` large (5461 zeroed flags per
  `48 x 48` `f64` call on an 8 MiB L2), while sizing by `m` alone made the
  per-row-chunk `fill` quadratic for tall products (+18% at `1000000 x 32 x 32`).
- With changes 2 and 3, a warm call allocates nothing for the shapes
  `gemm-f64/tests/alloc_count.rs` checks: `f64`, `m`, `n`, `k` up to 96, with
  `Parallelism::None` and `Rayon(1)`. That is a test of those shapes, not a
  guarantee for every call.

### 4. Apple Silicon SME kernels (only on Apple `aarch64` with `apple-sme`)

Hand-written SME microkernels for `f64`, `f32`, `c32` and `c64`, behind a
measured size gate: a threshold table per element type on the shape `(m, n, k)`
(`gemm-common/src/sme_thresholds.rs`, applied by `__inject_sme_gate!` in
`gemm-common/src/gemm.rs`), so a large call takes SME and a small one takes the
NEON (`f32`/`f64`) or `neonfcma` (`c32`/`c64`) kernel.

They are compiled only under

    all(target_vendor = "apple", target_arch = "aarch64", feature = "apple-sme")

and the `apple-sme` feature is a no-op elsewhere: on any other target, or with
`--no-default-features`, no `_t4a_gemm_*_sme_ukr` symbol is built. At run time
the probe (`HasAppleSme`, `gemm-common/src/cache.rs`) also requires `FEAT_SME`,
`FEAT_SME_F64F64` (`f64`/`c64`) or `FEAT_SME_F32F32` (`f32`/`c32`),
`svl == 64` and macOS 26 or newer; otherwise the kernels below are used.

SME is an Armv9 feature, not an Apple one, but this implementation is
Apple-only on purpose: Linux SME and any non-Apple implementation have not been
validated, and saving and restoring `ZA` and the streaming-mode state across a
context switch, which the `k` loop depends on, is the OS's job and has only been
measured on Apple (macOS 26, M5 Max).

- Not promised: commit `5615de0` records a known gap on this path, about 20%
  slower at `n = 512..1024`, 6 threads, `k = 512`.

### 5. Public API differences

- Crate names are `t4a-gemm`, `t4a-gemm-common`, `t4a-gemm-f16`, `-f32`,
  `-f64`, `-c32` and `-c64`, at 0.20.x; the library names are upstream's
  (`gemm`, `gemm_common`, `gemm_f64`, ...), so with
  `gemm = { package = "t4a-gemm", ... }` the `use` paths do not change.
- `apple-sme` is a default feature of `t4a-gemm`, `-f32`, `-f64`, `-c32` and
  `-c64` (not `-f16`), and it enables `std`.
- `gemm_common::gemm_def!` and `gemm_common::gemm_cplx_def!` (both
  `#[macro_export]`) take two more arguments, the SME probe and the threshold
  table, so code that invokes them directly does not compile unchanged.
- New public items: `gemm_common::sme_thresholds` (every target) and
  `gemm_common::cache::HasAppleSme` (Apple `aarch64` with `apple-sme` only).
- `gemm_c32::microkernel` and `gemm_c64::microkernel` are `pub`; upstream keeps
  them private (`f32`/`f64` were already `pub`).

The benchmark harness (`gemm/benches/bench.rs`) is unchanged from upstream.

### Which kernel a call takes

The order of the checks in `init_gemm_fn` (`gemm-common/src/gemm.rs:1233-1288`,
`f32`/`f64`) and `init_gemm_cplx_fn` (`:1362-1390`, `c32`/`c64`). Only the two
Apple SME rows are the fork's; the rest is upstream's dispatch.

| Target | `f32`, `f64` | `c32`, `c64` |
|---|---|---|
| `x86`, `x86_64` | `avx512f` (only with the `x86-v4` feature), else `fma`, else `scalar` | `avx512f` (only with `x86-v4`), else `fma`, else `scalar_cplx` |
| `aarch64`, Apple, `apple-sme`, probe passes | size gate: SME or NEON | size gate: SME or `neonfcma`, with no separate FCMA check |
| `aarch64`, otherwise | AMX if `experimental-apple-amx` is on and the probe finds an M1/M2/M3 (then at every size), else NEON; `scalar` if NEON is not detected | `neonfcma` if NEON **and** FCMA are detected, else `scalar_cplx` |
| `wasm32` | `simd128` if detected, else `scalar` | `scalar_cplx` (there is no `wasm32` branch) |
| any other | `scalar` | `scalar_cplx` |

So a generic `aarch64` CPU without FCMA runs complex products on the scalar
kernel, and on an SME host a call below the gate stays on NEON even with AMX
compiled in. `t4a-gemm-f16` keeps upstream's dispatch; only `kc` (change 1)
differs there.

### What this fork does not promise

- Results bitwise identical to upstream's (change 1 regroups the `k` sum).
- The measured speed-ups anywhere but the Apple M5 Max they were measured on.
- SME on any target other than Apple `aarch64` with macOS 26 or newer.
- Zero allocation beyond the shapes `alloc_count.rs` checks.

### Verified hosts

- `aarch64-apple-darwin`, macOS 26.7, Apple M5 Max, rustc 1.96.0: build and the
  full test suite pass with default features (SME kernels executed) and with
  `--no-default-features --features std,rayon` (SME path compiled out).
- `x86_64` Linux and Windows: CI (`.github/workflows/run-tests.yml`, nightly on
  `ubuntu-latest` and `windows-latest`) runs the `t4a-gemm` package's tests
  only, not the per-type crates' (`alloc_count.rs`, `sme.rs`, `za_context.rs`).
- `x86_64-unknown-linux-gnu`, AMD Ryzen AI MAX+ 395, rustc 1.99.1, 2026-11-11:
  the tests of `t4a-gemm-common`, `-f16`, `-f32`, `-f64`, `-c32` and `-c64` pass
  (the SME microkernel tests are compiled out there; their reference comparisons
  run through the x86 kernels). The `t4a-gemm` package's own tests were not run
  on that host: its `diol` dev-dependency needs the fontconfig headers.
- `aarch64-unknown-linux-gnu`: `cargo check` only, never run.
