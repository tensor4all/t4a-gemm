# Mixed real GEMM with f64 arithmetic (colmat-rs#293)

## Decisions

- Support only `f32 × f32 → f64` and `f32 × f64 → f64`, with `Dst = Acc = f64`. The public
  `MixedRealGemmToF64` trait is sealed to those pairs; CCR and every other dtype/domain combination
  remain outside this change.
- Share the existing blocked driver through typed packing. Homogeneous operands use a blanket
  identity implementation that delegates to the original packer. Every `f32` operand is forced into
  packed `f64` storage, converted exactly with `f64::from`, and then multiplied and accumulated by
  the existing `f64` microkernels. There is no whole-matrix conversion or per-pair microkernel.
- Mixed calls bypass homogeneous GEMV, horizontal and other early routes. The low-level facade keeps
  `dst := alpha * dst + beta * lhs * rhs`, handles zero depth and negative strides, and does not swap
  `f32 × f64` into an unsupported transposed pair. Scratch and the optional caller stack are sized in
  packed `f64` elements. `PackOperand` is sealed so downstream users cannot add custom packing
  implementations. Allocation wording distinguishes one-time process-wide kernel/cache
  initialization from per-call working memory; only the latter is promised to be zero with a
  caller-provided stack after initialization.
- The internal dynamic dispatcher returns `UnsupportedMixedGemm` before shape handling or any
  destination access. The sealed public wrappers reject unsupported pairs at compile time instead.
  A short caller stack also panics before output mutation for `Parallelism::None`; with Rayon, the
  supplied stack is ignored.
- `K` blocking can change `f64` summation order. Results are not block-size independent, and bitwise
  equality across execution routes or thread counts is not promised. Tests use independent naive
  `f64` references with scale-aware tolerances; the colmat layer adds an `fx128` oracle.
- No performance claim is made. The mixed route is a correctness/API slice, not a measured adoption
  against another backend.

## Verification conclusions and constraints

Host: Linux x86_64, Rust 1.96.0. The host lacks `pkg-config`; the supported
`RUST_FONTCONFIG_DLOPEN=1` setting was used for the facade's test dependency graph. No registry
source was edited.

- The mixed integration suite passes in the default build and with `x86-v4`: exact promotion before
  multiplication, both supported pairs, pure Add, zero-scale old output, ignored nonzero Replace
  alpha, empty/zero-depth products, blocked `K`, row/column-major, padded layouts, independent
  destination-row, destination-column, and lhs-depth reversals, Rayon workers (including an empty
  caller stack), caller-stack equality, short-stack output atomicity, and no per-call allocation
  after process-wide initialization.
- Exact extreme-value checks pass for the minimum positive `f32` subnormal, `f32::MIN_POSITIVE`, and
  `f32::MAX`, in both supported pairs. Their products are finite and nonzero in `f64`.
- The internal unsupported-pair test returns an error and leaves the destination bitwise unchanged.
- Existing caller-stack tests remain bit-for-bit unchanged, and the new public rustdoc examples run.
- The full workspace suite passes after the review updates, and passes again after the final
  packer/facade lint corrections and review-coverage additions: 6 facade unit tests including
  dynamic rejection, 9 mixed integration tests, 3 caller-stack tests, and 4 rustdoc examples. The
  `x86-v4` mixed suite also passes all 9 tests. The no-default-features workspace build passes as
  well.
- Rustdoc builds with warnings denied. `cargo fmt --all -- --check` still reports only the inherited
  formatting differences in `gemm-common/src/sme_thresholds.rs` and
  `gemm-f64/examples/did_pack_sizing.rs`. Workspace clippy with warnings denied still fails on
  unchanged upstream diagnostics; the new packer/facade/test diagnostics were corrected, and a
  changed-file diagnostic pass finds only inherited warnings after those corrections.
- The `x86-v4` result is a feature-build and runtime-selection check on this host, not proof that
  every SIMD path was forced. Non-x86 targets, Apple SME/AMX, and wasm have not been executed.
