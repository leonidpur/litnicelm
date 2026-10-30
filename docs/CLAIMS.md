# Claims register

Every public statement about LitniceLM should be traceable to a precise entry
here. Each entry records its conditions, verification basis, evidence, and a
reproduction command. Benchmark speed and memory are deliberately absent until
the locked-clock comparison is complete.

## C1: Independent C++/CUDA engine with pluggable backends

- **Claim:** LitniceLM implements training and inference in its own C++17
  engine and selects tensor implementations through dynamically loaded backend
  plugins. CUDA/cuBLAS, CPU, and OpenBLAS backends are supported when their
  build dependencies are available.
- **Conditions:** Source and build configuration at the commit being described;
  optional plugins depend on locally installed libraries.
- **Verification basis:** Successful clean build plus inspection of the plugin
  targets and runtime backend interface. This is an implementation claim, not
  a performance claim.
- **Evidence:** `CMakeLists.txt`,
  `src/model/backend/backend_plugin_api.hpp`, `src/model/`, and
  `src/backend_libs/`.
- **Reproduce:**

  ```bash
  cmake -S . -B build
  cmake --build build -j
  ./build/litnicelm --help
  ```

## C2: FP32 validation-loss parity with PyTorch

- **Claim:** For the configured RoPE FP32 Doyle export and identical validation
  windows, PyTorch and LitniceLM validation losses differ by less than `1e-7`.
- **Conditions:** The exact workload, corpus binary, export directory, and
  evaluation policy in `tools/vs_bench/run.py`; the current recorded losses are
  `2.434160013` and `2.434160000`, an absolute difference of `1.35e-8`.
- **Standard:** PyTorch cross-entropy evaluation of the exported safetensors
  weights over the same token windows, with FP32 execution and TF32 disabled.
- **Evidence:** The configured result is recorded in `docs/COMPARISON.md`; each
  reproduction writes its structured result, exact spec, environment, and Git
  commit to the dated raw benchmark directory.
- **Reproduce:**

  ```bash
  cd tools
  ../.venv/bin/python -m vs_bench
  ```

## C3: BF16 mixed precision with FP32 master weights

- **Claim:** `precision: bf16_mixed` stores activations and activation gradients
  in BF16 while retaining FP32 parameters/master weights and optimizer state;
  BF16 GEMMs accumulate into FP32.
- **Conditions:** CUDA backend with BF16-activation kernel capability.
- **Standard:** The declared precision policy and tensor roles are checked
  against dtype-specific layouts and the CUDA backend's BF16/FP32 behavior.
- **Evidence:** `tests/precision_policy_test.cpp`,
  `tests/compute_weights_test.cpp`, and `tests/backend_ops_test.cpp`.
- **Reproduce:**

  ```bash
  cmake -S . -B build
  cmake --build build -j
  ctest --test-dir build --output-on-failure -R 'precision_policy|compute_weights|backend_ops_litnice_backend_cublas_plugin'
  ```

## C4: Safetensors export readable by the official implementation

- **Claim:** `litnicelm export` produces `model.safetensors` that can be opened
  and enumerated by the official Python `safetensors` reader.
- **Conditions:** A valid LitniceLM checkpoint and its matching YAML config.
- **Standard:** Successful parsing and tensor loading through
  `safetensors.safe_open`, independently of LitniceLM's writer.
- **Evidence:** `tools/hf_upload/verify_export.py`; the native export format is
  also covered by the `hf_export` CTest.
- **Reproduce:**

  ```bash
  ./build/litnicelm export --config <config.yaml> --out <export-dir> --checkpoint <checkpoint>
  tools/hf_upload/verify_export.py <export-dir>
  ```

## C5: Reference, BF16 conversion, and train/resume/infer tests

- **Claim:** The test suite compares backend operations with a host reference,
  exhaustively checks BF16 conversion on the CUDA backend, and exercises
  known-answer FP32/BF16 training, resume, cross-precision resume, and
  inference paths.
- **Conditions:** Tests requiring CUDA run only when the CUDA plugin is built;
  exhaustive conversion is labelled `slow`.
- **Standard:** Per-operation numerical comparison, exhaustive input-domain
  conversion checks, and fixed-seed end-to-end expected ranges and resume
  invariants.
- **Evidence:** `tests/backend_ops_test.cpp`,
  `tests/cast_exhaustive_test.cpp`, and `tests/train_infer_e2e.sh`.
- **Reproduce:**

  ```bash
  cmake -S . -B build
  cmake --build build -j
  ctest --test-dir build --output-on-failure -R 'backend_ops|cast_exhaustive|train_infer_e2e'
  ```
