# Claims register

Every public statement about LitniceLM should be traceable to a precise entry
here. Each entry records its conditions, verification basis, evidence, and a
reproduction command. Speed and memory entries (C6-C10) come from one
locked-clock publication run; each lists its evidence.

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

## Speed and memory: shared conditions (C6-C10)

- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU (AD106, 36 SMs, 8 GB),
  SM clock locked at 1500 MHz (`nvidia-smi -lgc 1500,1500`, verified under
  load for every measured process); memory clock not locked (8001 MHz
  nominal, sampled under load and recorded per row). Intel Core i7-13620H.
- **Host power:** on AC, platform profile `balanced`, CPU governor
  `powersave`, EPP `balance_performance`; checked before the run and recorded
  for every measured process (no row flagged).
- **Software:** Linux 6.14.0-37, NVIDIA driver 580.95.05. LitniceLM commit
  d01241e, Release build, CUDA 12.2 (nvcc), cuDNN 9.28.1, cudnn-frontend
  1.17.0. PyTorch 2.15.0+cu132 (CUDA 13.2, cuDNN 9.26), Triton 3.9.0,
  Python 3.13.3.
- **Workload:** 4 layers, 4 heads, d_model 256, d_ff 1024, RoPE, vocabulary
  512; batch 1 × 1024 tokens, 4 batches per optimizer step (4,096 tokens per
  step); AdamW, global-norm clip 0.06, cosine LR; Conan Doyle corpus;
  214 optimizer steps (2 epochs) per measured process; 3 processes per
  variant. TF32 off in both engines. BF16 means BF16 activations with FP32
  parameters, optimizer state and GEMM accumulation, in both engines.
- **Timing:** steady-state tokens/s, median of 3 processes. LitniceLM: the
  `[SteadyState]` wall time of every batch after the first 5 optimizer steps
  (the first step includes cuDNN graph build and plan selection). PyTorch:
  wall time after 5 warm-up steps; compile time excluded. LitniceLM's figure
  agrees within 0.3% with an independent epoch differential (1 vs 3 epochs,
  process wall time) in every LitniceLM row.
- **Loss read:** LitniceLM reads every batch's loss on the host (one GPU
  synchronization per batch). The best PyTorch rows, compiled and eager, read
  the loss only at the end of the run; reading it every batch costs eager
  PyTorch 6% (FP32) and 3% (BF16) in this run.
- **Training equivalence:** every row's final training loss is within 0.02
  of LitniceLM's (same repeat and precision). Observed: at most 0.0046;
  LitniceLM 3.6207–3.6212, PyTorch 3.6171–3.6179 in both precisions.
- **Evidence:** [`run_180749_results.json`](run_180749_results.json): spec,
  environment with versions, `git_commit` d01241e and `git_dirty: false`,
  and every measured row. The per-process logs and PyTorch request files are
  kept outside the repository with the raw benchmark data.
- **Reproduce:** not yet possible from a fresh clone: the corpus, its
  tokenizer artifacts and tokenized dataset, the untracked base config under
  `conf/local/`, the parity export and a pinned Python environment are not in
  the repository ([COMPARISON.md](COMPARISON.md#reproduce) lists each). On a
  machine with those inputs, build as in [TRY_CUDA.md](TRY_CUDA.md) (which
  does not yet cover cuDNN, needed by `max_fused`), with AC power, the
  `balanced` platform profile and passwordless sudo for `/usr/bin/nvidia-smi`;
  the run checks all three before measuring:

  ```bash
  git checkout d01241e
  cd tools
  ../.venv/bin/python -m vs_bench
  ```

## C6: FP32 training throughput 1.04× best compiled PyTorch

- **Claim:** In FP32, LitniceLM `preset:portable_fused` trains at 121,428
  tokens/s: 1.04× the best compiled PyTorch mode (`max-autotune`, 116,847)
  and 1.15× best eager PyTorch (loss read at end, 105,740).
- **Conditions:** The shared conditions above. Ranges over 3 processes:
  LitniceLM 121,341–121,548; PyTorch `max-autotune` 116,814–117,174. The
  ranges do not overlap.
- **Standard:** PyTorch 2.15 on the same model, data and optimizer, eager
  (loss read every batch and at end) and `torch.compile` in every mode
  (`default`, `reduce-overhead`, `max-autotune`,
  `max-autotune-no-cudagraphs`); the best of each is taken.
- **Evidence and reproduce:** shared conditions above.

## C7: BF16 training throughput on par with best compiled PyTorch, 2.48× eager

- **Claim:** In BF16 mixed precision, LitniceLM `preset:max_fused` trains at
  335,711 tokens/s: on par with the best compiled PyTorch mode
  (`reduce-overhead`, 328,500; ratio 1.02×) and 2.48× best eager PyTorch
  (loss read at end, 135,167).
- **Conditions:** The shared conditions above. Ranges over 3 processes:
  LitniceLM 335,186–336,239; PyTorch `reduce-overhead` 327,202–328,515,
  `max-autotune` 327,256–336,171. "On par", not "faster": PyTorch's
  `max-autotune` processes reach LitniceLM's range, so a 2% lead is not
  established. The eager ratio holds for the recorded host state: with
  identical software, PyTorch's BF16 eager throughput was ~108k tokens/s
  on an earlier boot and ~133k on this one.
- **One-time cost (not in the throughput):** LitniceLM cuDNN plan selection,
  0.24 s per process (building and timing every plan of the forward and
  backward attention graphs). PyTorch compile time is likewise excluded.
- **Standard:** As C6.
- **Evidence and reproduce:** shared conditions above.

## C8: Peak GPU memory per process

- **Claim:** Peak GPU memory per process, CUDA context included: BF16,
  LitniceLM `max_fused` 292 MiB vs PyTorch best compiled 340 MiB (0.86×);
  FP32, LitniceLM `portable_fused` 454 MiB vs PyTorch best compiled 356 MiB
  (1.28×).
- **Conditions:** The shared conditions above; median of 3 processes. The
  figure includes each process's CUDA context, and the engines load
  different CUDA runtimes (LitniceLM 12.2, PyTorch 13.2), so part of the
  difference is context size. LitniceLM's own allocations are also larger:
  its managed arenas total 304 MiB (FP32) and 129 MiB (BF16), against a
  PyTorch allocator peak of 59 MiB (`torch.cuda.max_memory_allocated`, best
  compiled mode); the two are measured differently. The temp arena keeps
  every layer's work buffers and unused transpose buffers (open backlog
  item).
- **Standard:** `nvidia-smi --query-compute-apps=pid,used_memory`, sampled
  every 100 ms for the measured process.
- **Evidence and reproduce:** shared conditions above.

## C9: Model FLOPs utilization

- **Claim:** At 1500 MHz, LitniceLM reaches 39.2% MFU in BF16 (`max_fused`)
  and 28.3% in FP32 (`portable_fused`); PyTorch's best compiled mode reaches
  38.3% and 27.3%.
- **Formula:** MFU = tokens/s × F / peak, with F the training FLOPs per token
  of the PaLM paper (Chowdhery et al. 2022, appendix B):
  F = 6N + 12·L·d·S, where N = 3,276,800 matmul weights
  (4 × (256·768 + 256·256 + 2·256·1024) + 256·512 for the output head;
  embedding lookup and vector parameters excluded), L = 4, d = 256,
  S = 1024, so F = 32,243,712.
- **Peak:** 36 SMs × 1.5 GHz × FLOPs per SM per clock. BF16 tensor cores with
  FP32 accumulate: 512, so 27.65 TFLOPS. FP32 on CUDA cores (no TF32): 256,
  so 13.82 TFLOPS. Per-SM rates from the NVIDIA Ada Lovelace architecture
  whitepaper (RTX 4090: 165.2 TFLOPS BF16 with FP32 accumulate and
  82.6 TFLOPS FP32, over 128 SMs at 2520 MHz).
- **Note, causal attention:** the PaLM formula counts attention FLOPs without
  the causal mask. Counting only the unmasked half, F = 6N + 6·L·d·S =
  25,952,256, and MFU is 31.5% (BF16) and 22.8% (FP32) for LitniceLM, 30.8%
  and 21.9% for PyTorch.
- **Evidence and reproduce:** throughput from C6 and C7.

## C10: Fusion ladder

- **Claim:** Each fusion step raises LitniceLM throughput (steady-state
  tokens/s):

  | variant | adds | bf16 | fp32 |
  |---|---|---:|---:|
  | `unfused` | per-head GEMMs; separate scale, mask and softmax | 184,100 | 106,132 |
  | `fused_inplace` | one fused scaled causal softmax | 213,004 (+15.7%) | 116,933 (+10.2%) |
  | `preset:portable_fused` | + cuBLASLt fused-linear projections and FFN | 226,233 (+6.2%) | 121,428 (+3.8%) |
  | `preset:max_fused` | + cuDNN flash attention (BF16 only) | 335,711 (+48.4%) | — |

- **Conditions:** The shared conditions above. `cudnn_flash` alone (with
  unfused projections and FFN) gives 307,715 in BF16.
- **Evidence and reproduce:** shared conditions above.
