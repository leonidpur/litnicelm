# LitniceLM vs PyTorch

Two questions: does LitniceLM compute the same model as an external
reference, and how fast and memory-hungry is it next to PyTorch?

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop (AD106, 36 SMs, 8 GB), driver 580.95.05, SM clock locked at 1500 MHz |
| Host | Intel Core i7-13620H, Linux 6.14.0-37, on AC, platform profile `balanced` |
| LitniceLM | commit d01241e, Release build, CUDA 12.2 (nvcc), cuDNN 9.28.1, cudnn-frontend 1.17.0 |
| PyTorch | 2.15.0+cu132 (CUDA 13.2, cuDNN 9.26), Triton 3.9.0, Python 3.13.3 |
| Model | 4 layers, 4 heads, d_model 256, d_ff 1024, RoPE (and learned positions for parity), vocabulary 512 (SentencePiece) |
| Data | Conan Doyle corpus, 485,847 tokens; validation: 1 of every 10 chunks of 8,192 tokens |
| Training | batch 1 × 1024 tokens, 4 batches per optimizer step, AdamW (β 0.9/0.95, ε 1e-8, weight decay 0.1 on embeddings and matrices), global-norm clip 0.06, LR 6e-4 cosine to 10% after 107 warmup steps |

Config: `conf/local/doyle/doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4.yaml`
(the learned-position variant replaces `rope` with `learned`).

## Correctness: same weights, same data, same loss

The `vs_bench` parity vertical loads a `litnicelm export` (`model.safetensors`
and `config.json`) into a PyTorch model written from `config.json` alone. It
evaluates exactly the validation windows LitniceLM uses (37 batches, 37,888
tokens), averaging per-batch mean cross-entropy as LitniceLM does.

| checkpoint | evaluated as | PyTorch | LitniceLM | difference |
|---|---|---|---|---|
| RoPE, trained fp32 | fp32 | 2.4341600 | 2.4341600 | 1.4e-8 |
| learned, trained fp32 | fp32 | 3.5777680 | 3.5777678 | 1.6e-7 |
| RoPE, trained bf16_mixed | bf16 autocast | 2.4286395 | 2.4287624 | 1.2e-4 |
| learned, trained bf16_mixed | bf16 autocast | 3.5783493 | 3.5783910 | 4.2e-5 |

The currently configured `vs_bench` run reproduces the RoPE FP32 row: its
measured difference is 1.35e-8 with a tolerance of 1e-5. The learned-position
and BF16 rows predate the standalone runner and remain historical until they
are represented as explicit parity settings. The BF16 rows compare two
different BF16 implementations and are informational.

```bash
cd tools
../.venv/bin/python -m vs_bench
```

The corpus, export directory, tolerance, and all other conditions are the
hard-coded `SPEC` in `tools/vs_bench/run.py`; there is intentionally no CLI.

## Speed

One locked-clock run (`run_180749`, 2026-11-23) measured every LitniceLM
variant and every PyTorch mode in the same session: 3 processes each, 214
optimizer steps of 4,096 tokens per process. Results, spec and environment:
[`run_180749_results.json`](run_180749_results.json). Claims C6–C10 in
[CLAIMS.md](CLAIMS.md).

| precision | engine | variant | loss read | tokens/s | range (3 processes) | vs best compiled PyTorch | vs best eager PyTorch | MFU |
|---|---|---|---|---:|---|---:|---:|---:|
| fp32 | LitniceLM | `preset:portable_fused` | every batch | **121,428** | 121,341–121,548 | **1.04×** | 1.15× | 28.3% |
| fp32 | PyTorch | compiled `max-autotune` | at end | 116,847 | 116,814–117,174 | 1.00× | 1.11× | 27.3% |
| fp32 | PyTorch | eager | at end | 105,740 | 105,727–105,938 | 0.90× | 1.00× | 24.7% |
| fp32 | PyTorch | eager | every batch | 99,787 | 99,227–99,835 | 0.85× | 0.94× | 23.3% |
| bf16 | LitniceLM | `preset:max_fused` | every batch | **335,711** | 335,186–336,239 | **1.02× (on par)** | **2.48×** | 39.2% |
| bf16 | PyTorch | compiled `reduce-overhead` | at end | 328,500 | 327,202–328,515 | 1.00× | 2.43× | 38.3% |
| bf16 | PyTorch | compiled `max-autotune` | at end | 327,673 | 327,256–336,171 | 1.00× | 2.42× | 38.2% |
| bf16 | PyTorch | eager | at end | 135,167 | 131,621–135,707 | 0.41× | 1.00× | 15.8% |
| bf16 | PyTorch | eager | every batch | 131,079 | 128,144–132,781 | 0.40× | 0.97× | 15.3% |

- **FP32:** LitniceLM is 4% ahead of PyTorch's best compiled mode; the ranges
  do not overlap.
- **BF16:** on par with compiled PyTorch. The median is 2% higher, but
  PyTorch's `max-autotune` processes reach LitniceLM's range, so no lead is
  claimed. Against eager PyTorch, LitniceLM is 2.48× faster.
- **Loss read.** LitniceLM reads every batch's loss on the host, which
  synchronizes the GPU once per batch. The best PyTorch rows, compiled and
  eager, read the loss only at the end of the run. Reading it every batch
  costs eager PyTorch 6% in FP32 and 3% in BF16 here. The comparison is
  therefore against PyTorch's faster setting.
- **Host state matters for eager PyTorch.** With identical software, PyTorch's
  BF16 eager throughput was ~108k tokens/s on one boot and ~133k on another,
  while FP32 and the CUDA-graph modes did not move. Every process here
  records AC power, platform profile and CPU policy.

### Fusion ladder

Each LitniceLM variant adds one fusion to the previous one.

| variant | attention core | projections and FFN | bf16 tokens/s | fp32 tokens/s |
|---|---|---|---:|---:|
| `unfused` | per-head GEMMs; separate scale, mask and softmax | cuBLAS GEMM; separate bias and activation | 184,100 | 106,132 |
| `fused_inplace` | per-head GEMMs; one fused scaled causal softmax | as above | 213,004 (+15.7%) | 116,933 (+10.2%) |
| `preset:portable_fused` | as above | cuBLASLt fused linear (bias and ReLU in the GEMM epilogue) | 226,233 (+6.2%) | 121,428 (+3.8%) |
| `preset:max_fused` | cuDNN flash attention (no score matrix) | as above | 335,711 (+48.4%) | — (BF16 only) |

`cudnn_flash` alone (with unfused projections and FFN) reaches 307,715 in
BF16: flash attention is the largest single step at this sequence length.

### Memory

Peak GPU memory per process (`nvidia-smi`, sampled every 100 ms):

| precision | LitniceLM | PyTorch best compiled | ratio |
|---|---:|---:|---:|
| bf16 | 292 MiB (`max_fused`) | 340 MiB | 0.86× |
| fp32 | 454 MiB (`portable_fused`) | 356 MiB | 1.28× |

- These figures include each process's CUDA context. The engines load
  different CUDA runtimes (LitniceLM 12.2, PyTorch 13.2), so part of each
  difference is context size.
- LitniceLM's own allocations are also larger. Its managed arenas total
  304 MiB in FP32 and 129 MiB in BF16, against a PyTorch allocator peak of
  59 MiB in its best compiled mode. The two figures are measured differently
  (reserved arenas vs `torch.cuda.max_memory_allocated`). The temp arena
  keeps every layer's work buffers and transpose buffers that are never used;
  reducing it is an open backlog item.

### How it was measured

- **Steady state, both engines.** PyTorch: wall time after 5 warm-up steps;
  compile time excluded. LitniceLM: the wall time of every batch after its
  first 5 optimizer steps. Its first step includes cuDNN graph build and plan
  selection (0.24 s per process for `max_fused`), reported separately, like
  PyTorch's compile time. LitniceLM's figure agrees within 0.3% with an
  independent epoch differential (1 vs 3 epochs, process wall time).
- **Same model, same math.** Both engines train the same model on the same
  data with the same optimizer. TF32 is off in both. BF16 means BF16
  activations with FP32 parameters, optimizer state and GEMM accumulation in
  both. Final training losses agree within 0.005 (LitniceLM 3.6207–3.6212,
  PyTorch 3.6171–3.6179).
- **Fixed conditions.** SM clock locked at 1500 MHz and verified under load
  for every process; host on AC with the `balanced` platform profile, checked
  before the run and recorded for every process.
- **MFU** = tokens/s × F / peak. F = 6N + 12·L·d·S training FLOPs per token
  (the PaLM paper's formula; N = 3,276,800 matmul weights, F = 32.2 M). Peak
  at 1500 MHz: 27.65 TFLOPS for BF16 tensor cores with FP32 accumulate,
  13.82 TFLOPS for FP32 without TF32. The PaLM formula counts attention
  without the causal mask; counting only the unmasked half lowers every MFU
  by about a fifth (LitniceLM 31.5% BF16, 22.8% FP32). Details in C9.

PyTorch implementation: `F.scaled_dot_product_attention(is_causal=True)`
(FlashAttention-2 in BF16, memory-efficient `fmha_cutlass` in FP32),
`torch.optim.AdamW(fused=True)`, `clip_grad_norm_`, `torch.autocast(bfloat16)`
with FP32 master weights for BF16, `torch.compile` in every mode.

### Reproduce

**Not yet reproducible from a fresh clone.** The run reads inputs that the
repository does not contain:

- the Conan Doyle corpus `c.doyle`, assembled from three books whose source
  editions and cleanup are not yet recorded (see the README's "Conan Doyle
  Corpus" section);
- the base config
  `conf/local/doyle/doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4.yaml`,
  which is untracked;
- the SentencePiece tokenizer artifacts and the tokenized dataset, at the
  absolute paths in `tools/vs_bench/run.py`;
- the parity export of the 12-epoch FP32 checkpoint (`export_dir` in
  `tools/vs_bench/run.py`);
- a Python environment with the PyTorch versions above; the dependencies in
  `tools/vs_bench/pyproject.toml` are not pinned.

On a machine that has these inputs, build LitniceLM with the CUDA plugin as
in [TRY_CUDA.md](TRY_CUDA.md); TRY_CUDA.md does not yet cover cuDNN, which
`max_fused` needs. The run then needs AC power, the `balanced` platform
profile and passwordless sudo for `/usr/bin/nvidia-smi` (clock lock); it
checks all three before measuring:

```bash
git checkout d01241e
cd tools
../.venv/bin/python -m vs_bench
```

It writes a results JSON (spec, environment, versions, every row), a
markdown report and every child log to a dated raw-benchmark directory.

## Tool status

`tools/vs_bench/` is the maintained comparison workflow. The earlier scripts
under `tools/litnicelm_vs_pytorch/` are retained temporarily as historical
verification material; they are not the documented reproduction path.
