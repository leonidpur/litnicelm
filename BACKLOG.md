# Backlog

## bf16_mixed

### F32 logits for decoding

Under `precision: bf16_mixed` the logits are an activation, so they are stored
in BF16 (8-bit mantissa) and nearby logits round to the same value. With
`inference.report_margin: true`, two bf16_mixed runs of the Doyle spm512
config (4 layers, d256, 4 batches per step) showed exact top-1/top-2 ties
(margin 0.000000 at steps 36 and 23 of 900); fp32 runs of the same config
showed only near-ties (0.0006, 0.00008). On a tie, greedy decoding picks the
lower token id. Training is unaffected: cross-entropy computes in FP32.

Done for inference (Wave A step 1, 2026-11-26): `Fp32LogitsOutputHead` casts
the final hidden states to FP32 and multiplies by the FP32 `lm_head_w`. Still
open: training and validation logits (deliberately not changed).

## Found during the PyTorch comparison (docs/COMPARISON.md)

- **Per-batch loss read.** LitniceLM reads every batch's loss on the host,
  which synchronizes the GPU. In PyTorch the same read costs about 12% of
  fp32 throughput at d_model 256. Read losses at the end of a step, or
  accumulate them on the device.
- **AdamW epsilon.** The AdamW kernels use a fixed `kEps = 1e-8` and ignore
  `training.eps`.
- **Memory.** 1.6× PyTorch at process level; the temp arena holds unused
  transpose buffers (`attn.xT`, `ffn.aT`, ...) and all layers' work buffers.

## Phase 4 follow-up: BF16 forward epilogue (2026-11-09)

- cuBLASLt (CUDA 12.2) has no algorithm for BF16 D with an F32 bias
  (`CUBLASLT_EPILOGUE_BIAS` / `RELU_BIAS`, status NOT_SUPPORTED), so
  `linear_forward` in `_bf16act` falls back to GEMM + bias kernel. Options:
  a BF16 compute copy of the biases (layout change, bias rounded once to
  BF16), or a newer cuBLASLt. Worth ~1.9 ms per 3 optimizer steps of the
  bench profile (add_bias_rowwise + add_bias_relu_rowwise_inplace).

## Row-warp LayerNorm follow-ups (2026-11-17)

- **`layernorm_param_grads_*`.** Unchanged at about 5.2 µs per call (bf16
  max_fused, 1024 rows x 256), now the largest LayerNorm kernel: it runs on
  `cols / 32` blocks (8 at d_model 256). Split the rows over more blocks with
  a fixed-order second pass to keep it bitwise reproducible.
- **Backward register pressure.** `layernorm_backward_rowwarp_*` hold 2 x 32
  floats per lane for widths up to 1024 and sit at the 128-register cap of
  their launch bounds, with a 32-byte stack frame on some architectures. A
  narrower row-warp kernel for widths up to 512 would halve that.
- **Vector loads.** The row-warp kernels use coalesced scalar loads, so the
  path needs only `cols <= 1024`. `float2` / `__nv_bfloat162` loads would
  add width-parity and alignment conditions; measure before adding them.

## Benchmark timing follow-ups (2026-11-22)

- **Full-run report mislabels LitniceLM's speed as steady state.** The
  `train_speed` vertical's "steady-state tokens/s" for LitniceLM is the
  `batch_step` total of one `train` run, which includes the first batch's
  one-time setup (cuDNN graph build and plan timing, ~300 ms in bf16
  `max_fused`), validation and checkpoints. PyTorch's figure excludes warm-up.
  The gate's PyTorch ratio now uses the steady-state differential; the full
  report's ratio and conclusions do not yet.
- **Re-measure PyTorch.** The gate's reference is `run_003316` (2026-11-03).
- **LitniceLM first-call cost.** About 300 ms per process in bf16 `max_fused`,
  mostly cuDNN graph building; plan timing adds ~112 ms of it.
- **`train_infer_e2e` is intermittent under `ctest -j4`.** It failed in the
  parallel run on 2026-11-18 and 2026-11-23 and passed when rerun alone both
  times; not yet investigated (likely a shared path or GPU contention).
- **Benchmark not reproducible from a fresh clone.** `vs_bench` needs the
  Doyle corpus (sources and cleanup unrecorded), the untracked base config
  under `conf/local/doyle/`, tokenizer artifacts and tokenized `.bin` at
  absolute SSD paths in `run.py`, the 12-epoch FP32 parity export, and an
  unpinned Python environment. TRY_CUDA.md does not cover cuDNN (needed by
  `max_fused`). docs/COMPARISON.md "Reproduce" states this.
- **Inference re-selects cuDNN plans for every new length.** Done (Wave A
  step 1, 2026-11-26): inference builds only the heuristic's first plan,
  untimed, and decodes through a KV cache. Still built once per new prompt
  length: one SDPA graph (prefill), and past the context every step
  re-prefills (no rolling cache yet).
- **CPU `copy` handles strided views only at rank 2.** `DefaultCpuBackend::copy`
  falls back to a `dims[0] x dims[1]` loop for non-contiguous views, so a
  strided rank-3 view (e.g. `[1, S, D]` columns of qkv) is copied wrongly with
  no error; CUDA copies such views correctly. `KvCacheAttention` passes rank-2
  views to stay clear of it (found 2026-11-26).
- **`reference_torch.py` can't read the val loss from a hand-written model card.**
  It looks for the exporter's `| best val loss | …` row; the TinyStories card
  (2026-11-25) labels it differently, so the comparison needs `--litnicelm-loss`.
