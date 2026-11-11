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

Idea: give the logits their own item type in `PrecisionPolicy`, F32 in both
presets like `Loss`, and have the LM-head GEMM write F32 output (BF16 inputs,
F32 output via `cublasGemmEx`). Cost on that config: about 2 MiB more temp
memory. Needs a plan first: it changes the precision table and the LM-head
and cross-entropy operand roles.

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
