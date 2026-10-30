# LitniceLM vs PyTorch

Two questions: does LitniceLM compute the same model as an external
reference, and how fast and memory-hungry is it next to PyTorch?

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop (8 GB), driver 580.95.05 |
| LitniceLM | commit 110327b, CUDA 12.2, cuBLAS 12.2.5.6, gcc 14.2, Linux 6.14 |
| PyTorch | 2.8.0+cu128 (CUDA 12.8, cuDNN 9.10.2), Python 3.13.3, eager mode |
| Model | 4 layers, 4 heads, d_model 256, d_ff 1024, RoPE (and learned positions for parity), vocabulary 512 (SentencePiece) |
| Data | Conan Doyle corpus, 485,847 tokens; validation: 1 of every 10 chunks of 8,192 tokens |
| Training | batch 1 × 1024 tokens, 4 batches per optimizer step, AdamW (β 0.9/0.95, ε 1e-8, weight decay 0.1 on embeddings and matrices), global-norm clip 0.06, LR 6e-4 cosine to 10% after 107 warmup steps |

Config: `conf/local/doyle/doyle_clean_spm512_seq1024_l4_d256_b1_rope_val_cosine_e12_accum4.yaml`
(the learned-position variant replaces `rope` with `learned`).

## Correctness: same weights, same data, same loss

`tools/litnicelm_vs_pytorch/reference_torch.py` loads a `litnicelm export`
(model.safetensors and
config.json) into a PyTorch model written from config.json alone, and
evaluates it over exactly the validation windows LitniceLM uses (37 batches,
37,888 tokens), averaging per-batch mean cross-entropy as LitniceLM does.

| checkpoint | evaluated as | PyTorch | LitniceLM | difference |
|---|---|---|---|---|
| RoPE, trained fp32 | fp32 | 2.4341600 | 2.4341600 | 1.4e-8 |
| learned, trained fp32 | fp32 | 3.5777680 | 3.5777678 | 1.6e-7 |
| RoPE, trained bf16_mixed | bf16 autocast | 2.4286395 | 2.4287624 | 1.2e-4 |
| learned, trained bf16_mixed | bf16 autocast | 3.5783493 | 3.5783910 | 4.2e-5 |

FP32 agrees to float32 rounding (tolerance used: 1e-5). The bf16 rows compare
two different BF16 implementations and are informational.

```
tools/litnicelm_vs_pytorch/reference_torch.py <export dir> --config <config.yaml> --tolerance 1e-5
```

## Speed and memory

`tools/litnicelm_vs_pytorch/bench_compare.py --config <config.yaml> --repeats 3` trains each
engine for 214 optimizer steps (876,544 tokens), interleaved, 3 repeats;
medians shown, tokens/s range in brackets.

| engine | precision | loss read | tokens/s | tokens/s per GHz | process peak (MiB) | engine memory (MiB) | SM clock (MHz) |
|---|---|---|---|---|---|---|---|
| LitniceLM | fp32 | per batch | 95,831 (95,475–116,981) | 58,971 | 538 | 360 | 1619 |
| LitniceLM | bf16_mixed | per batch | 180,332 (180,295–204,320) | 85,246 | 404 | 219 | 2115 |
| PyTorch | fp32 | per batch | 113,514 (112,295–120,142) | 57,469 | 330 | 140 | 1954 |
| PyTorch | fp32 | at end | 120,734 (120,623–120,807) | 65,337 | 332 | 140 | 1849 |
| PyTorch | bf16 | per batch | 113,919 (112,862–115,114) | 52,522 | 338 | 131 | 2169 |
| PyTorch | bf16 | at end | 114,794 (112,097–115,254) | 53,467 | 338 | 131 | 2140 |

- GPU clocks could not be locked (no root); the boost clock moved between
  1.6 and 2.4 GHz from run to run. "Tokens/s per GHz" divides by each run's
  mean SM clock; it is the steadier comparison, though memory-bound work does
  not scale with the clock. Raw data: `docs/comparison_results.json`.
- Throughput counts training batches only (no validation, no checkpoints).
  LitniceLM's comes from its profiler over all batches; PyTorch's from wall
  time after 5 warm-up steps.
- Process peak: `nvidia-smi` per-process memory (includes the CUDA context).
  Engine memory: LitniceLM's managed arenas; PyTorch's
  `torch.cuda.max_memory_allocated`.

PyTorch implementation: `F.scaled_dot_product_attention(is_causal=True)`
(memory-efficient attention `fmha_cutlass*_f32` in fp32, FlashAttention-2 in
bf16), `torch.optim.AdamW(fused=True)`, `clip_grad_norm_`, TF32 off,
`torch.autocast(bfloat16)` with FP32 master weights for bf16. LitniceLM:
cuBLAS SGEMM (fp32) or GemmEx BF16 with FP32 accumulation (bf16_mixed),
attention unfused (per-head GEMMs and softmax kernels).

## Where LitniceLM is behind

- **Memory**: 1.6× PyTorch at process level (538 vs 330 MiB in fp32). The
  temp arena keeps every layer's activations and work buffers, including
  transpose buffers that are never used.
- **fp32 speed**: on par with PyTorch reading the loss every batch, about 10%
  behind PyTorch reading it at the end. LitniceLM reads every batch's loss on
  the host, which synchronizes the GPU; in PyTorch that read costs about 12%
  in fp32 at this size.
- **Attention** is unfused and launched per head; PyTorch uses fused kernels.

In bf16, LitniceLM is ahead (about 1.6× per GHz): at d_model 256 PyTorch's
eager autocast pays per-op casting and launch overhead that outweighs its
tensor-core gains (its bf16 is slower than its fp32 here). This is one small
model; larger models will shift the balance toward GEMM throughput.
