// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <cstdint>

#include <litnicelm_utils/tensor_view_struct.h>
#include "backend_timing.hpp"

// Tensor views cross the plugin boundary as TensorViewStruct pointers (see
// litnicelm_utils/tensor_view_struct.h); a host TensorView is passed as is.
struct BackendMemoryInfo {
  uint32_t available;
  uint64_t free_bytes;
  uint64_t total_bytes;
};

// Kernel variants a backend implements, as bits of
// BackendApiV1::precision_variants.
inline constexpr uint32_t kPrecisionVariantF32 = 1u << 0;  // <op>_f32: every tensor FP32
// <op>_bf16act: activations and activation gradients BF16; parameters, their
// gradients and optimizer state FP32.
inline constexpr uint32_t kPrecisionVariantBf16Act = 1u << 1;

// Optional fused ops a backend implements, as bits of
// BackendApiV1::fused_ops. Their entries may be null when the bit is clear.
// Causal attention over heads side by side, BF16 activations (_bf16act
// only): causal_attention_forward and causal_attention_backward.
inline constexpr uint32_t kFusedOpCausalAttentionBf16Act = 1u << 0;
// Linear layers with the bias (and ReLU) and the bias gradient fused into
// the GEMMs, both precision variants: linear_forward and
// linear_backward_params. Without it, Ops composes them from primitives.
inline constexpr uint32_t kFusedOpLinear = 1u << 1;

struct BackendApiV1 {
  uint32_t abi_version;
  void *(*create)(uint32_t device);
  void (*destroy)(void *backend);
  uint32_t (*device)(void *backend);
  void *(*alloc)(void *backend, uint64_t bytes, uint32_t alignment);
  void (*free)(void *backend, void *ptr);
  void (*copy_host2device)(void *backend, void *dst, const void *src,
                           uint64_t bytes);
  void (*copy_device2host)(void *backend, void *dst, const void *src,
                           uint64_t bytes);
  void (*copy)(void *backend, const TensorViewStruct *src,
               const TensorViewStruct *dst);
  void (*fill)(void *backend, const TensorViewStruct *t, float v);
  void (*add)(void *backend, const TensorViewStruct *a,
              const TensorViewStruct *b, const TensorViewStruct *out);
  void (*add_inplace)(void *backend, const TensorViewStruct *a,
                      const TensorViewStruct *b);
  void (*add_bias_rowwise)(void *backend, const TensorViewStruct *x,
                           const TensorViewStruct *bias_1xC,
                           const TensorViewStruct *out);
  void (*add_bias_relu_rowwise)(void *backend, const TensorViewStruct *x,
                                const TensorViewStruct *bias_1xC,
                                const TensorViewStruct *out);
  void (*add_bias_relu_rowwise_inplace)(void *backend,
                                        const TensorViewStruct *x,
                                        const TensorViewStruct *bias_1xC);
  void (*mul_scalar)(void *backend, const TensorViewStruct *x, float s,
                     const TensorViewStruct *out);
  float (*sum_squares_f32)(void *backend, const TensorViewStruct *x);
  void (*relu)(void *backend, const TensorViewStruct *x,
               const TensorViewStruct *out);
  void (*relu_backward)(void *backend, const TensorViewStruct *preact,
                        const TensorViewStruct *dout,
                        const TensorViewStruct *dx);
  void (*relu_backward_inplace)(void *backend, const TensorViewStruct *preact,
                                const TensorViewStruct *dout_dx);
  void (*row_sum)(void *backend, const TensorViewStruct *x,
                  const TensorViewStruct *out_1xC);
  void (*gemm)(void *backend, const TensorViewStruct *a,
               const TensorViewStruct *b, const TensorViewStruct *out);
  void (*gemm_ranked_matrix_rhs)(void *backend, const TensorViewStruct *a,
                                 const TensorViewStruct *b,
                                 const TensorViewStruct *out);
  void (*gemm_ranked_matrix_rhs_t)(void *backend, const TensorViewStruct *a,
                                   const TensorViewStruct *b,
                                   const TensorViewStruct *out);
  void (*gemm_ranked_reduce_lhs_t)(void *backend, const TensorViewStruct *a,
                                   const TensorViewStruct *b,
                                   const TensorViewStruct *out);
  void (*gemm_batched)(void *backend, const TensorViewStruct *a,
                       const TensorViewStruct *b,
                       const TensorViewStruct *out);
  void (*gemm_batched_lhs_t)(void *backend, const TensorViewStruct *a,
                             const TensorViewStruct *b,
                             const TensorViewStruct *out);
  void (*gemm_batched_rhs_t)(void *backend, const TensorViewStruct *a,
                             const TensorViewStruct *b,
                             const TensorViewStruct *out);
  void (*transpose)(void *backend, const TensorViewStruct *x,
                    const TensorViewStruct *out);
  void (*layernorm_forward)(void *backend, const TensorViewStruct *x,
                            const TensorViewStruct *gamma_1xC,
                            const TensorViewStruct *beta_1xC,
                            const TensorViewStruct *out);
  void (*layernorm_backward)(void *backend, const TensorViewStruct *x,
                             const TensorViewStruct *gamma_1xC,
                             const TensorViewStruct *dout,
                             const TensorViewStruct *dx,
                             const TensorViewStruct *dgamma_1xC,
                             const TensorViewStruct *dbeta_1xC);
  void (*embedding_lookup)(void *backend, const TensorViewStruct *table,
                           const TensorViewStruct *ids,
                           const TensorViewStruct *out);
  void (*accumulate_embedding_grads)(void *backend,
                                     const TensorViewStruct *ids,
                                     const TensorViewStruct *d_cur,
                                     const TensorViewStruct *d_tok);
  void (*cross_entropy_mean)(void *backend, const TensorViewStruct *logits,
                             const TensorViewStruct *targets,
                             const TensorViewStruct *out_loss);
  void (*cross_entropy_mean_backward_inplace)(
      void *backend, const TensorViewStruct *logits,
      const TensorViewStruct *targets, const TensorViewStruct *out_loss);
  float (*read_scalar_f32)(void *backend, const TensorViewStruct *x);
  void (*backward_from_logits_targets)(void *backend,
                                       const TensorViewStruct *logits,
                                       const TensorViewStruct *targets);
  void (*softmax_rows)(void *backend, const TensorViewStruct *x,
                       const TensorViewStruct *out);
  void (*softmax_backward_rows)(void *backend,
                                const TensorViewStruct *softmax,
                                const TensorViewStruct *dout,
                                const TensorViewStruct *dx);
  void (*scaled_causal_softmax_rows)(void *backend,
                                     const TensorViewStruct *scores,
                                     float scale,
                                     const TensorViewStruct *out);
  uint32_t (*supports_exec_context_iteration)(void *backend);
  void (*start_exec_context_iteration)(void *backend);
  void (*finish_exec_context_iteration)(void *backend);
  void (*start_exec_context_group)(void *backend);
  void (*finish_exec_context_group)(void *backend);
  void (*gemm_batched_exec_context)(void *backend, const TensorViewStruct *a,
                                    const TensorViewStruct *b,
                                    const TensorViewStruct *out);
  void (*gemm_batched_lhs_t_exec_context)(
      void *backend, const TensorViewStruct *a, const TensorViewStruct *b,
      const TensorViewStruct *out);
  void (*gemm_batched_rhs_t_exec_context)(
      void *backend, const TensorViewStruct *a, const TensorViewStruct *b,
      const TensorViewStruct *out);
  void (*scaled_causal_softmax_rows_exec_context)(
      void *backend, const TensorViewStruct *scores, float scale,
      const TensorViewStruct *out);
  void (*softmax_backward_causal_rows_exec_context)(
      void *backend, const TensorViewStruct *softmax,
      const TensorViewStruct *dout, const TensorViewStruct *dx);
  void (*softmax_backward_causal_rows)(void *backend,
                                       const TensorViewStruct *softmax,
                                       const TensorViewStruct *dout,
                                       const TensorViewStruct *dx);
  void (*apply_causal_mask_inplace)(void *backend,
                                    const TensorViewStruct *scores,
                                    float neg_inf);
  // Rotates channel pairs (i, i + head_dim/2) of every head of x [..., S, D]
  // by angle pos * base^(-2i/head_dim); inverse != 0 rotates by the negative
  // angle (the transpose, used for gradients).
  void (*rotary_embedding_inplace)(void *backend, const TensorViewStruct *x,
                                   int64_t n_heads, float base,
                                   uint32_t inverse);
  void (*adamw_step)(void *backend, const TensorViewStruct *params,
                     const TensorViewStruct *grads,
                     const TensorViewStruct *m,
                     const TensorViewStruct *v, uint64_t step,
                     float learning_rate, float beta1, float beta2,
                     float weight_decay, uint32_t apply_weight_decay);
  uint32_t (*is_file2device_read_supported)(void *backend);
  void (*read_file2device)(void *backend, const char *path, void *dst,
                           uint64_t size, uint64_t file_offset);
  BackendMemoryInfo (*memory_info)(void *backend);
  uint32_t (*precision_variants)(void *backend);
  // F32 -> BF16 (round to nearest even, NaN and Inf kept) or BF16 -> F32.
  void (*cast)(void *backend, const TensorViewStruct *src,
               const TensorViewStruct *dst);
  void (*timing_mark_begin)(void *backend, uint32_t op_id);
  void (*timing_mark_end)(void *backend);
  BackendTimingReport (*timing_report)(void *backend);
  void (*timing_prepare)(void *backend, uint64_t max_marks);
  // Optional fused ops: the kFusedOp* bits this backend implements. Null
  // (as for a table without these entries) means none.
  uint32_t (*fused_ops)(void *backend);
  // qkv [B, S, 3D] (Q | K | V, heads side by side; rows may be strided) ->
  // out [B, S, D] = softmax(scale * Q K^T + causal mask) V per head. stats
  // [B, H, S] F32 receives each row's softmax log-sum-exp for backward; a
  // null stats->ptr skips it (inference).
  void (*causal_attention_forward)(void *backend, const TensorViewStruct *qkv,
                                   int64_t n_heads, float scale,
                                   const TensorViewStruct *out,
                                   const TensorViewStruct *stats);
  // dout [B, S, D] -> dqkv [B, S, 3D], given the forward's qkv, out and stats.
  void (*causal_attention_backward)(void *backend, const TensorViewStruct *qkv,
                                    const TensorViewStruct *out,
                                    const TensorViewStruct *dout,
                                    const TensorViewStruct *stats,
                                    int64_t n_heads, float scale,
                                    const TensorViewStruct *dqkv);
  // x [..., in] (contiguous rows), w [in, out] (the compute weight), bias
  // [1, out] F32 -> out [..., out] = x w + bias, then ReLU when relu != 0;
  // the bias is added before the single rounding of out.
  void (*linear_forward)(void *backend, const TensorViewStruct *x,
                         const TensorViewStruct *w,
                         const TensorViewStruct *bias, uint32_t relu,
                         const TensorViewStruct *out);
  // dw [in, out] = x^T dy and dbias [1, out] = column sums of dy over all
  // rows, both F32 and overwritten.
  void (*linear_backward_params)(void *backend, const TensorViewStruct *x,
                                 const TensorViewStruct *dy,
                                 const TensorViewStruct *dw,
                                 const TensorViewStruct *dbias);
};

inline constexpr uint32_t kBackendApiVersion = 24;

extern "C" {
typedef const BackendApiV1 *(*BackendGetApiFn)();
}
