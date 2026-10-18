#pragma once

#include "cuda_tensor_view.hpp"

#include <cuda_runtime.h>

namespace cuda_cublas_plugin {

// F32 -> BF16 rounds to nearest even and keeps NaN (with its sign) and Inf,
// bit-for-bit like the host's f32_to_bf16_bits; BF16 -> F32 is exact.
void launch_cast_f32_to_bf16(const TensorView &src, TensorView &dst);
void launch_cast_bf16_to_f32(const TensorView &src, TensorView &dst);

// Elementwise ops: <op>_f32 for all-F32 operands, <op>_bf16act for BF16
// activations with F32 parameters (bias, position table) and parameter
// gradients (add_inplace's accumulator).

void launch_fill_f32(TensorView &t, float value);
void launch_fill_bf16act(TensorView &t, float value);
void launch_add_f32(const TensorView &a, const TensorView &b, TensorView &out);
void launch_add_bf16act(const TensorView &a, const TensorView &b,
                        TensorView &out);
void launch_add_batch_seq_plus_pos_f32(const TensorView &a, const TensorView &b,
                                       TensorView &out, int64_t seq_len);
void launch_add_batch_seq_plus_pos_bf16act(const TensorView &a,
                                           const TensorView &b, TensorView &out,
                                           int64_t seq_len);
void launch_add_inplace_f32(TensorView &a, const TensorView &b);
void launch_add_inplace_bf16act(TensorView &a, const TensorView &b);
void launch_add_bias_rowwise_f32(const TensorView &x, const TensorView &bias,
                                 TensorView &out);
void launch_add_bias_rowwise_bf16act(const TensorView &x,
                                     const TensorView &bias, TensorView &out);
void launch_add_bias_relu_rowwise_f32(const TensorView &x,
                                      const TensorView &bias, TensorView &out);
void launch_add_bias_relu_rowwise_bf16act(const TensorView &x,
                                          const TensorView &bias,
                                          TensorView &out);
void launch_add_bias_relu_rowwise_inplace_f32(TensorView &x,
                                              const TensorView &bias);
void launch_add_bias_relu_rowwise_inplace_bf16act(TensorView &x,
                                                  const TensorView &bias);
void launch_mul_scalar_f32(const TensorView &x, float scale, TensorView &out);
void launch_mul_scalar_bf16act(const TensorView &x, float scale,
                               TensorView &out);
void launch_relu_f32(const TensorView &x, TensorView &out);
void launch_relu_bf16act(const TensorView &x, TensorView &out);
void launch_relu_backward_f32(const TensorView &preact, const TensorView &dout,
                              TensorView &dx);
void launch_relu_backward_bf16act(const TensorView &preact,
                                  const TensorView &dout, TensorView &dx);
void launch_relu_backward_inplace_f32(const TensorView &preact,
                                      TensorView &dout_dx);
void launch_relu_backward_inplace_bf16act(const TensorView &preact,
                                          TensorView &dout_dx);
void launch_transpose(const TensorView &x, TensorView &out);
void launch_apply_causal_mask_inplace_f32(TensorView &scores, float neg_inf);
void launch_apply_causal_mask_inplace_bf16act(TensorView &scores,
                                              float neg_inf);
void launch_rotary_embedding_inplace_f32(TensorView &x, int64_t n_heads,
                                         float base, bool inverse);
void launch_rotary_embedding_inplace_bf16act(TensorView &x, int64_t n_heads,
                                             float base, bool inverse);

void launch_sum_squares_f32(const TensorView &x, float *device_sum_sq);
void launch_row_sum_f32(const TensorView &x, TensorView &out_1xC);
// x BF16 (an activation gradient), out_1xC F32.
void launch_row_sum_bf16act(const TensorView &x, TensorView &out_1xC);

void launch_softmax_rows(const TensorView &x, TensorView &out);
void launch_softmax_backward_rows(const TensorView &softmax,
                                  const TensorView &dout, TensorView &dx);
void launch_scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                       TensorView &out);
void launch_scaled_causal_softmax_rows_on_stream(const TensorView &scores,
                                                 float scale, TensorView &out,
                                                 cudaStream_t stream);
void launch_softmax_backward_causal_rows(const TensorView &softmax,
                                         const TensorView &dout,
                                         TensorView &dx);
void launch_softmax_backward_causal_rows_on_stream(const TensorView &softmax,
                                                   const TensorView &dout,
                                                   TensorView &dx,
                                                   cudaStream_t stream);

void launch_cross_entropy_mean_backward_inplace(TensorView &logits,
                                                const TensorView &targets,
                                                TensorView &out_loss);

void launch_layernorm_forward(const TensorView &x, const TensorView &gamma,
                              const TensorView &beta, TensorView &out);
void launch_layernorm_backward(const TensorView &x, const TensorView &gamma,
                               const TensorView &dout, TensorView &dx,
                               TensorView &dgamma, TensorView &dbeta);

void launch_embedding_lookup(const TensorView &table, const TensorView &ids,
                             TensorView &out);
void launch_accumulate_embedding_grads(const TensorView &ids,
                                       const TensorView &d_cur,
                                       TensorView &d_tok);

void launch_adamw_step(TensorView &params, const TensorView &grads,
                       TensorView &m, TensorView &v, float learning_rate,
                       float beta1, float beta2, float weight_decay,
                       float inv_b1_corr, float inv_b2_corr,
                       uint32_t apply_weight_decay);

} // namespace cuda_cublas_plugin
