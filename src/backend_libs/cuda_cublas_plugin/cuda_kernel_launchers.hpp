#pragma once

#include "cuda_tensor_view.hpp"

#include <cuda_runtime.h>

namespace cuda_cublas_plugin {

void launch_fill(TensorView &t, float value);
void launch_add(const TensorView &a, const TensorView &b, TensorView &out);
void launch_add_batch_seq_plus_pos(const TensorView &a, const TensorView &b,
                                   TensorView &out, int64_t seq_len);
void launch_add_inplace(TensorView &a, const TensorView &b);
void launch_add_bias_rowwise(const TensorView &x, const TensorView &bias,
                             TensorView &out);
void launch_add_bias_relu_rowwise(const TensorView &x, const TensorView &bias,
                                  TensorView &out);
void launch_add_bias_relu_rowwise_inplace(TensorView &x,
                                          const TensorView &bias);
void launch_mul_scalar(const TensorView &x, float scale, TensorView &out);
void launch_relu(const TensorView &x, TensorView &out);
void launch_relu_backward(const TensorView &preact, const TensorView &dout,
                          TensorView &dx);
void launch_relu_backward_inplace(const TensorView &preact, TensorView &dout_dx);
void launch_transpose(const TensorView &x, TensorView &out);
void launch_apply_causal_mask_inplace(TensorView &scores, float neg_inf);

void launch_sum_squares_f32(const TensorView &x, float *device_sum_sq);
void launch_row_sum(const TensorView &x, TensorView &out_1xC);

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
                                       TensorView &d_tok, TensorView &d_pos,
                                       int64_t seq_len);

void launch_adamw_step(TensorView &params, const TensorView &grads,
                       TensorView &m, TensorView &v, float learning_rate,
                       float beta1, float beta2, float weight_decay,
                       float inv_b1_corr, float inv_b2_corr,
                       uint32_t apply_weight_decay);

} // namespace cuda_cublas_plugin
