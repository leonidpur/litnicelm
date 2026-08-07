#pragma once

#include "backend/device_backend.hpp"
#include "tensor.hpp"

class Ops {
public:
  explicit Ops(DeviceBackend &device_backend);

  void copy(const TensorView &src, TensorView &dst) const;
  void fill(TensorView &t, float v) const;

  void add(const TensorView &a, const TensorView &b, TensorView &out) const;
  void add_inplace(TensorView &a, const TensorView &b) const;
  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) const;
  void add_bias_relu_rowwise(const TensorView &x,
                             const TensorView &bias_1xC,
                             TensorView &out) const;
  void add_bias_relu_rowwise_inplace(TensorView &x,
                                     const TensorView &bias_1xC) const;

  void mul_scalar(const TensorView &x, float s, TensorView &out) const;
  float sum_squares_f32(const TensorView &x) const;
  void relu(const TensorView &x, TensorView &out) const;
  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) const;
  void relu_backward_inplace(const TensorView &preact,
                             TensorView &dout_dx) const;
  void row_sum(const TensorView &x, TensorView &out_1xC) const;

  void gemm(const TensorView &a, const TensorView &b, TensorView &out) const;
  void gemm_ranked_matrix_rhs(const TensorView &a, const TensorView &b,
                              TensorView &out) const;
  void gemm_ranked_matrix_rhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) const;
  void gemm_ranked_reduce_lhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) const;
  void gemm_batched(const TensorView &a, const TensorView &b,
                    TensorView &out) const;
  void gemm_batched_lhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) const;
  void gemm_batched_rhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) const;
  void transpose(const TensorView &x, TensorView &out) const;
  void layernorm(const TensorView &x, const TensorView &gamma_1xC,
                 const TensorView &beta_1xC, TensorView &out) const;
  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) const;
  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) const;
  void accumulate_embedding_grads(const TensorView &ids,
                                  const TensorView &d_cur, TensorView &d_tok,
                                  TensorView &d_pos) const;
  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) const;
  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) const;
  float read_scalar_f32(const TensorView &x) const;
  bool supports_backward() const;
  void backward_from_logits_targets(const TensorView &logits,
                                    const TensorView &targets) const;

  void softmax_rows(const TensorView &x, TensorView &out) const;
  void softmax_backward_rows(const TensorView &softmax, const TensorView &dout,
                             TensorView &dx) const;
  void scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                  TensorView &out) const;
  bool supports_exec_context_iteration() const;
  void start_exec_context_iteration() const;
  void finish_exec_context_iteration() const;
  void start_exec_context_group() const;
  void finish_exec_context_group() const;
  void gemm_batched_exec_context(const TensorView &a, const TensorView &b,
                                 TensorView &out) const;
  void gemm_batched_lhs_t_exec_context(const TensorView &a,
                                       const TensorView &b,
                                       TensorView &out) const;
  void gemm_batched_rhs_t_exec_context(const TensorView &a,
                                       const TensorView &b,
                                       TensorView &out) const;
  void scaled_causal_softmax_rows_exec_context(const TensorView &scores,
                                               float scale,
                                               TensorView &out) const;
  void softmax_backward_causal_rows_exec_context(const TensorView &softmax,
                                                const TensorView &dout,
                                                TensorView &dx) const;
  void softmax_backward_causal_rows(const TensorView &softmax,
                                    const TensorView &dout,
                                    TensorView &dx) const;
  void apply_causal_mask_inplace(TensorView &scores, float neg_inf = -1e9f) const;
  DeviceBackend &backend() const { return device_backend_; }

private:
  DeviceBackend &device_backend_;
};
