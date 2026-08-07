#pragma once

#include "memory/memory_resource_info.hpp"

#include <config.hpp>
#include "tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>

class DeviceBackend {
public:
  static std::unique_ptr<DeviceBackend> create_instance(const Config &cfg);

  virtual ~DeviceBackend() = default;
  virtual Device device() const = 0;
  virtual void *alloc(uint64_t bytes, uint32_t alignment) = 0;
  virtual void free(void *ptr) = 0;
  virtual DeviceMemoryInfo memory_info() const = 0;
  virtual void copy_host2device(void *dst, const void *src, uint64_t bytes) = 0;
  virtual void copy_device2host(void *dst, const void *src, uint64_t bytes) = 0;
  virtual void copy(const TensorView &src, TensorView &dst) = 0;
  virtual void fill(TensorView &t, float v) = 0;
  virtual void add(const TensorView &a, const TensorView &b, TensorView &out) = 0;
  virtual void add_inplace(TensorView &a, const TensorView &b) = 0;
  virtual void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                                TensorView &out) = 0;
  virtual void add_bias_relu_rowwise(const TensorView &x,
                                     const TensorView &bias_1xC,
                                     TensorView &out) = 0;
  virtual void add_bias_relu_rowwise_inplace(TensorView &x,
                                             const TensorView &bias_1xC) = 0;
  virtual void mul_scalar(const TensorView &x, float s, TensorView &out) = 0;
  virtual float sum_squares_f32(const TensorView &x) = 0;
  virtual void relu(const TensorView &x, TensorView &out) = 0;
  virtual void relu_backward(const TensorView &preact, const TensorView &dout,
                             TensorView &dx) = 0;
  virtual void relu_backward_inplace(const TensorView &preact,
                                     TensorView &dout_dx) = 0;
  virtual void row_sum(const TensorView &x, TensorView &out_1xC) = 0;
  virtual void matmul(const TensorView &a, const TensorView &b, TensorView &out) = 0;
  virtual void matmul_left_transposed(const TensorView &a, const TensorView &b,
                                      TensorView &out) = 0;
  virtual void matmul_right_transposed(const TensorView &a, const TensorView &b,
                                       TensorView &out) = 0;
  virtual void transpose(const TensorView &x, TensorView &out) = 0;
  virtual void layernorm_forward(const TensorView &x,
                                 const TensorView &gamma_1xC,
                                 const TensorView &beta_1xC,
                                 TensorView &out) = 0;
  virtual void layernorm_backward(const TensorView &x,
                                  const TensorView &gamma_1xC,
                                  const TensorView &dout, TensorView &dx,
                                  TensorView &dgamma_1xC,
                                  TensorView &dbeta_1xC) = 0;
  virtual void embedding_lookup(const TensorView &table, const TensorView &ids,
                                TensorView &out) = 0;
  virtual void accumulate_embedding_grads(const TensorView &ids,
                                          const TensorView &d_cur,
                                          TensorView &d_tok,
                                          TensorView &d_pos) = 0;
  virtual void cross_entropy_mean(const TensorView &logits,
                                  const TensorView &targets,
                                  TensorView &out_loss) = 0;
  virtual void cross_entropy_mean_backward_inplace(TensorView &logits,
                                                   const TensorView &targets,
                                                   TensorView &out_loss) = 0;
  virtual float read_scalar_f32(const TensorView &x) = 0;
  virtual void backward_from_logits_targets(TensorView &logits,
                                            const TensorView &targets) = 0;
  virtual void softmax_rows(const TensorView &x, TensorView &out) = 0;
  virtual void softmax_backward_rows(const TensorView &softmax,
                                     const TensorView &dout,
                                     TensorView &dx) = 0;
  virtual void scaled_causal_softmax_rows(const TensorView &scores,
                                          float scale, TensorView &out) = 0;
  virtual bool supports_exec_context_iteration() const = 0;
  virtual void start_exec_context_iteration() = 0;
  virtual void finish_exec_context_iteration() = 0;
  virtual void start_exec_context_group() = 0;
  virtual void finish_exec_context_group() = 0;
  virtual void matmul_exec_context(const TensorView &a, const TensorView &b,
                                   TensorView &out) = 0;
  virtual void matmul_left_transposed_exec_context(const TensorView &a,
                                                  const TensorView &b,
                                                  TensorView &out) = 0;
  virtual void matmul_right_transposed_exec_context(const TensorView &a,
                                                   const TensorView &b,
                                                   TensorView &out) = 0;
  virtual void scaled_causal_softmax_rows_exec_context(
      const TensorView &scores, float scale, TensorView &out) = 0;
  virtual void softmax_backward_causal_rows_exec_context(
      const TensorView &softmax, const TensorView &dout, TensorView &dx) = 0;
  virtual void softmax_backward_causal_rows(const TensorView &softmax,
                                            const TensorView &dout,
                                            TensorView &dx) = 0;
  virtual void apply_causal_mask_inplace(TensorView &scores,
                                         float neg_inf = -1e9f) = 0;
  virtual void adamw_step(TensorView &params, const TensorView &grads,
                          TensorView &m, TensorView &v, uint64_t step,
                          float learning_rate, float beta1, float beta2,
                          float weight_decay, bool apply_weight_decay) = 0;
  virtual bool is_file2device_read_supported() const = 0;
  virtual void read_file2device(const std::string &path, void *dst,
                                uint64_t size, uint64_t file_offset) = 0;
};

class DefaultCpuBackend final : public DeviceBackend {
public:
  DefaultCpuBackend() = default;
  Device device() const override;
  void *alloc(uint64_t bytes, uint32_t alignment) override;
  void free(void *ptr) override;
  DeviceMemoryInfo memory_info() const override;
  void copy_host2device(void *dst, const void *src, uint64_t bytes) override;
  void copy_device2host(void *dst, const void *src, uint64_t bytes) override;
  void copy(const TensorView &src, TensorView &dst) override;
  void fill(TensorView &t, float v) override;
  void add(const TensorView &a, const TensorView &b, TensorView &out) override;
  void add_inplace(TensorView &a, const TensorView &b) override;
  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override;
  void add_bias_relu_rowwise(const TensorView &x,
                             const TensorView &bias_1xC,
                             TensorView &out) override;
  void add_bias_relu_rowwise_inplace(TensorView &x,
                                     const TensorView &bias_1xC) override;
  void mul_scalar(const TensorView &x, float s, TensorView &out) override;
  float sum_squares_f32(const TensorView &x) override;
  void relu(const TensorView &x, TensorView &out) override;
  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) override;
  void relu_backward_inplace(const TensorView &preact,
                             TensorView &dout_dx) override;
  void row_sum(const TensorView &x, TensorView &out_1xC) override;
  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override;
  void matmul_left_transposed(const TensorView &a, const TensorView &b,
                              TensorView &out) override;
  void matmul_right_transposed(const TensorView &a, const TensorView &b,
                               TensorView &out) override;
  void transpose(const TensorView &x, TensorView &out) override;
  void layernorm_forward(const TensorView &x, const TensorView &gamma_1xC,
                         const TensorView &beta_1xC, TensorView &out) override;
  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) override;
  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) override;
  void accumulate_embedding_grads(const TensorView &ids, const TensorView &d_cur,
                                  TensorView &d_tok,
                                  TensorView &d_pos) override;
  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override;
  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) override;
  float read_scalar_f32(const TensorView &x) override;
  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override;
  void softmax_rows(const TensorView &x, TensorView &out) override;
  void softmax_backward_rows(const TensorView &softmax, const TensorView &dout,
                             TensorView &dx) override;
  void scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                  TensorView &out) override;
  bool supports_exec_context_iteration() const override;
  void start_exec_context_iteration() override;
  void finish_exec_context_iteration() override;
  void start_exec_context_group() override;
  void finish_exec_context_group() override;
  void matmul_exec_context(const TensorView &a, const TensorView &b,
                           TensorView &out) override;
  void matmul_left_transposed_exec_context(const TensorView &a,
                                          const TensorView &b,
                                          TensorView &out) override;
  void matmul_right_transposed_exec_context(const TensorView &a,
                                           const TensorView &b,
                                           TensorView &out) override;
  void scaled_causal_softmax_rows_exec_context(
      const TensorView &scores, float scale, TensorView &out) override;
  void softmax_backward_causal_rows_exec_context(
      const TensorView &softmax, const TensorView &dout,
      TensorView &dx) override;
  void softmax_backward_causal_rows(const TensorView &softmax,
                                    const TensorView &dout,
                                    TensorView &dx) override;
  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override;
  void adamw_step(TensorView &params, const TensorView &grads, TensorView &m,
                  TensorView &v, uint64_t step, float learning_rate,
                  float beta1, float beta2, float weight_decay,
                  bool apply_weight_decay) override;
  bool is_file2device_read_supported() const override;
  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override;
};
