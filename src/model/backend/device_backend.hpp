#pragma once

#include <config.hpp>
#include "tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>

class DeviceBackend {
public:
  virtual ~DeviceBackend() = default;
  virtual void *alloc(uint64_t bytes, uint32_t alignment) = 0;
  virtual void free(void *ptr) = 0;
  virtual void copy_host2device(void *dst, const void *src, uint64_t bytes) = 0;
  virtual void copy_device2host(void *dst, const void *src, uint64_t bytes) = 0;
  virtual void copy(const TensorView &src, TensorView &dst) = 0;
  virtual void fill(TensorView &t, float v) = 0;
  virtual void add(const TensorView &a, const TensorView &b, TensorView &out) = 0;
  virtual void add_inplace(TensorView &a, const TensorView &b) = 0;
  virtual void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                                TensorView &out) = 0;
  virtual void mul_scalar(const TensorView &x, float s, TensorView &out) = 0;
  virtual void relu(const TensorView &x, TensorView &out) = 0;
  virtual void matmul(const TensorView &a, const TensorView &b, TensorView &out) = 0;
  virtual void matmul_transposed(const TensorView &a, const TensorView &b,
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
  virtual void cross_entropy_mean(const TensorView &logits,
                                  const TensorView &targets,
                                  TensorView &out_loss) = 0;
  virtual float read_scalar_f32(const TensorView &x) = 0;
  virtual void backward_from_logits_targets(TensorView &logits,
                                            const TensorView &targets) = 0;
  virtual void softmax_rows(const TensorView &x, TensorView &out) = 0;
  virtual void apply_causal_mask_inplace(TensorView &scores,
                                         float neg_inf = -1e9f) = 0;
  virtual bool is_file2device_read_supported() const = 0;
  virtual void read_file2device(const std::string &path, void *dst,
                                uint64_t size, uint64_t file_offset) = 0;
};

class CpuBackend final : public DeviceBackend {
public:
  CpuBackend() = default;
  void *alloc(uint64_t bytes, uint32_t alignment) override;
  void free(void *ptr) override;
  void copy_host2device(void *dst, const void *src, uint64_t bytes) override;
  void copy_device2host(void *dst, const void *src, uint64_t bytes) override;
  void copy(const TensorView &src, TensorView &dst) override;
  void fill(TensorView &t, float v) override;
  void add(const TensorView &a, const TensorView &b, TensorView &out) override;
  void add_inplace(TensorView &a, const TensorView &b) override;
  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override;
  void mul_scalar(const TensorView &x, float s, TensorView &out) override;
  void relu(const TensorView &x, TensorView &out) override;
  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override;
  void matmul_transposed(const TensorView &a, const TensorView &b,
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
  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override;
  float read_scalar_f32(const TensorView &x) override;
  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override;
  void softmax_rows(const TensorView &x, TensorView &out) override;
  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override;
  bool is_file2device_read_supported() const override;
  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override;
};

class CudaBackend final : public DeviceBackend {
public:
  CudaBackend() = default;
  void *alloc(uint64_t bytes, uint32_t alignment) override;
  void free(void *ptr) override;
  void copy_host2device(void *dst, const void *src, uint64_t bytes) override;
  void copy_device2host(void *dst, const void *src, uint64_t bytes) override;
  void copy(const TensorView &src, TensorView &dst) override;
  void fill(TensorView &t, float v) override;
  void add(const TensorView &a, const TensorView &b, TensorView &out) override;
  void add_inplace(TensorView &a, const TensorView &b) override;
  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override;
  void mul_scalar(const TensorView &x, float s, TensorView &out) override;
  void relu(const TensorView &x, TensorView &out) override;
  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override;
  void matmul_transposed(const TensorView &a, const TensorView &b,
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
  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override;
  float read_scalar_f32(const TensorView &x) override;
  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override;
  void softmax_rows(const TensorView &x, TensorView &out) override;
  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override;
  bool is_file2device_read_supported() const override;
  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override;
};

std::unique_ptr<DeviceBackend> make_device_backend(Device device);
