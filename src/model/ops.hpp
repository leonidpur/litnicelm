#pragma once

#include "backend/device_backend.hpp"
#include "tensor.hpp"

class Ops {
public:
  explicit Ops(Device device, DeviceBackend &device_backend);

  void copy(const TensorView &src, TensorView &dst) const;
  void fill(TensorView &t, float v) const;

  void add(const TensorView &a, const TensorView &b, TensorView &out) const;
  void add_inplace(TensorView &a, const TensorView &b) const;
  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) const;

  void mul_scalar(const TensorView &x, float s, TensorView &out) const;
  void relu(const TensorView &x, TensorView &out) const;

  void matmul(const TensorView &a, const TensorView &b, TensorView &out) const;
  void matmul_left_transposed(const TensorView &a, const TensorView &b,
                              TensorView &out) const;
  void matmul_right_transposed(const TensorView &a, const TensorView &b,
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
  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) const;
  float read_scalar_f32(const TensorView &x) const;
  bool supports_backward() const;
  void backward_from_logits_targets(const TensorView &logits,
                                    const TensorView &targets) const;

  void softmax_rows(const TensorView &x, TensorView &out) const;
  void apply_causal_mask_inplace(TensorView &scores, float neg_inf = -1e9f) const;

private:
  Device device_ = Device::CPU;
  DeviceBackend &device_backend_;
};
