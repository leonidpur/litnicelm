#include "ops.hpp"

#include <stdexcept>

Ops::Ops(Device device) : device_(device) {
  if (device_ != Device::CPU) {
    throw std::runtime_error("Ops: GPU backend not implemented");
  }
}

void Ops::copy(const TensorView &src, TensorView &dst) const { cpu_.copy(src, dst); }
void Ops::fill(TensorView &t, float v) const { cpu_.fill(t, v); }

void Ops::add(const TensorView &a, const TensorView &b, TensorView &out) const {
  cpu_.add(a, b, out);
}
void Ops::add_inplace(TensorView &a, const TensorView &b) const {
  cpu_.add_inplace(a, b);
}
void Ops::add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                           TensorView &out) const {
  cpu_.add_bias_rowwise(x, bias_1xC, out);
}

void Ops::mul_scalar(const TensorView &x, float s, TensorView &out) const {
  cpu_.mul_scalar(x, s, out);
}
void Ops::relu(const TensorView &x, TensorView &out) const { cpu_.relu(x, out); }

void Ops::matmul(const TensorView &a, const TensorView &b, TensorView &out) const {
  cpu_.matmul(a, b, out);
}
void Ops::matmul_transposed(const TensorView &a, const TensorView &b,
                            TensorView &out) const {
  cpu_.matmul_transposed(a, b, out);
}
void Ops::transpose(const TensorView &x, TensorView &out) const {
  cpu_.transpose(x, out);
}
void Ops::layernorm(const TensorView &x, const TensorView &gamma_1xC,
                    const TensorView &beta_1xC, TensorView &out) const {
  cpu_.layernorm(x, gamma_1xC, beta_1xC, out);
}
void Ops::embedding_lookup(const TensorView &table, const TensorView &ids,
                           TensorView &out) const {
  cpu_.embedding_lookup(table, ids, out);
}
void Ops::cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                             TensorView &out_loss) const {
  cpu_.cross_entropy_mean(logits, targets, out_loss);
}
float Ops::read_scalar_f32(const TensorView &x) const {
  return cpu_.read_scalar_f32(x);
}
bool Ops::supports_backward() const { return true; }
void Ops::backward_from_logits_targets(const TensorView &logits,
                                       const TensorView &targets) const {
  TensorView &logits_mut = const_cast<TensorView &>(logits);
  cpu_.backward_from_logits_targets(logits_mut, targets);
}

void Ops::softmax_rows(const TensorView &x, TensorView &out) const {
  cpu_.softmax_rows(x, out);
}
void Ops::apply_causal_mask_inplace(TensorView &scores, float neg_inf) const {
  cpu_.apply_causal_mask_inplace(scores, neg_inf);
}
