#include "ops.hpp"

#include <utils/assert.hpp>

#include <algorithm>
#include <string>

#define require_ops(cond, msg)                                                  \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("Ops: ") + std::string(msg); })

Ops::Ops(Device device, DeviceBackend &device_backend)
    : device_(device), device_backend_(device_backend) {
  require_ops(device_ == Device::CPU, "GPU backend not implemented");
}

void Ops::copy(const TensorView &src, TensorView &dst) const {
  require_ops(src.shape().r == dst.shape().r && src.shape().c == dst.shape().c,
              "copy shape mismatch");
  device_backend_.copy(src, dst);
}
void Ops::fill(TensorView &t, float v) const { device_backend_.fill(t, v); }

void Ops::add(const TensorView &a, const TensorView &b, TensorView &out) const {
  require_ops(a.shape().r == b.shape().r && a.shape().c == b.shape().c,
              "add shape mismatch");
  require_ops(out.shape().r == a.shape().r && out.shape().c == a.shape().c,
              "add out shape mismatch");
  device_backend_.add(a, b, out);
}
void Ops::add_inplace(TensorView &a, const TensorView &b) const {
  require_ops(a.shape().r == b.shape().r && a.shape().c == b.shape().c,
              "add_inplace shape mismatch");
  device_backend_.add_inplace(a, b);
}
void Ops::add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                           TensorView &out) const {
  require_ops(bias_1xC.shape().r == 1, "bias must be [1,C]");
  require_ops(bias_1xC.shape().c == x.shape().c, "bias C mismatch");
  require_ops(out.shape().r == x.shape().r && out.shape().c == x.shape().c,
              "out shape mismatch");
  device_backend_.add_bias_rowwise(x, bias_1xC, out);
}

void Ops::mul_scalar(const TensorView &x, float s, TensorView &out) const {
  require_ops(out.shape().r == x.shape().r && out.shape().c == x.shape().c,
              "mul_scalar shape mismatch");
  device_backend_.mul_scalar(x, s, out);
}
void Ops::relu(const TensorView &x, TensorView &out) const {
  require_ops(out.shape().r == x.shape().r && out.shape().c == x.shape().c,
              "relu shape mismatch");
  device_backend_.relu(x, out);
}

void Ops::matmul(const TensorView &a, const TensorView &b, TensorView &out) const {
  require_ops(b.shape().r == a.shape().c, "matmul inner dim mismatch");
  require_ops(out.shape().r == a.shape().r && out.shape().c == b.shape().c,
              "matmul out shape mismatch");
  device_backend_.matmul(a, b, out);
}
void Ops::matmul_transposed(const TensorView &a, const TensorView &b,
                            TensorView &out) const {
  require_ops(b.shape().c == a.shape().c, "matmul_transposed inner dim mismatch");
  require_ops(out.shape().r == a.shape().r && out.shape().c == b.shape().r,
              "matmul_transposed out shape mismatch");
  device_backend_.matmul_transposed(a, b, out);
}
void Ops::transpose(const TensorView &x, TensorView &out) const {
  require_ops(out.shape().r == x.shape().c && out.shape().c == x.shape().r,
              "transpose shape mismatch");
  device_backend_.transpose(x, out);
}
void Ops::layernorm(const TensorView &x, const TensorView &gamma_1xC,
                    const TensorView &beta_1xC, TensorView &out) const {
  require_ops(gamma_1xC.shape().r == 1 && gamma_1xC.shape().c == x.shape().c,
              "layernorm gamma must be [1,C]");
  require_ops(beta_1xC.shape().r == 1 && beta_1xC.shape().c == x.shape().c,
              "layernorm beta must be [1,C]");
  require_ops(out.shape().r == x.shape().r && out.shape().c == x.shape().c,
              "layernorm out shape mismatch");
  device_backend_.layernorm_forward(x, gamma_1xC, beta_1xC, out);
}
void Ops::layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                             const TensorView &dout, TensorView &dx,
                             TensorView &dgamma_1xC,
                             TensorView &dbeta_1xC) const {
  require_ops(gamma_1xC.shape().r == 1 && gamma_1xC.shape().c == x.shape().c,
              "layernorm_backward gamma shape mismatch");
  require_ops(dout.shape().r == x.shape().r && dout.shape().c == x.shape().c,
              "layernorm_backward dout shape mismatch");
  require_ops(dx.shape().r == x.shape().r && dx.shape().c == x.shape().c,
              "layernorm_backward dx shape mismatch");
  require_ops(dgamma_1xC.shape().r == 1 && dgamma_1xC.shape().c == x.shape().c,
              "layernorm_backward dgamma shape mismatch");
  require_ops(dbeta_1xC.shape().r == 1 && dbeta_1xC.shape().c == x.shape().c,
              "layernorm_backward dbeta shape mismatch");
  device_backend_.layernorm_backward(x, gamma_1xC, dout, dx, dgamma_1xC,
                                     dbeta_1xC);
}
void Ops::embedding_lookup(const TensorView &table, const TensorView &ids,
                           TensorView &out) const {
  require_ops(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
              "embedding_lookup(ids) requires I32 or F32");
  require_ops(out.shape().c == table.shape().c,
              "embedding_lookup out cols must equal table cols");
  require_ops(ids.shape().r == out.shape().r,
              "embedding_lookup ids rows must equal out rows");
  require_ops(ids.shape().c == 1 || ids.shape().c == 0,
              "embedding_lookup ids must be [T] or [T,1]");
  device_backend_.embedding_lookup(table, ids, out);
}
void Ops::cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                             TensorView &out_loss) const {
  require_ops(targets.dtype() == DType::I32 || targets.dtype() == DType::F32,
              "cross_entropy_mean(targets) requires I32 or F32");
  require_ops(out_loss.shape().r == 1 && out_loss.shape().c == 1,
              "cross_entropy_mean out_loss must be [1,1]");
  require_ops(targets.shape().r == logits.shape().r,
              "cross_entropy_mean targets rows mismatch");
  require_ops(targets.shape().c == 1 || targets.shape().c == 0,
              "cross_entropy_mean targets must be [T] or [T,1]");
  device_backend_.cross_entropy_mean(logits, targets, out_loss);
}
float Ops::read_scalar_f32(const TensorView &x) const {
  require_ops(x.shape().r == 1 && x.shape().c == 1,
              "read_scalar_f32 x must be [1,1]");
  return device_backend_.read_scalar_f32(x);
}
bool Ops::supports_backward() const { return true; }
void Ops::backward_from_logits_targets(const TensorView &logits,
                                       const TensorView &targets) const {
  require_ops(targets.dtype() == DType::I32 || targets.dtype() == DType::F32,
              "backward_from_logits_targets(targets) requires I32 or F32");
  require_ops(targets.shape().r == logits.shape().r,
              "backward_from_logits_targets targets rows mismatch");
  require_ops(targets.shape().c == 1 || targets.shape().c == 0,
              "backward_from_logits_targets targets must be [T] or [T,1]");
  require_ops(logits.shape().r > 0 && logits.shape().c > 0,
              "backward_from_logits_targets invalid logits shape");
  TensorView &logits_mut = const_cast<TensorView &>(logits);
  device_backend_.backward_from_logits_targets(logits_mut, targets);
}

void Ops::softmax_rows(const TensorView &x, TensorView &out) const {
  require_ops(out.shape().r == x.shape().r && out.shape().c == x.shape().c,
              "softmax_rows shape mismatch");
  device_backend_.softmax_rows(x, out);
}
void Ops::apply_causal_mask_inplace(TensorView &scores, float neg_inf) const {
  require_ops(scores.shape().r == scores.shape().c,
              "apply_causal_mask_inplace scores must be [T,T]");
  device_backend_.apply_causal_mask_inplace(scores, neg_inf);
}

#undef require_ops
