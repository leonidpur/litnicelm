#include "ops.hpp"

#include <utils/assert.hpp>

#include <algorithm>
#include <string>

#define require_ops(cond, msg)                                                  \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("Ops: ") + std::string(msg); })

namespace {
bool same_ranked_shape(const TensorView &a, const TensorView &b) {
  if (a.rank() != b.rank()) {
    return false;
  }
  for (size_t i = 0; i < a.rank(); ++i) {
    if (a.dim(i) != b.dim(i)) {
      return false;
    }
  }
  return true;
}

bool same_shape(const TensorView &a, const TensorView &b) {
  return same_ranked_shape(a, b);
}

bool same_ranked_prefix(const TensorView &a, const TensorView &b,
                        size_t prefix_rank) {
  if (a.rank() != b.rank() || a.rank() < prefix_rank) {
    return false;
  }
  for (size_t i = 0; i < prefix_rank; ++i) {
    if (a.dim(i) != b.dim(i)) {
      return false;
    }
  }
  return true;
}

} // namespace

Ops::Ops(DeviceBackend &device_backend) : device_backend_(device_backend) {}

void Ops::copy(const TensorView &src, TensorView &dst) const {
  require_ops(same_shape(src, dst), "copy shape mismatch");
  device_backend_.copy(src, dst);
}
void Ops::fill(TensorView &t, float v) const { device_backend_.fill(t, v); }

void Ops::add(const TensorView &a, const TensorView &b, TensorView &out) const {
  require_ops(same_shape(out, a), "add out shape mismatch");
  device_backend_.add(a, b, out);
}
void Ops::add_inplace(TensorView &a, const TensorView &b) const {
  require_ops(same_shape(a, b),
              "add_inplace shape mismatch");
  device_backend_.add_inplace(a, b);
}
void Ops::add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                           TensorView &out) const {
  const int64_t last_dim = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
  require_ops(bias_1xC.rank() == 2 && bias_1xC.dim(0) == 1,
              "bias must be [1,C]");
  require_ops(bias_1xC.dim(1) == last_dim, "bias C mismatch");
  require_ops(same_shape(out, x),
              "add out shape mismatch");
  device_backend_.add_bias_rowwise(x, bias_1xC, out);
}

void Ops::mul_scalar(const TensorView &x, float s, TensorView &out) const {
  require_ops(same_shape(out, x), "mul_scalar shape mismatch");
  device_backend_.mul_scalar(x, s, out);
}
float Ops::sum_squares_f32(const TensorView &x) const {
  require_ops(x.dtype() == DType::F32, "sum_squares_f32 requires F32");
  return device_backend_.sum_squares_f32(x);
}
void Ops::relu(const TensorView &x, TensorView &out) const {
  require_ops(same_shape(out, x), "relu shape mismatch");
  device_backend_.relu(x, out);
}
void Ops::relu_backward(const TensorView &preact, const TensorView &dout,
                        TensorView &dx) const {
  require_ops(same_shape(preact, dout), "relu_backward shape mismatch");
  require_ops(same_shape(dx, preact), "relu_backward dx shape mismatch");
  device_backend_.relu_backward(preact, dout, dx);
}
void Ops::row_sum(const TensorView &x, TensorView &out_1xC) const {
  const int64_t last_dim = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
  require_ops(out_1xC.rank() == 2 && out_1xC.dim(0) == 1 &&
                  out_1xC.dim(1) == last_dim,
              "row_sum out shape mismatch");
  device_backend_.row_sum(x, out_1xC);
}

void Ops::matmul(const TensorView &a, const TensorView &b, TensorView &out) const {
  if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank()) {
    require_ops(same_ranked_prefix(a, b, a.rank() - 2),
                "matmul prefix dim mismatch");
    require_ops(same_ranked_prefix(a, out, a.rank() - 2),
                "matmul out prefix dim mismatch");
    require_ops(a.dim(a.rank() - 1) == b.dim(b.rank() - 2),
                "matmul inner dim mismatch");
    require_ops(out.dim(out.rank() - 2) == a.dim(a.rank() - 2) &&
                    out.dim(out.rank() - 1) == b.dim(b.rank() - 1),
                "matmul out shape mismatch");
    device_backend_.matmul(a, b, out);
    return;
  }
  if (a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank()) {
    require_ops(same_ranked_prefix(a, out, a.rank() - 2),
                "matmul out prefix dim mismatch");
    require_ops(a.dim(a.rank() - 1) == b.dim(0), "matmul inner dim mismatch");
    require_ops(out.dim(out.rank() - 2) == a.dim(a.rank() - 2) &&
                    out.dim(out.rank() - 1) == b.dim(1),
                "matmul out shape mismatch");
    device_backend_.matmul(a, b, out);
    return;
  }
  require_ops(b.dim(0) == a.dim(1), "matmul inner dim mismatch");
  require_ops(out.dim(0) == a.dim(0) && out.dim(1) == b.dim(1),
              "matmul out shape mismatch");
  device_backend_.matmul(a, b, out);
}
void Ops::matmul_left_transposed(const TensorView &a, const TensorView &b,
                                 TensorView &out) const {
  if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank()) {
    require_ops(same_ranked_prefix(a, b, a.rank() - 2),
                "matmul_left_transposed prefix dim mismatch");
    require_ops(same_ranked_prefix(a, out, a.rank() - 2),
                "matmul_left_transposed out prefix dim mismatch");
    require_ops(a.dim(a.rank() - 2) == b.dim(b.rank() - 2),
                "matmul_left_transposed shared dim mismatch");
    require_ops(out.dim(out.rank() - 2) == a.dim(a.rank() - 1) &&
                    out.dim(out.rank() - 1) == b.dim(b.rank() - 1),
                "matmul_left_transposed out shape mismatch");
    device_backend_.matmul_left_transposed(a, b, out);
    return;
  }
  if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == 2) {
    require_ops(same_ranked_prefix(a, b, a.rank() - 2),
                "matmul_left_transposed prefix dim mismatch");
    require_ops(a.dim(a.rank() - 2) == b.dim(b.rank() - 2),
                "matmul_left_transposed shared dim mismatch");
    require_ops(out.dim(0) == a.dim(a.rank() - 1) &&
                    out.dim(1) == b.dim(b.rank() - 1),
                "matmul_left_transposed out shape mismatch");
    device_backend_.matmul_left_transposed(a, b, out);
    return;
  }
  require_ops(a.dim(0) == b.dim(0),
              "matmul_left_transposed shared row dim mismatch");
  require_ops(out.dim(0) == a.dim(1) && out.dim(1) == b.dim(1),
              "matmul_left_transposed out shape mismatch");
  device_backend_.matmul_left_transposed(a, b, out);
}
void Ops::matmul_right_transposed(const TensorView &a, const TensorView &b,
                                  TensorView &out) const {
  if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank()) {
    require_ops(same_ranked_prefix(a, b, a.rank() - 2),
                "matmul_right_transposed prefix dim mismatch");
    require_ops(same_ranked_prefix(a, out, a.rank() - 2),
                "matmul_right_transposed out prefix dim mismatch");
    require_ops(a.dim(a.rank() - 1) == b.dim(b.rank() - 1),
                "matmul_right_transposed inner dim mismatch");
    require_ops(out.dim(out.rank() - 2) == a.dim(a.rank() - 2) &&
                    out.dim(out.rank() - 1) == b.dim(b.rank() - 2),
                "matmul_right_transposed out shape mismatch");
    device_backend_.matmul_right_transposed(a, b, out);
    return;
  }
  if (a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank()) {
    require_ops(same_ranked_prefix(a, out, a.rank() - 2),
                "matmul_right_transposed out prefix dim mismatch");
    require_ops(a.dim(a.rank() - 1) == b.dim(1),
                "matmul_right_transposed inner dim mismatch");
    require_ops(out.dim(out.rank() - 2) == a.dim(a.rank() - 2) &&
                    out.dim(out.rank() - 1) == b.dim(0),
                "matmul_right_transposed out shape mismatch");
    device_backend_.matmul_right_transposed(a, b, out);
    return;
  }
  require_ops(a.dim(1) == b.dim(1),
              "matmul_right_transposed inner dim mismatch");
  require_ops(out.dim(0) == a.dim(0) && out.dim(1) == b.dim(0),
              "matmul_right_transposed out shape mismatch");
  device_backend_.matmul_right_transposed(a, b, out);
}
void Ops::transpose(const TensorView &x, TensorView &out) const {
  require_ops(out.dim(0) == x.dim(1) && out.dim(1) == x.dim(0),
              "transpose shape mismatch");
  device_backend_.transpose(x, out);
}
void Ops::layernorm(const TensorView &x, const TensorView &gamma_1xC,
                    const TensorView &beta_1xC, TensorView &out) const {
  const int64_t last_dim = x.dim(x.rank() - 1);
  require_ops(gamma_1xC.rank() == 2 && gamma_1xC.dim(0) == 1 &&
                  gamma_1xC.dim(1) == last_dim,
              "layernorm gamma must be [1,C]");
  require_ops(beta_1xC.rank() == 2 && beta_1xC.dim(0) == 1 &&
                  beta_1xC.dim(1) == last_dim,
              "layernorm beta must be [1,C]");
  require_ops(same_ranked_shape(out, x),
              "layernorm out shape mismatch");
  device_backend_.layernorm_forward(x, gamma_1xC, beta_1xC, out);
}
void Ops::layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                             const TensorView &dout, TensorView &dx,
                             TensorView &dgamma_1xC,
                             TensorView &dbeta_1xC) const {
  const int64_t last_dim = x.dim(x.rank() - 1);
  require_ops(gamma_1xC.rank() == 2 && gamma_1xC.dim(0) == 1 &&
                  gamma_1xC.dim(1) == last_dim,
              "layernorm_backward gamma shape mismatch");
  require_ops(same_ranked_shape(dout, x),
              "layernorm_backward dout shape mismatch");
  require_ops(same_ranked_shape(dx, x),
              "layernorm_backward dx shape mismatch");
  require_ops(dgamma_1xC.rank() == 2 && dgamma_1xC.dim(0) == 1 &&
                  dgamma_1xC.dim(1) == last_dim,
              "layernorm_backward dgamma shape mismatch");
  require_ops(dbeta_1xC.rank() == 2 && dbeta_1xC.dim(0) == 1 &&
                  dbeta_1xC.dim(1) == last_dim,
              "layernorm_backward dbeta shape mismatch");
  device_backend_.layernorm_backward(x, gamma_1xC, dout, dx, dgamma_1xC,
                                     dbeta_1xC);
}
void Ops::embedding_lookup(const TensorView &table, const TensorView &ids,
                           TensorView &out) const {
  require_ops(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
              "embedding_lookup(ids) requires I32 or F32");
  require_ops(table.rank() == 2, "embedding_lookup table must be [V, D]");
  require_ops(out.rank() >= 1, "embedding_lookup out must have feature axis");
  require_ops(out.dim(out.rank() - 1) == table.dim(1),
              "embedding_lookup out last dim must equal table dim(1)");
  require_ops(ids.numel() * static_cast<uint64_t>(table.dim(1)) == out.numel(),
              "embedding_lookup ids/out numel mismatch");
  device_backend_.embedding_lookup(table, ids, out);
}
void Ops::accumulate_embedding_grads(const TensorView &ids,
                                     const TensorView &d_cur, TensorView &d_tok,
                                     TensorView &d_pos) const {
  require_ops(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
              "accumulate_embedding_grads(ids) requires I32 or F32");
  require_ops(ids.numel() * static_cast<uint64_t>(d_cur.dim(d_cur.rank() - 1)) ==
                  d_cur.numel(),
              "accumulate_embedding_grads ids/d_cur numel mismatch");
  require_ops(d_tok.dim(1) == d_cur.dim(d_cur.rank() - 1),
              "accumulate_embedding_grads d_tok cols mismatch");
  require_ops(d_pos.dim(1) == d_cur.dim(d_cur.rank() - 1),
              "accumulate_embedding_grads d_pos cols mismatch");
  require_ops(ids.rank() >= 1 && d_pos.dim(0) >= ids.dim(ids.rank() - 1),
              "accumulate_embedding_grads d_pos rows too small");
  device_backend_.accumulate_embedding_grads(ids, d_cur, d_tok, d_pos);
}
void Ops::cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                             TensorView &out_loss) const {
  require_ops(targets.dtype() == DType::I32 || targets.dtype() == DType::F32,
              "cross_entropy_mean(targets) requires I32 or F32");
  require_ops(out_loss.rank() == 2 && out_loss.dim(0) == 1 &&
                  out_loss.dim(1) == 1,
              "cross_entropy_mean out_loss must be [1,1]");
  require_ops(logits.rank() >= 2, "cross_entropy_mean logits must have class axis");
  require_ops(targets.numel() * static_cast<uint64_t>(logits.dim(logits.rank() - 1)) ==
                  logits.numel(),
              "cross_entropy_mean targets/logits numel mismatch");
  device_backend_.cross_entropy_mean(logits, targets, out_loss);
}
void Ops::cross_entropy_mean_backward_inplace(
    TensorView &logits, const TensorView &targets, TensorView &out_loss) const {
  require_ops(targets.dtype() == DType::I32 || targets.dtype() == DType::F32,
              "cross_entropy_mean_backward_inplace(targets) requires I32 or F32");
  require_ops(out_loss.rank() == 2 && out_loss.dim(0) == 1 &&
                  out_loss.dim(1) == 1,
              "cross_entropy_mean_backward_inplace out_loss must be [1,1]");
  require_ops(logits.rank() >= 2 && logits.dim(logits.rank() - 1) > 0,
              "cross_entropy_mean_backward_inplace invalid logits shape");
  require_ops(targets.numel() * static_cast<uint64_t>(logits.dim(logits.rank() - 1)) ==
                  logits.numel(),
              "cross_entropy_mean_backward_inplace targets/logits numel mismatch");
  device_backend_.cross_entropy_mean_backward_inplace(logits, targets, out_loss);
}
float Ops::read_scalar_f32(const TensorView &x) const {
  require_ops(x.rank() == 2 && x.dim(0) == 1 && x.dim(1) == 1,
              "read_scalar_f32 x must be [1,1]");
  return device_backend_.read_scalar_f32(x);
}
bool Ops::supports_backward() const { return true; }
void Ops::backward_from_logits_targets(const TensorView &logits,
                                       const TensorView &targets) const {
  require_ops(targets.dtype() == DType::I32 || targets.dtype() == DType::F32,
              "backward_from_logits_targets(targets) requires I32 or F32");
  require_ops(logits.rank() >= 2 && logits.dim(logits.rank() - 1) > 0,
              "backward_from_logits_targets invalid logits shape");
  require_ops(targets.numel() * static_cast<uint64_t>(logits.dim(logits.rank() - 1)) ==
                  logits.numel(),
              "backward_from_logits_targets targets/logits numel mismatch");
  TensorView logits_mut = logits;
  device_backend_.backward_from_logits_targets(logits_mut, targets);
}

void Ops::softmax_rows(const TensorView &x, TensorView &out) const {
  require_ops(same_shape(out, x) || same_ranked_shape(out, x),
              "softmax_rows shape mismatch");
  device_backend_.softmax_rows(x, out);
}
void Ops::softmax_backward_rows(const TensorView &softmax,
                                const TensorView &dout,
                                TensorView &dx) const {
  require_ops(same_shape(softmax, dout), "softmax_backward_rows shape mismatch");
  require_ops(same_shape(dx, softmax), "softmax_backward_rows dx shape mismatch");
  device_backend_.softmax_backward_rows(softmax, dout, dx);
}
void Ops::apply_causal_mask_inplace(TensorView &scores, float neg_inf) const {
  if (scores.rank() >= 2) {
    require_ops(scores.dim(scores.rank() - 2) == scores.dim(scores.rank() - 1),
                "apply_causal_mask_inplace scores must end with [T,T]");
  } else {
    require_ops(scores.dim(0) == scores.dim(1),
                "apply_causal_mask_inplace scores must be [T,T]");
  }
  device_backend_.apply_causal_mask_inplace(scores, neg_inf);
}

#undef require_ops
