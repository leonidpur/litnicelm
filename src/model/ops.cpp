#include "ops.hpp"

#include <litnicelm_utils/checks.hpp>


Ops::Ops(DeviceBackend &device_backend) : device_backend_(device_backend) {}

void Ops::copy(const TensorView &src, TensorView &dst) const {
  checker().check_same_shape(src, dst, "Ops copy");
  device_backend_.copy(src, dst);
}
void Ops::cast(const TensorView &src, TensorView &dst) const {
  checker().check_same_shape(src, dst, "Ops cast");
  device_backend_.cast(src, dst);
}
void Ops::fill(TensorView &t, float v) const { device_backend_.fill(t, v); }

void Ops::add(const TensorView &a, const TensorView &b, TensorView &out) const {
  checker().check_same_shape(out, a, "Ops add");
  device_backend_.add(a, b, out);
}
void Ops::add_inplace(TensorView &a, const TensorView &b) const {
  checker().check_same_shape(a, b, "Ops add_inplace");
  device_backend_.add_inplace(a, b);
}
void Ops::add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                           TensorView &out) const {
  const int64_t last_dim = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
  checker().check_dims(bias_1xC, {1, last_dim},
                       "Ops add_bias_rowwise bias_1xC");
  checker().check_same_shape(out, x, "Ops add_bias_rowwise");
  device_backend_.add_bias_rowwise(x, bias_1xC, out);
}

void Ops::add_bias_relu_rowwise(const TensorView &x,
                                const TensorView &bias_1xC,
                                TensorView &out) const {
  const int64_t last_dim = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
  checker().check_dims(bias_1xC, {1, last_dim},
                       "Ops add_bias_relu_rowwise bias_1xC");
  checker().check_same_shape(out, x, "Ops add_bias_relu_rowwise");
  device_backend_.add_bias_relu_rowwise(x, bias_1xC, out);
}

void Ops::add_bias_relu_rowwise_inplace(TensorView &x,
                                        const TensorView &bias_1xC) const {
  const int64_t last_dim = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
  checker().check_dims(bias_1xC, {1, last_dim},
                       "Ops add_bias_relu_rowwise_inplace bias_1xC");
  device_backend_.add_bias_relu_rowwise_inplace(x, bias_1xC);
}

void Ops::mul_scalar(const TensorView &x, float s, TensorView &out) const {
  checker().check_same_shape(out, x, "Ops mul_scalar");
  device_backend_.mul_scalar(x, s, out);
}
float Ops::sum_squares_f32(const TensorView &x) const {
  checker().check_dtype_one_of(x, {DType::F32}, "Ops sum_squares_f32 x");
  return device_backend_.sum_squares_f32(x);
}
void Ops::relu(const TensorView &x, TensorView &out) const {
  checker().check_same_shape(out, x, "Ops relu");
  device_backend_.relu(x, out);
}
void Ops::relu_backward(const TensorView &preact, const TensorView &dout,
                        TensorView &dx) const {
  checker().check_same_shape(preact, dout, "Ops relu_backward");
  checker().check_same_shape(dx, preact, "Ops relu_backward");
  device_backend_.relu_backward(preact, dout, dx);
}
void Ops::relu_backward_inplace(const TensorView &preact,
                                TensorView &dout_dx) const {
  checker().check_same_shape(preact, dout_dx, "Ops relu_backward_inplace");
  device_backend_.relu_backward_inplace(preact, dout_dx);
}
void Ops::row_sum(const TensorView &x, TensorView &out_1xC) const {
  const int64_t last_dim = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
  checker().check_dims(out_1xC, {1, last_dim}, "Ops row_sum out_1xC");
  device_backend_.row_sum(x, out_1xC);
}

void Ops::gemm(const TensorView &a, const TensorView &b, TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Plain, false, false,
                         "Ops gemm");
  device_backend_.gemm(a, b, out);
}

void Ops::gemm_ranked_matrix_rhs(const TensorView &a, const TensorView &b,
                                 TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::BroadcastRhs, false, false,
                         "Ops gemm_ranked_matrix_rhs");
  device_backend_.gemm_ranked_matrix_rhs(a, b, out);
}

void Ops::gemm_ranked_matrix_rhs_t(const TensorView &a, const TensorView &b,
                                   TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::BroadcastRhs, false, true,
                         "Ops gemm_ranked_matrix_rhs_t");
  device_backend_.gemm_ranked_matrix_rhs_t(a, b, out);
}

void Ops::gemm_ranked_reduce_lhs_t(const TensorView &a, const TensorView &b,
                                   TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::ReducePrefix, true, false,
                         "Ops gemm_ranked_reduce_lhs_t");
  device_backend_.gemm_ranked_reduce_lhs_t(a, b, out);
}

void Ops::gemm_batched(const TensorView &a, const TensorView &b,
                       TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Batched, false, false,
                         "Ops gemm_batched");
  device_backend_.gemm_batched(a, b, out);
}

void Ops::gemm_batched_lhs_t(const TensorView &a, const TensorView &b,
                             TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Batched, true, false,
                         "Ops gemm_batched_lhs_t");
  device_backend_.gemm_batched_lhs_t(a, b, out);
}

void Ops::gemm_batched_rhs_t(const TensorView &a, const TensorView &b,
                             TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Batched, false, true,
                         "Ops gemm_batched_rhs_t");
  device_backend_.gemm_batched_rhs_t(a, b, out);
}
void Ops::transpose(const TensorView &x, TensorView &out) const {
  checker().check_rank(x, 2, "Ops transpose x");
  checker().check_dims(out, {x.dim(1), x.dim(0)}, "Ops transpose out");
  device_backend_.transpose(x, out);
}
void Ops::layernorm(const TensorView &x, const TensorView &gamma_1xC,
                    const TensorView &beta_1xC, TensorView &out) const {
  const int64_t last_dim = x.dim(x.rank() - 1);
  checker().check_dims(gamma_1xC, {1, last_dim}, "Ops layernorm gamma_1xC");
  checker().check_dims(beta_1xC, {1, last_dim}, "Ops layernorm beta_1xC");
  checker().check_same_shape(out, x, "Ops layernorm");
  device_backend_.layernorm_forward(x, gamma_1xC, beta_1xC, out);
}
void Ops::layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                             const TensorView &dout, TensorView &dx,
                             TensorView &dgamma_1xC,
                             TensorView &dbeta_1xC) const {
  const int64_t last_dim = x.dim(x.rank() - 1);
  checker().check_dims(gamma_1xC, {1, last_dim},
                       "Ops layernorm_backward gamma_1xC");
  checker().check_same_shape(dout, x, "Ops layernorm_backward");
  checker().check_same_shape(dx, x, "Ops layernorm_backward");
  checker().check_dims(dgamma_1xC, {1, last_dim},
                       "Ops layernorm_backward dgamma_1xC");
  checker().check_dims(dbeta_1xC, {1, last_dim},
                       "Ops layernorm_backward dbeta_1xC");
  device_backend_.layernorm_backward(x, gamma_1xC, dout, dx, dgamma_1xC,
                                     dbeta_1xC);
}
void Ops::embedding_lookup(const TensorView &table, const TensorView &ids,
                           TensorView &out) const {
  checker().check_dtype_one_of(ids, {DType::I32, DType::F32},
                               "Ops embedding_lookup ids");
  checker().check_rank(table, 2, "Ops embedding_lookup table");
  checker().check(CheckKind::Contract, out.rank() >= 1,
                  "Ops embedding_lookup", "out must have feature axis");
  checker().check(CheckKind::Contract, out.dim(out.rank() - 1) == table.dim(1),
                  "Ops embedding_lookup",
                  "out last dim must equal table dim(1)");
  checker().check(CheckKind::Contract,
                  ids.numel() * static_cast<uint64_t>(table.dim(1)) ==
                      out.numel(),
                  "Ops embedding_lookup", "ids/out numel mismatch");
  device_backend_.embedding_lookup(table, ids, out);
}
void Ops::accumulate_embedding_grads(const TensorView &ids,
                                     const TensorView &d_cur,
                                     TensorView &d_tok) const {
  checker().check_dtype_one_of(ids, {DType::I32, DType::F32},
                               "Ops accumulate_embedding_grads ids");
  checker().check(CheckKind::Contract,
                  ids.numel() * static_cast<uint64_t>(d_cur.dim(d_cur.rank() - 1)) ==
                      d_cur.numel(),
                  "Ops accumulate_embedding_grads", "ids/d_cur numel mismatch");
  checker().check(CheckKind::Contract,
                  d_tok.dim(1) == d_cur.dim(d_cur.rank() - 1),
                  "Ops accumulate_embedding_grads", "d_tok cols mismatch");
  device_backend_.accumulate_embedding_grads(ids, d_cur, d_tok);
}
void Ops::cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                             TensorView &out_loss) const {
  checker().check_dtype_one_of(targets, {DType::I32, DType::F32},
                               "Ops cross_entropy_mean targets");
  checker().check_dims(out_loss, {1, 1}, "Ops cross_entropy_mean out_loss");
  checker().check(CheckKind::Contract, logits.rank() >= 2,
                  "Ops cross_entropy_mean", "logits must have class axis");
  checker().check(CheckKind::Contract,
                  targets.numel() * static_cast<uint64_t>(logits.dim(logits.rank() - 1)) ==
                      logits.numel(),
                  "Ops cross_entropy_mean", "targets/logits numel mismatch");
  device_backend_.cross_entropy_mean(logits, targets, out_loss);
}
void Ops::cross_entropy_mean_backward_inplace(
    TensorView &logits, const TensorView &targets, TensorView &out_loss) const {
  checker().check_dtype_one_of(targets, {DType::I32, DType::F32},
                               "Ops cross_entropy_mean_backward_inplace targets");
  checker().check_dims(out_loss, {1, 1},
                       "Ops cross_entropy_mean_backward_inplace out_loss");
  checker().check(CheckKind::Contract,
                  logits.rank() >= 2 && logits.dim(logits.rank() - 1) > 0,
                  "Ops cross_entropy_mean_backward_inplace",
                  "invalid logits shape");
  checker().check(CheckKind::Contract,
                  targets.numel() * static_cast<uint64_t>(logits.dim(logits.rank() - 1)) ==
                      logits.numel(),
                  "Ops cross_entropy_mean_backward_inplace",
                  "targets/logits numel mismatch");
  device_backend_.cross_entropy_mean_backward_inplace(logits, targets, out_loss);
}
float Ops::read_scalar_f32(const TensorView &x) const {
  checker().check_dims(x, {1, 1}, "Ops read_scalar_f32 x");
  return device_backend_.read_scalar_f32(x);
}
bool Ops::supports_backward() const { return true; }
void Ops::backward_from_logits_targets(const TensorView &logits,
                                       const TensorView &targets) const {
  checker().check_dtype_one_of(targets, {DType::I32, DType::F32},
                               "Ops backward_from_logits_targets targets");
  checker().check(CheckKind::Contract,
                  logits.rank() >= 2 && logits.dim(logits.rank() - 1) > 0,
                  "Ops backward_from_logits_targets", "invalid logits shape");
  checker().check(CheckKind::Contract,
                  targets.numel() * static_cast<uint64_t>(logits.dim(logits.rank() - 1)) ==
                      logits.numel(),
                  "Ops backward_from_logits_targets",
                  "targets/logits numel mismatch");
  TensorView logits_mut = logits;
  device_backend_.backward_from_logits_targets(logits_mut, targets);
}

void Ops::softmax_rows(const TensorView &x, TensorView &out) const {
  checker().check_same_shape(out, x, "Ops softmax_rows");
  device_backend_.softmax_rows(x, out);
}
void Ops::softmax_backward_rows(const TensorView &softmax,
                                const TensorView &dout,
                                TensorView &dx) const {
  checker().check_same_shape(softmax, dout, "Ops softmax_backward_rows");
  checker().check_same_shape(dx, softmax, "Ops softmax_backward_rows");
  device_backend_.softmax_backward_rows(softmax, dout, dx);
}
void Ops::scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                     TensorView &out) const {
  checker().check_same_shape(out, scores, "Ops scaled_causal_softmax_rows");
  checker().check(CheckKind::Contract, scores.rank() >= 2,
                  "Ops scaled_causal_softmax_rows", "requires rank >= 2");
  checker().check(CheckKind::Contract,
                  scores.dim(scores.rank() - 2) ==
                      scores.dim(scores.rank() - 1),
                  "Ops scaled_causal_softmax_rows", "requires trailing [T,T]");
  device_backend_.scaled_causal_softmax_rows(scores, scale, out);
}

bool Ops::supports_exec_context_iteration() const {
  return device_backend_.supports_exec_context_iteration();
}

void Ops::start_exec_context_iteration() const {
  device_backend_.start_exec_context_iteration();
}

void Ops::finish_exec_context_iteration() const {
  device_backend_.finish_exec_context_iteration();
}

void Ops::start_exec_context_group() const {
  device_backend_.start_exec_context_group();
}

void Ops::finish_exec_context_group() const {
  device_backend_.finish_exec_context_group();
}

void Ops::gemm_batched_exec_context(const TensorView &a, const TensorView &b,
                                    TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Batched, false, false,
                         "Ops gemm_batched_exec_context");
  device_backend_.gemm_batched_exec_context(a, b, out);
}

void Ops::gemm_batched_rhs_t_exec_context(const TensorView &a,
                                          const TensorView &b,
                                          TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Batched, false, true,
                         "Ops gemm_batched_rhs_t_exec_context");
  device_backend_.gemm_batched_rhs_t_exec_context(a, b, out);
}

void Ops::gemm_batched_lhs_t_exec_context(const TensorView &a,
                                          const TensorView &b,
                                          TensorView &out) const {
  checker().check_matmul(a, b, out, MatmulKind::Batched, true, false,
                         "Ops gemm_batched_lhs_t_exec_context");
  device_backend_.gemm_batched_lhs_t_exec_context(a, b, out);
}

void Ops::scaled_causal_softmax_rows_exec_context(const TensorView &scores,
                                                  float scale,
                                                  TensorView &out) const {
  checker().check_same_shape(out, scores,
                             "Ops scaled_causal_softmax_rows_exec_context");
  checker().check(CheckKind::Contract, scores.rank() >= 2,
                  "Ops scaled_causal_softmax_rows_exec_context",
                  "requires rank >= 2");
  checker().check(CheckKind::Contract,
                  scores.dim(scores.rank() - 2) ==
                      scores.dim(scores.rank() - 1),
                  "Ops scaled_causal_softmax_rows_exec_context",
                  "requires trailing [T,T]");
  device_backend_.scaled_causal_softmax_rows_exec_context(scores, scale, out);
}

void Ops::softmax_backward_causal_rows_exec_context(const TensorView &softmax,
                                                    const TensorView &dout,
                                                    TensorView &dx) const {
  checker().check_same_shape(softmax, dout,
                             "Ops softmax_backward_causal_rows_exec_context");
  checker().check_same_shape(dx, softmax,
                             "Ops softmax_backward_causal_rows_exec_context");
  checker().check(CheckKind::Contract, softmax.rank() >= 2,
                  "Ops softmax_backward_causal_rows_exec_context",
                  "requires rank >= 2");
  checker().check(CheckKind::Contract,
                  softmax.dim(softmax.rank() - 2) ==
                      softmax.dim(softmax.rank() - 1),
                  "Ops softmax_backward_causal_rows_exec_context",
                  "requires trailing [T,T]");
  device_backend_.softmax_backward_causal_rows_exec_context(softmax, dout, dx);
}

void Ops::softmax_backward_causal_rows(const TensorView &softmax,
                                       const TensorView &dout,
                                       TensorView &dx) const {
  checker().check_same_shape(softmax, dout, "Ops softmax_backward_causal_rows");
  checker().check_same_shape(dx, softmax, "Ops softmax_backward_causal_rows");
  checker().check(CheckKind::Contract, softmax.rank() >= 2,
                  "Ops softmax_backward_causal_rows", "requires rank >= 2");
  checker().check(CheckKind::Contract,
                  softmax.dim(softmax.rank() - 2) ==
                      softmax.dim(softmax.rank() - 1),
                  "Ops softmax_backward_causal_rows",
                  "requires trailing [T,T]");
  device_backend_.softmax_backward_causal_rows(softmax, dout, dx);
}
void Ops::apply_causal_mask_inplace(TensorView &scores, float neg_inf) const {
  if (scores.rank() >= 2) {
    checker().check(CheckKind::Contract,
                    scores.dim(scores.rank() - 2) ==
                        scores.dim(scores.rank() - 1),
                    "Ops apply_causal_mask_inplace",
                    "scores must end with [T,T]");
  } else {
    checker().check(CheckKind::Contract, scores.dim(0) == scores.dim(1),
                    "Ops apply_causal_mask_inplace", "scores must be [T,T]");
  }
  device_backend_.apply_causal_mask_inplace(scores, neg_inf);
}
void Ops::rotary_embedding_inplace(TensorView &x, int64_t n_heads, float base,
                                   bool inverse) const {
  checker().check_rank(x, 3, "Ops rotary_embedding_inplace x");
  checker().check(CheckKind::Contract, n_heads > 0 && x.dim(2) % n_heads == 0,
                  "Ops rotary_embedding_inplace",
                  "D must be divisible by n_heads");
  checker().check(CheckKind::Contract, (x.dim(2) / n_heads) % 2 == 0,
                  "Ops rotary_embedding_inplace", "head_dim must be even");
  checker().check(CheckKind::Contract, base > 1.0f,
                  "Ops rotary_embedding_inplace", "base must be > 1");
  device_backend_.rotary_embedding_inplace(x, n_heads, base, inverse);
}
