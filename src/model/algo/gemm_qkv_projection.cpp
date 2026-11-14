// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "gemm_qkv_projection.hpp"
#include "training_diagnostics_controller.hpp"

GemmQkvProjection::GemmQkvProjection(int layer_index,
                                     const LinearParams &params,
                                     const Ops &ops)
    : idx_(layer_index), params_(params), ops_(ops) {}

void GemmQkvProjection::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void GemmQkvProjection::forward(const TensorView &x, TensorView &qkv) {
  ops_.gemm_ranked_matrix_rhs(x, params_.compute_weight, qkv);
  ops_.add_bias_rowwise(qkv, params_.bias, qkv);
  cache_x_ = x;
}

void GemmQkvProjection::backward(const TensorView &dqkv, TensorView &dx) {
  ops_.gemm_ranked_matrix_rhs_t(dqkv, params_.compute_weight, dx);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dx(idx_, dx);

  ops_.gemm_ranked_reduce_lhs_t_accumulate(cache_x_, dqkv, params_.dweight);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dWqkv(idx_, params_.dweight);
  ops_.row_sum(dqkv, params_.dbias);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dbqkv(idx_, params_.dbias);
}
