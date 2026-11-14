// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "fused_linear_qkv_projection.hpp"
#include "training_diagnostics_controller.hpp"

FusedLinearQkvProjection::FusedLinearQkvProjection(int layer_index, const LinearParams &params, const Ops &ops)
    : idx_(layer_index), params_(params), ops_(ops) {}

void FusedLinearQkvProjection::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void FusedLinearQkvProjection::forward(const TensorView &x, TensorView &qkv) {
  ops_.linear_forward(x, params_.compute_weight, params_.bias, false, qkv);
  cache_x_ = x;
}

void FusedLinearQkvProjection::backward(const TensorView &dqkv, TensorView &dx) {
  ops_.linear_backward_params_accumulate(cache_x_, dqkv, params_.dweight, params_.dbias);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dWqkv(idx_, params_.dweight);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dbqkv(idx_, params_.dbias);
  ops_.gemm_ranked_matrix_rhs_t(dqkv, params_.compute_weight, dx);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dx(idx_, dx);
}
