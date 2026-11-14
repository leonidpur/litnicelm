// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "fused_linear_output_projection.hpp"
#include "training_diagnostics_controller.hpp"

FusedLinearOutputProjection::FusedLinearOutputProjection(int layer_index, const LinearParams &params, const Ops &ops)
    : idx_(layer_index), params_(params), ops_(ops) {}

void FusedLinearOutputProjection::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void FusedLinearOutputProjection::forward(const TensorView &context, TensorView &out) {
  ops_.linear_forward(context, params_.compute_weight, params_.bias, false, out);
  cache_context_ = context;
}

void FusedLinearOutputProjection::backward(const TensorView &dout, TensorView &dcontext) {
  ops_.linear_backward_params_accumulate(cache_context_, dout, params_.dweight, params_.dbias);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dWo(idx_, params_.dweight);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dbo(idx_, params_.dbias);
  ops_.gemm_ranked_matrix_rhs_t(dout, params_.compute_weight, dcontext);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dcontext(idx_, dcontext);
}
