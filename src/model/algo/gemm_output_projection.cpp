// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "gemm_output_projection.hpp"
#include "training_diagnostics_controller.hpp"

GemmOutputProjection::GemmOutputProjection(int layer_index,
                                           const LinearParams &params,
                                           const Ops &ops)
    : idx_(layer_index), params_(params), ops_(ops) {}

void GemmOutputProjection::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void GemmOutputProjection::forward(const TensorView &context, TensorView &out) {
  ops_.gemm_ranked_matrix_rhs(context, params_.compute_weight, out);
  ops_.add_bias_rowwise(out, params_.bias, out);
  cache_context_ = context;
}

void GemmOutputProjection::backward(const TensorView &dout,
                                    TensorView &dcontext) {
  ops_.gemm_ranked_reduce_lhs_t(cache_context_, dout, params_.dweight);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dWo(idx_, params_.dweight);
  ops_.row_sum(dout, params_.dbias);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dbo(idx_, params_.dbias);

  ops_.gemm_ranked_matrix_rhs_t(dout, params_.compute_weight, dcontext);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dcontext(idx_, dcontext);
}
