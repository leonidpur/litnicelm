// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "fused_linear_ffn.hpp"
#include "tensor_contracts.hpp"
#include "training_diagnostics_controller.hpp"

#include <litnicelm_utils/checks.hpp>

FusedLinearFFN::FusedLinearFFN(int layer_index, const Config &cfg,
                               TensorStore &tensor_store,
                               const FfnParams &params, const Ops &ops)
    : idx_(layer_index),
      cfg_(cfg),
      tensorStore_(tensor_store),
      params_(params),
      ops_(ops) {
  TensorContracts::check_ffn_params(
      params_.w1.weight, params_.w1.bias, params_.w2.weight, params_.w2.bias,
      static_cast<int64_t>(cfg_.model.d_model),
      static_cast<int64_t>(cfg_.model.d_ff), "FusedLinearFFN");
}

void FusedLinearFFN::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void FusedLinearFFN::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void FusedLinearFFN::forward(const TensorView &x, TensorView &out) {
  observer_->on_ffn_start(idx_);
  const TensorContracts::BatchSeqDims dims = TensorContracts::check_bsd_io(
      x, out, static_cast<int64_t>(cfg_.model.d_model), "FusedLinearFFN");
  TensorView h = tensorStore_.temp_ffn_h(idx_, dims.batch_size, dims.seq_len);

  ops_.linear_forward(x, params_.w1.compute_weight, params_.w1.bias, true, h);
  ops_.linear_forward(h, params_.w2.compute_weight, params_.w2.bias, false, out);

  cache_x_ = x;
  cache_h_ = h;
  observer_->on_ffn_end(idx_);
}

void FusedLinearFFN::backward(const TensorView &dout, TensorView &dx) {
  observer_->on_ffn_start(idx_);
  checker().check_same_device_dtype(dout, cache_x_, "FusedLinearFFN dout/cache_x");
  checker().check_same_shape(dx, cache_x_, "FusedLinearFFN dx");
  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);

  ops_.linear_backward_params_accumulate(cache_h_, dout, params_.w2.dweight, params_.w2.dbias);
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_dW2(idx_, params_.w2.dweight);
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_db2(idx_, params_.w2.dbias);

  TensorView da = tensorStore_.temp_ffn_da(idx_, batch_size, seq_len);
  ops_.gemm_ranked_matrix_rhs_t(dout, params_.w2.compute_weight, da);
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_da(idx_, da);
  ops_.relu_backward_inplace(cache_h_, da);
  TensorView dh = da;
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_dh(idx_, dh);

  ops_.linear_backward_params_accumulate(cache_x_, dh, params_.w1.dweight, params_.w1.dbias);
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_dW1(idx_, params_.w1.dweight);
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_db1(idx_, params_.w1.dbias);

  ops_.gemm_ranked_matrix_rhs_t(dh, params_.w1.compute_weight, dx);
  if (diagnostics_ != nullptr) diagnostics_->bk_ffn_dx(idx_, dx);
  observer_->on_ffn_end(idx_);
}
