// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "self_attention_fused_inplace_multistream.hpp"
#include "slice_views.hpp"
#include "tensor_contracts.hpp"
#include "training_diagnostics_controller.hpp"

#include <litnicelm_utils/checks.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

SelfAttentionFusedInplaceMultistream::SelfAttentionFusedInplaceMultistream(
    int layer_index, const Config &cfg, const AttentionParams &params,
    const Ops &ops,
    IPositionEncoding &position_encoding, NodeSlices slices)
    : idx_(layer_index),
      cfg_(cfg),
      params_(params),
      ops_(ops),
      positionEncoding_(position_encoding),
      slices_{slices.find("attn.qkv"),      slices.find("attn.context"),
              slices.find("attn.weights_cache"),
              slices.find("attn.dcontext"), slices.find("attn.dqkv"),
              slices.find("attn.dweights"), slices.find("attn.dscores")} {}

std::vector<SliceRequest>
SelfAttentionFusedInplaceMultistream::slice_requests(const AttentionDims &dims) {
  const int64_t B = dims.batch_size;
  const int64_t S = dims.seq_len;
  const int64_t D = dims.model_dim;
  std::vector<SliceRequest> requests{
      {"attn.qkv", Shape{B, S, 3 * D}, ItemType::Activation, SliceRole::Saved,
       SliceLifetime::Step},
      {"attn.context", Shape{B, S, D}, ItemType::Activation, SliceRole::Saved,
       SliceLifetime::Step},
      {"attn.weights_cache", Shape{dims.n_heads, B, S, S}, ItemType::Activation,
       SliceRole::Saved, SliceLifetime::Step},
  };
  if (dims.with_backward) {
    requests.push_back({"attn.dcontext", Shape{B, S, D}, ItemType::Activation,
                        SliceRole::Workspace, SliceLifetime::Op});
    requests.push_back({"attn.dqkv", Shape{B, S, 3 * D}, ItemType::Activation,
                        SliceRole::Workspace, SliceLifetime::Op});
    requests.push_back({"attn.dweights", Shape{B, S, S}, ItemType::Activation,
                        SliceRole::Workspace, SliceLifetime::Op});
    requests.push_back({"attn.dscores", Shape{B, S, S}, ItemType::Activation,
                        SliceRole::Workspace, SliceLifetime::Op});
  }
  return requests;
}

void SelfAttentionFusedInplaceMultistream::set_observer(
    ITrainingObserver *observer) {
  observer_ = observer;
}

void SelfAttentionFusedInplaceMultistream::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void SelfAttentionFusedInplaceMultistream::forward(const TensorView &x,
                                                   TensorView &out) {
  observer_->on_attention_start(idx_);
  if (!ops_.supports_exec_context_iteration()) {
    throw std::runtime_error(
        "SelfAttentionFusedInplaceMultistream: backend does not support exec context iteration");
  }

  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const TensorContracts::BatchSeqDims dims =
      TensorContracts::check_bsd_io(x, out, model_dim,
                                    "SelfAttentionFusedInplaceMultistream");
  const int64_t batch_size = dims.batch_size;
  const int64_t seq_len = dims.seq_len;

  const int64_t num_heads = static_cast<int64_t>(cfg_.model.n_heads);
  TensorContracts::check_attention_config(
      model_dim, num_heads, "SelfAttentionFusedInplaceMultistream");
  const int64_t head_dim = model_dim / num_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const TensorView &compute_Wqkv = params_.qkv.compute_weight;
  const TensorView &bqkv = params_.qkv.bias;
  const TensorView &compute_Wo = params_.out.compute_weight;
  const TensorView &bo = params_.out.bias;

  TensorView qkv = slice_views::batch_seq(
      slices_.qkv, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream qkv");
  ops_.gemm_ranked_matrix_rhs(x, compute_Wqkv, qkv);
  ops_.add_bias_rowwise(qkv, bqkv, qkv);
  positionEncoding_.apply_to_qk(qkv);

  TensorView Q = qkv.subcols(0, model_dim);
  TensorView K = qkv.subcols(model_dim, model_dim);
  TensorView V = qkv.subcols(2 * model_dim, model_dim);
  TensorView context = slice_views::batch_seq(
      slices_.context, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream context");
  ops_.fill(context, 0.0f);
  TensorView cached_weights = slice_views::head_batch_seq_square(
      slices_.weights_cache, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream weights_cache");

  ops_.start_exec_context_iteration();
  for (int64_t h = 0; h < num_heads; ++h) {
    const int64_t col0 = h * head_dim;

    TensorView Qh = Q.subcols(col0, head_dim);
    TensorView Kh = K.subcols(col0, head_dim);
    TensorView Vh = V.subcols(col0, head_dim);
    TensorView context_h = context.subcols(col0, head_dim);
    TensorView weights_h = cached_weights.select(0, h);

    ops_.start_exec_context_group();
    ops_.gemm_batched_rhs_t_exec_context(Qh, Kh, weights_h);
    ops_.scaled_causal_softmax_rows_exec_context(weights_h, scale, weights_h);
    ops_.gemm_batched_exec_context(weights_h, Vh, context_h);
    ops_.finish_exec_context_group();
  }
  ops_.finish_exec_context_iteration();

  ops_.gemm_ranked_matrix_rhs(context, compute_Wo, out);
  ops_.add_bias_rowwise(out, bo, out);

  cache_x_ = x;
  cache_qkv_ = qkv;
  cache_context_ = context;
  observer_->on_attention_end(idx_);
}

void SelfAttentionFusedInplaceMultistream::backward(const TensorView &dout,
                                                    TensorView &dx) {
  observer_->on_attention_start(idx_);
  if (!ops_.supports_exec_context_iteration()) {
    throw std::runtime_error(
        "SelfAttentionFusedInplaceMultistream: backend does not support exec context iteration");
  }
  checker().check_same_shape(dout, cache_x_,
                             "SelfAttentionFusedInplaceMultistream dout");
  checker().check_same_shape(dx, cache_x_,
                             "SelfAttentionFusedInplaceMultistream dx");

  const int64_t model_dim = cache_x_.dim(2);
  const int64_t batch_size = cache_x_.dim(0);
  const int64_t seq_len = cache_x_.dim(1);
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView &compute_Wqkv = params_.qkv.compute_weight;
  const TensorView &compute_Wo = params_.out.compute_weight;
  TensorView &dWo = params_.out.dweight;
  ops_.gemm_ranked_reduce_lhs_t(cache_context_, dout, dWo);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dWo(idx_, dWo);
  TensorView &dbo = params_.out.dbias;
  ops_.row_sum(dout, dbo);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dbo(idx_, dbo);

  TensorView dcontext = slice_views::batch_seq(
      slices_.dcontext, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream dcontext");
  ops_.gemm_ranked_matrix_rhs_t(dout, compute_Wo, dcontext);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dcontext(idx_, dcontext);

  TensorView dqkv = slice_views::batch_seq(
      slices_.dqkv, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream dqkv");
  ops_.fill(dqkv, 0.0f);
  TensorView dQ = dqkv.subcols(0, model_dim);
  TensorView dK = dqkv.subcols(model_dim, model_dim);
  TensorView dV = dqkv.subcols(2 * model_dim, model_dim);

  TensorView Q = cache_qkv_.subcols(0, model_dim);
  TensorView K = cache_qkv_.subcols(model_dim, model_dim);
  TensorView V = cache_qkv_.subcols(2 * model_dim, model_dim);
  TensorView cached_weights = slice_views::head_batch_seq_square(
      slices_.weights_cache, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream weights_cache");

  TensorView dweights = slice_views::batch_seq_square(
      slices_.dweights, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream dweights");
  TensorView dscores = slice_views::batch_seq_square(
      slices_.dscores, batch_size, seq_len,
      "SelfAttentionFusedInplaceMultistream dscores");

  // Keep the backward pass non-aliasing: softmax probabilities, dweights, and
  // dscores must remain distinct while gradients are computed. The forward path
  // still uses multiple streams; backward prioritizes correctness over scratch
  // reuse because overwriting cached attention weights can silently corrupt dQ/dK.
  constexpr int64_t kScratchLanes = 1;
  for (int64_t h0 = 0; h0 < H; h0 += kScratchLanes) {
    const int64_t lane_count = std::min<int64_t>(kScratchLanes, H - h0);

    ops_.start_exec_context_iteration();
    for (int64_t lane = 0; lane < lane_count; ++lane) {
      const int64_t h = h0 + lane;
      const int64_t col0 = h * dh;
      TensorView Qh = Q.subcols(col0, dh);
      TensorView Kh = K.subcols(col0, dh);
      TensorView Vh = V.subcols(col0, dh);
      TensorView dQh = dQ.subcols(col0, dh);
      TensorView dKh = dK.subcols(col0, dh);
      TensorView dVh = dV.subcols(col0, dh);
      TensorView dhead = dcontext.subcols(col0, dh);
      TensorView weights_h = cached_weights.select(0, h);
      TensorView dweights_lane = lane == 0 ? dweights : dscores;

      ops_.start_exec_context_group();
      ops_.gemm_batched_rhs_t_exec_context(dhead, Vh, dweights_lane);
      ops_.gemm_batched_lhs_t_exec_context(weights_h, dhead, dVh);
      ops_.softmax_backward_causal_rows_exec_context(weights_h, dweights_lane,
                                                     dscores);
      ops_.gemm_batched_exec_context(dscores, Kh, dQh);
      ops_.gemm_batched_lhs_t_exec_context(dscores, Qh, dKh);
      ops_.finish_exec_context_group();
    }
    ops_.finish_exec_context_iteration();

    for (int64_t lane = 0; lane < lane_count; ++lane) {
      const int64_t h = h0 + lane;
      const int64_t col0 = h * dh;
      TensorView dQh = dQ.subcols(col0, dh);
      TensorView dKh = dK.subcols(col0, dh);
      TensorView dVh = dV.subcols(col0, dh);
      TensorView weights_h = cached_weights.select(0, h);
      TensorView dweights_lane = lane == 0 ? dweights : dscores;

      if (diagnostics_ != nullptr) diagnostics_->bk_attn_dweights(idx_, h, dweights_lane);
      if (diagnostics_ != nullptr) diagnostics_->bk_attn_dVh(idx_, h, dVh);
      if (diagnostics_ != nullptr) diagnostics_->bk_attn_dscores_softmax_backward(idx_, h, dscores);
      if (diagnostics_ != nullptr) diagnostics_->bk_attn_dscores_masked(idx_, h, dscores);

      ops_.mul_scalar(dQh, scale, dQh);
      if (diagnostics_ != nullptr) diagnostics_->bk_attn_dQh(idx_, h, dQh);

      ops_.mul_scalar(dKh, scale, dKh);
      if (diagnostics_ != nullptr) diagnostics_->bk_attn_dKh(idx_, h, dKh);
    }
  }

  positionEncoding_.backward_qk(dqkv);
  ops_.gemm_ranked_matrix_rhs_t(dqkv, compute_Wqkv, dx);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dx(idx_, dx);

  TensorView &dWqkv = params_.qkv.dweight;
  ops_.gemm_ranked_reduce_lhs_t(cache_x_, dqkv, dWqkv);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dWqkv(idx_, dWqkv);
  TensorView &dbqkv = params_.qkv.dbias;
  ops_.row_sum(dqkv, dbqkv);
  if (diagnostics_ != nullptr) diagnostics_->bk_attn_dbqkv(idx_, dbqkv);
  observer_->on_attention_end(idx_);
}
