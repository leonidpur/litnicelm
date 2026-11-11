// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "fused_inplace_attention_core.hpp"
#include "slice_views.hpp"
#include "training_diagnostics_controller.hpp"

#include <cmath>

FusedInplaceAttentionCore::FusedInplaceAttentionCore(int layer_index,
                                                     const Config &cfg,
                                                     const Ops &ops,
                                                     NodeSlices slices)
    : idx_(layer_index),
      cfg_(cfg),
      ops_(ops),
      slices_{slices.find("attn.scores"),
               slices.find("attn.weights_cache"),
               slices.find("attn.dweights"),
               slices.find("attn.dscores")} {}

void FusedInplaceAttentionCore::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void FusedInplaceAttentionCore::forward(const TensorView &qkv, TensorView &context) {
  const int64_t batch_size = qkv.dim(0);
  const int64_t seq_len = qkv.dim(1);
  const int64_t model_dim = context.dim(2);
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  TensorView Q = qkv.subcols(0, model_dim);
  TensorView K = qkv.subcols(model_dim, model_dim);
  TensorView V = qkv.subcols(2 * model_dim, model_dim);

  ops_.fill(context, 0.0f);

  TensorView scores = slice_views::batch_seq_square(
      slices_.scores, batch_size, seq_len, "FusedInplaceAttentionCore scores");
  TensorView cached_weights =
      slice_views::head_batch_seq_square(
          slices_.weights_cache, batch_size, seq_len, "FusedInplaceAttentionCore cached_weights");

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;

    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);
    TensorView context_h = context.subcols(col0, dh);
    TensorView weights_h = cached_weights.select(0, h);

    ops_.gemm_batched_rhs_t(Qh, Kh, scores);
    ops_.scaled_causal_softmax_rows(scores, scale, weights_h);
    ops_.gemm_batched(weights_h, Vh, context_h);
  }
  cache_qkv_ = qkv;
}

void FusedInplaceAttentionCore::backward(const TensorView &dcontext, TensorView &dqkv) {
  const int64_t batch_size = cache_qkv_.dim(0);
  const int64_t seq_len = cache_qkv_.dim(1);
  const int64_t model_dim = dcontext.dim(2);
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  ops_.fill(dqkv, 0.0f);
  TensorView dQ = dqkv.subcols(0, model_dim);
  TensorView dK = dqkv.subcols(model_dim, model_dim);
  TensorView dV = dqkv.subcols(2 * model_dim, model_dim);

  TensorView Q = cache_qkv_.subcols(0, model_dim);
  TensorView K = cache_qkv_.subcols(model_dim, model_dim);
  TensorView V = cache_qkv_.subcols(2 * model_dim, model_dim);
  TensorView cached_weights =
      slice_views::head_batch_seq_square(
          slices_.weights_cache, batch_size, seq_len, "FusedInplaceAttentionCore cached_weights");

  TensorView dweights =
      slice_views::batch_seq_square(
          slices_.dweights, batch_size, seq_len, "FusedInplaceAttentionCore dweights");
  TensorView dscores =
      slice_views::batch_seq_square(
          slices_.dscores, batch_size, seq_len, "FusedInplaceAttentionCore dscores");

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;
    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);
    TensorView dQh = dQ.subcols(col0, dh);
    TensorView dKh = dK.subcols(col0, dh);
    TensorView dVh = dV.subcols(col0, dh);
    TensorView dhead = dcontext.subcols(col0, dh);
    TensorView weights_h = cached_weights.select(0, h);

    ops_.gemm_batched_rhs_t(dhead, Vh, dweights);
    if (diagnostics_ != nullptr) diagnostics_->bk_attn_dweights(idx_, h, dweights);
    ops_.gemm_batched_lhs_t(weights_h, dhead, dVh);
    if (diagnostics_ != nullptr) diagnostics_->bk_attn_dVh(idx_, h, dVh);

    ops_.softmax_backward_causal_rows(weights_h, dweights, dscores);
    if (diagnostics_ != nullptr) diagnostics_->bk_attn_dscores_softmax_backward(idx_, h, dscores);
    if (diagnostics_ != nullptr) diagnostics_->bk_attn_dscores_masked(idx_, h, dscores);

    ops_.gemm_batched(dscores, Kh, dQh);
    ops_.mul_scalar(dQh, scale, dQh);
    if (diagnostics_ != nullptr) diagnostics_->bk_attn_dQh(idx_, h, dQh);

    ops_.gemm_batched_lhs_t(dscores, Qh, dKh);
    ops_.mul_scalar(dKh, scale, dKh);
    if (diagnostics_ != nullptr) diagnostics_->bk_attn_dKh(idx_, h, dKh);
  }
}

std::vector<SliceRequest>
FusedInplaceAttentionCore::slice_requests(const AttentionDims &dims) {
  const int64_t B = dims.batch_size;
  const int64_t S = dims.seq_len;
  std::vector<SliceRequest> requests{
      {"attn.scores", Shape{B, S, S}, ItemType::Activation, SliceRole::Workspace,
       SliceLifetime::Op},
      {"attn.weights_cache", Shape{dims.n_heads, B, S, S}, ItemType::Activation,
       SliceRole::Saved, SliceLifetime::Step},
  };
  if (dims.with_backward) {
    requests.push_back({"attn.dweights", Shape{B, S, S}, ItemType::Activation,
                        SliceRole::Workspace, SliceLifetime::Op});
    requests.push_back({"attn.dscores", Shape{B, S, S}, ItemType::Activation,
                        SliceRole::Workspace, SliceLifetime::Op});
  }
  return requests;
}
