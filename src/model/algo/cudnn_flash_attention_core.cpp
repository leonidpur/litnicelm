// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "cudnn_flash_attention_core.hpp"
#include "slice_views.hpp"

#include <cmath>

CudnnFlashAttentionCore::CudnnFlashAttentionCore(int layer_index,
                                                 const Config &cfg,
                                                 const Ops &ops,
                                                 NodeSlices slices)
    : idx_(layer_index),
      cfg_(cfg),
      ops_(ops),
      slices_{slices.find("attn.stats")} {}

std::vector<SliceRequest>
CudnnFlashAttentionCore::slice_requests(const AttentionDims &dims) {
  if (!dims.with_backward) {
    return {};
  }
  return {{"attn.stats", Shape{dims.batch_size, dims.n_heads, dims.seq_len},
           ItemType::Statistics, SliceRole::Saved, SliceLifetime::Step}};
}

void CudnnFlashAttentionCore::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
}

void CudnnFlashAttentionCore::forward(const TensorView &qkv, TensorView &context) {
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = context.dim(2) / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
  TensorView stats;
  if (slices_.stats.data() != nullptr) {
    stats = slice_views::batch_head_seq(slices_.stats, qkv.dim(0), qkv.dim(1),
                                        "CudnnFlashAttentionCore stats");
  }
  ops_.causal_attention_forward(qkv, H, scale, context, stats);
  cache_qkv_ = qkv;
  cache_context_ = context;
  cache_stats_ = stats;
}

void CudnnFlashAttentionCore::backward(const TensorView &dcontext,
                                       TensorView &dqkv) {
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = dcontext.dim(2) / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
  ops_.causal_attention_backward(cache_qkv_, cache_context_, dcontext,
                                 cache_stats_, H, scale, dqkv);
}
