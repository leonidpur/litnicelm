// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "kv_cache_attention.hpp"

#include <litnicelm_utils/checks.hpp>

#include <cmath>
#include <string>

KvCacheAttention::KvCacheAttention(int layer_index, const Config &cfg,
                                   const Ops &ops, NodeSlices slices)
    : idx_(layer_index),
      cfg_(cfg),
      ops_(ops),
      slices_{slices.at("attn.k_cache"), slices.at("attn.v_cache"),
              slices.at("attn.decode_scores"),
              slices.at("attn.decode_weights")} {}

std::vector<SliceRequest>
KvCacheAttention::slice_requests(const AttentionDims &dims) {
  if (!dims.with_kv_cache) {
    return {};
  }
  const Shape cache{1, dims.seq_len, dims.model_dim};
  const Shape row{1, 1, dims.seq_len};
  return {{"attn.k_cache", cache, ItemType::Activation, SliceRole::Saved,
           SliceLifetime::Session},
          {"attn.v_cache", cache, ItemType::Activation, SliceRole::Saved,
           SliceLifetime::Session},
          {"attn.decode_scores", row, ItemType::Activation,
           SliceRole::Workspace, SliceLifetime::Op},
          {"attn.decode_weights", row, ItemType::Activation,
           SliceRole::Workspace, SliceLifetime::Op}};
}

void KvCacheAttention::append(const TensorView &qkv, int64_t first_position) {
  const Checker &c = checker();
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  c.check_bsd(qkv, 1, -1, 3 * model_dim, "KvCacheAttention qkv");
  const int64_t seq_len = qkv.dim(1);
  c.check(CheckKind::Contract,
          first_position >= 0 && first_position + seq_len <= slices_.k.dim(1),
          "KvCacheAttention", [&] {
            return "layer " + std::to_string(idx_) + ": positions [" +
                   std::to_string(first_position) + ", " +
                   std::to_string(first_position + seq_len) +
                   ") beyond the cache of " + std::to_string(slices_.k.dim(1));
          });
  // Rank-2 [S, D] views: every backend's copy takes strided rows there.
  const TensorView rows = qkv.select(0, 0);
  TensorView k_rows = slices_.k.select(0, 0).subrows(first_position, seq_len);
  TensorView v_rows = slices_.v.select(0, 0).subrows(first_position, seq_len);
  ops_.copy(rows.subcols(model_dim, model_dim), k_rows);
  ops_.copy(rows.subcols(2 * model_dim, model_dim), v_rows);
}

void KvCacheAttention::attend(const TensorView &qkv, int64_t position,
                              TensorView &context) {
  const Checker &c = checker();
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  c.check_bsd(qkv, 1, 1, 3 * model_dim, "KvCacheAttention attend qkv");
  c.check_bsd(context, 1, 1, model_dim, "KvCacheAttention attend context");
  c.check(CheckKind::Contract, position >= 0 && position < slices_.k.dim(1),
          "KvCacheAttention", "position beyond the cache");
  const int64_t keys = position + 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView Q = qkv.subcols(0, model_dim);
  const TensorView K = slices_.k.slice(1, 0, keys);
  const TensorView V = slices_.v.slice(1, 0, keys);
  TensorView scores = slices_.scores.subcols(0, keys);
  TensorView weights = slices_.weights.subcols(0, keys);
  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;
    TensorView context_h = context.subcols(col0, dh);
    ops_.gemm_batched_rhs_t(Q.subcols(col0, dh), K.subcols(col0, dh), scores);
    ops_.mul_scalar(scores, scale, scores);
    ops_.softmax_rows(scores, weights);
    ops_.gemm_batched(weights, V.subcols(col0, dh), context_h);
  }
}
