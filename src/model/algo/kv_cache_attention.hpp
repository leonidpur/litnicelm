// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "attention_dims.hpp"
#include "ops.hpp"
#include "slice_request.hpp"

#include <cstdint>
#include <vector>

// One layer's KV cache for decoding: K and V [1, max_seq_len, D] (heads side
// by side, as in qkv) of every position processed so far, kept across
// forward calls (SliceLifetime::Session), and the attention of one new token
// over them. Composed from existing Ops (batched GEMMs and a row softmax) in
// both precision variants; positions are absolute, the caller tracks them.
class KvCacheAttention {
public:
  KvCacheAttention(int layer_index, const Config &cfg, const Ops &ops,
                   NodeSlices slices);

  // Nothing unless dims.with_kv_cache.
  static std::vector<SliceRequest> slice_requests(const AttentionDims &dims);

  // qkv [1, S, 3D], Q and K already position-encoded, rows at positions
  // [first_position, first_position + S): stores their K and V rows.
  void append(const TensorView &qkv, int64_t first_position);
  // qkv [1, 1, 3D] of the token at `position`, already appended ->
  // context [1, 1, D] = softmax(scale Q K^T) V per head over positions
  // [0, position].
  void attend(const TensorView &qkv, int64_t position, TensorView &context);

private:
  int idx_;
  const Config &cfg_;
  const Ops &ops_;
  struct Slices {
    TensorView k;        // [1, max_seq_len, D], Session
    TensorView v;        // [1, max_seq_len, D], Session
    TensorView scores;   // [1, 1, max_seq_len], Op
    TensorView weights;  // [1, 1, max_seq_len], Op
  };
  Slices slices_;
};
