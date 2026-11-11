// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <cstdint>

struct Config;

// Sizes an attention node is laid out for: the largest call (the training
// batch, or one inference window). Without backward, nodes request no
// gradient or backward-only slices.
struct AttentionDims {
  int64_t batch_size = 0;
  int64_t seq_len = 0;
  int64_t model_dim = 0;
  int64_t n_heads = 0;
  bool with_backward = true;

  int64_t head_dim() const { return model_dim / n_heads; }
};

// The training batch with backward, and one inference window without.
AttentionDims training_attention_dims(const Config &cfg);
AttentionDims inference_attention_dims(const Config &cfg);
