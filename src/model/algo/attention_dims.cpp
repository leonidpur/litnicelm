// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "attention_dims.hpp"

#include <config.hpp>

AttentionDims training_attention_dims(const Config &cfg) {
  return AttentionDims{static_cast<int64_t>(cfg.training.batch_size),
                       static_cast<int64_t>(cfg.training.train_seq_len),
                       static_cast<int64_t>(cfg.model.d_model),
                       static_cast<int64_t>(cfg.model.n_heads), true};
}

AttentionDims inference_attention_dims(const Config &cfg) {
  return AttentionDims{1, static_cast<int64_t>(cfg.model.max_seq_len),
                       static_cast<int64_t>(cfg.model.d_model),
                       static_cast<int64_t>(cfg.model.n_heads), false};
}
