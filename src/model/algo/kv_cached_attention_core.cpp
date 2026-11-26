// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "kv_cached_attention_core.hpp"

#include <stdexcept>
#include <utility>

KvCachedAttentionCore::KvCachedAttentionCore(std::unique_ptr<IAttentionCore> inner,
                                             std::unique_ptr<KvCacheAttention> cache)
    : inner_(std::move(inner)), cache_(std::move(cache)) {}

void KvCachedAttentionCore::set_diagnostics(
    TrainingDiagnosticsController *diagnostics) {
  inner_->set_diagnostics(diagnostics);
}

void KvCachedAttentionCore::forward(const TensorView &qkv, TensorView &context,
                                    int64_t first_position) {
  cache_->append(qkv, first_position);
  if (first_position == 0) {
    inner_->forward(qkv, context, 0);
    return;
  }
  cache_->attend(qkv, first_position, context);
}

void KvCachedAttentionCore::backward(const TensorView &, TensorView &) {
  throw std::runtime_error("KvCachedAttentionCore: inference only (no backward)");
}
