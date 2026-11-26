// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "i_attention_core.hpp"
#include "kv_cache_attention.hpp"

#include <memory>

class TrainingDiagnosticsController;

// Inference attention core with a KV cache (inference.kv_cache): decorates
// the configured core. Every call appends its K and V rows to the cache; a
// call at first_position 0 (prefill) runs the inner core on the whole
// window, a later one-token call (decode) attends over the cache. Forward
// only.
class KvCachedAttentionCore final : public IAttentionCore {
public:
  KvCachedAttentionCore(std::unique_ptr<IAttentionCore> inner,
                        std::unique_ptr<KvCacheAttention> cache);

  const char *name() const override { return "kv_cached"; }
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;
  void forward(const TensorView &qkv, TensorView &context,
               int64_t first_position) override;
  // Throws: inference only.
  void backward(const TensorView &dcontext, TensorView &dqkv) override;

private:
  std::unique_ptr<IAttentionCore> inner_;
  std::unique_ptr<KvCacheAttention> cache_;
};
