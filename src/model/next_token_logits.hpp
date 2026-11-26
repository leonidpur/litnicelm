// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>

#include <cstdint>
#include <memory>
#include <vector>

class DeviceBackend;
class TensorStore;
class Transformer;

// The model's next-token logits for a sequence that grows by one token per
// call (generation), as FP32 on the host. The model sees at most the last
// `context` tokens: past it the window slides.
class INextTokenLogits {
public:
  virtual ~INextTokenLogits() = default;
  // ids: the whole sequence so far (prompt plus generated tokens).
  virtual std::vector<float> next_logits(const std::vector<int32_t> &ids) = 0;
};

// Every call runs the model over the last `context` tokens (positions from
// 0). inference.kv_cache: false, and next-token inspection.
class FullWindowLogits final : public INextTokenLogits {
public:
  FullWindowLogits(Transformer &model, TensorStore &tensor_store,
                   DeviceBackend &backend, uint32_t context);
  std::vector<float> next_logits(const std::vector<int32_t> &ids) override;

private:
  Transformer &model_;
  TensorStore &tensorStore_;
  DeviceBackend &backend_;
  const uint32_t context_;
};

// inference.kv_cache: the first call runs the window once (prefill), each
// later call only the new token at its absolute position (decode), which
// attends to the cached K and V. Past the context every call re-prefills the
// last `context` tokens: the same windows as FullWindowLogits.
class KvCacheLogits final : public INextTokenLogits {
public:
  KvCacheLogits(Transformer &model, TensorStore &tensor_store,
                DeviceBackend &backend, uint32_t context);
  std::vector<float> next_logits(const std::vector<int32_t> &ids) override;

private:
  Transformer &model_;
  TensorStore &tensorStore_;
  DeviceBackend &backend_;
  const uint32_t context_;
  // Tokens in the cache: positions [0, cached_) hold ids' first cached_.
  size_t cached_ = 0;
};

// KvCacheLogits under inference.kv_cache, otherwise FullWindowLogits. One per
// generated sequence (the cache belongs to it).
std::unique_ptr<INextTokenLogits>
make_next_token_logits(const Config &cfg, Transformer &model,
                       TensorStore &tensor_store, DeviceBackend &backend,
                       uint32_t context);
