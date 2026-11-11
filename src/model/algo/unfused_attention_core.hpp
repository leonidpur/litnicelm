// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "attention_dims.hpp"
#include "i_attention_core.hpp"
#include "ops.hpp"
#include "slice_request.hpp"

#include <vector>

class TrainingDiagnosticsController;

// Per-head GEMMs with scale, causal mask and softmax as separate ops; the
// weights go through their own buffer before the per-head cache.
class UnfusedAttentionCore final : public IAttentionCore {
public:
  UnfusedAttentionCore(int layer_index, const Config &cfg, const Ops &ops,
                         NodeSlices slices);

  static std::vector<SliceRequest> slice_requests(const AttentionDims &dims);

  const char *name() const override { return "unfused"; }
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &qkv, TensorView &context) override;
  void backward(const TensorView &dcontext, TensorView &dqkv) override;

private:
  int idx_;
  const Config &cfg_;
  const Ops &ops_;
  // This node's slices, full extent; empty for backward-only slices of an
  // inference layout.
  struct Slices {
    TensorView scores;
    TensorView weights_cache;
    TensorView dweights;
    TensorView dscores;
    TensorView weights;
  };

  Slices slices_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_qkv_;
};
