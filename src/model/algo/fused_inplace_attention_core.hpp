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

// Per-head GEMMs with one fused scaled causal softmax that writes the
// weights straight into the per-head cache.
class FusedInplaceAttentionCore final : public IAttentionCore {
public:
  FusedInplaceAttentionCore(int layer_index, const Config &cfg, const Ops &ops,
                            NodeSlices slices);

  static std::vector<SliceRequest> slice_requests(const AttentionDims &dims);

  const char *name() const override { return "fused_inplace"; }
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
  };

  Slices slices_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_qkv_;
};
