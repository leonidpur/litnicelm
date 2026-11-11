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

// The backend's fused causal attention (kFusedOpCausalAttentionBf16Act,
// cuDNN flash attention in the CUDA plugin): no score or weight matrices,
// only each row's softmax log-sum-exp is kept for backward. BF16
// activations only; the factory checks precision and backend at setup.
// Writes no per-head diagnostics (there are no intermediate weights).
class CudnnFlashAttentionCore final : public IAttentionCore {
public:
  CudnnFlashAttentionCore(int layer_index, const Config &cfg, const Ops &ops,
                          NodeSlices slices);

  static std::vector<SliceRequest> slice_requests(const AttentionDims &dims);

  const char *name() const override { return "cudnn_flash"; }
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &qkv, TensorView &context) override;
  void backward(const TensorView &dcontext, TensorView &dqkv) override;

private:
  int idx_;
  const Config &cfg_;
  const Ops &ops_;
  // Full extent; empty in an inference layout.
  struct Slices {
    TensorView stats;  // [B, H, S] F32
  };
  Slices slices_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_qkv_;
  TensorView cache_context_;
  TensorView cache_stats_;
};
