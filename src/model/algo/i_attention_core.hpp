// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

class TrainingDiagnosticsController;

// Causal multi-head attention of one layer between the projections; no
// parameters. Q and K arrive already position-encoded.
class IAttentionCore {
public:
  virtual ~IAttentionCore() = default;

  virtual const char *name() const = 0;
  // Null (the default) runs without diagnostics, e.g. in contract tests.
  virtual void set_diagnostics(TrainingDiagnosticsController *diagnostics) = 0;

  // qkv [B, S, 3D] (Q | K | V) -> context [B, S, D], heads side by side.
  // Keeps qkv and whatever its backward needs in its Saved slices.
  virtual void forward(const TensorView &qkv, TensorView &context) = 0;
  // dcontext [B, S, D] -> dqkv [B, S, 3D] w.r.t. the qkv given to forward.
  virtual void backward(const TensorView &dcontext, TensorView &dqkv) = 0;
};
