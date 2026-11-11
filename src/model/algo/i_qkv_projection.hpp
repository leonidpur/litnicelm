// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

class TrainingDiagnosticsController;

// Attention input projection of one layer. Owns the parameters
// layer{i}.attn_qkv_w [D, 3D] and layer{i}.attn_qkv_b [1, 3D].
class IQkvProjection {
public:
  virtual ~IQkvProjection() = default;

  virtual const char *name() const = 0;
  // Null (the default) runs without diagnostics, e.g. in contract tests.
  virtual void set_diagnostics(TrainingDiagnosticsController *diagnostics) = 0;

  // x [B, S, D] -> qkv [B, S, 3D] = x W + b, columns Q | K | V. Keeps x for
  // backward.
  virtual void forward(const TensorView &x, TensorView &qkv) = 0;
  // dqkv [B, S, 3D] (w.r.t. this node's output) -> dx [B, S, D]; writes the
  // gradients of W and b.
  virtual void backward(const TensorView &dqkv, TensorView &dx) = 0;
};
