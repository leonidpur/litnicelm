// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

class TrainingDiagnosticsController;

// Attention output projection of one layer. Owns the parameters
// layer{i}.attn_out_w [D, D] and layer{i}.attn_out_b [1, D].
class IOutputProjection {
public:
  virtual ~IOutputProjection() = default;

  virtual const char *name() const = 0;
  // Null (the default) runs without diagnostics, e.g. in contract tests.
  virtual void set_diagnostics(TrainingDiagnosticsController *diagnostics) = 0;

  // context [B, S, D] -> out [B, S, D] = context W + b. Keeps context for
  // backward.
  virtual void forward(const TensorView &context, TensorView &out) = 0;
  // dout [B, S, D] -> dcontext [B, S, D]; writes the gradients of W and b.
  virtual void backward(const TensorView &dout, TensorView &dcontext) = 0;
};
