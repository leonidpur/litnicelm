// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "i_output_projection.hpp"
#include "linear_params.hpp"
#include "ops.hpp"

class TrainingDiagnosticsController;

// Output projection as one GEMM against the compute weight, then the bias.
class GemmOutputProjection final : public IOutputProjection {
public:
  GemmOutputProjection(int layer_index, const LinearParams &params, const Ops &ops);

  const char *name() const override { return "gemm"; }
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &context, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dcontext) override;

private:
  int idx_;
  LinearParams params_;
  const Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_context_;
};
