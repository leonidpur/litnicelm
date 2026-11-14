// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "i_qkv_projection.hpp"
#include "linear_params.hpp"
#include "ops.hpp"

class TrainingDiagnosticsController;

// QKV projection as Ops::linear_forward (GEMM with the bias in its epilogue) and
// Ops::linear_backward_params_accumulate (weight and bias gradients in one GEMM).
class FusedLinearQkvProjection final : public IQkvProjection {
public:
  FusedLinearQkvProjection(int layer_index, const LinearParams &params, const Ops &ops);

  const char *name() const override { return "fused_linear"; }
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &x, TensorView &qkv) override;
  void backward(const TensorView &dqkv, TensorView &dx) override;

private:
  int idx_;
  LinearParams params_;
  const Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
};
