// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "i_qkv_projection.hpp"
#include "linear_params.hpp"
#include "ops.hpp"

class TrainingDiagnosticsController;

// QKV projection as one GEMM against the compute weight, then the bias.
class GemmQkvProjection final : public IQkvProjection {
public:
  GemmQkvProjection(int layer_index, const LinearParams &params, const Ops &ops);

  const char *name() const override { return "gemm"; }
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
