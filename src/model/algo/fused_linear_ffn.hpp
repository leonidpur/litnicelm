// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "i_ffn.hpp"
#include "linear_params.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

class TrainingDiagnosticsController;

// out = relu(x W1 + b1) W2 + b2 with both biases and the ReLU in the GEMM
// epilogues (Ops::linear_forward) and each layer's weight and bias
// gradients from one GEMM (Ops::linear_backward_params). Keeps the
// post-ReLU h for backward and computes the ReLU gradient in place, like
// InplaceFusedBiasReluFFN; uses the same temp slices (ffn.h, ffn.da).
class FusedLinearFFN final : public IFFN {
public:
  FusedLinearFFN(int layer_index, const Config &cfg, TensorStore &tensor_store,
                 const FfnParams &params, const Ops &ops);
  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

private:
  int idx_;
  const Config &cfg_;
  TensorStore &tensorStore_;
  FfnParams params_;
  const Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_h_;
  ITrainingObserver *observer_ = &default_training_observer();
};
