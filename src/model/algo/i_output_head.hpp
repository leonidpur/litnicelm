// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

class ITrainingObserver;
class TrainingDiagnosticsController;

// The last stage of the model: final hidden states [B, S, D] to logits
// [B, S, V], with the gradient back to the hidden states.
class IOutputHead {
public:
  virtual ~IOutputHead() = default;

  virtual void set_observer(ITrainingObserver *observer) = 0;
  virtual void set_diagnostics(TrainingDiagnosticsController *diagnostics) = 0;
  // last_hidden (optional): receives the final normalized hidden [B, S, D].
  virtual void forward(const TensorView &x, TensorView &logits,
                       TensorView *last_hidden) = 0;
  virtual void backward(const TensorView &dlogits, TensorView &dx) = 0;
};
