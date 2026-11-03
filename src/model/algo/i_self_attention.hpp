// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

class ITrainingObserver;
class TrainingDiagnosticsController;

class ISelfAttention {
public:
  virtual ~ISelfAttention() = default;

  virtual void set_observer(ITrainingObserver *observer) = 0;
  virtual void set_diagnostics(TrainingDiagnosticsController *diagnostics) = 0;
  virtual void forward(const TensorView &x, TensorView &out) = 0;
  virtual void backward(const TensorView &dout, TensorView &dx) = 0;
};
