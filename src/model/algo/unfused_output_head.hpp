// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "i_output_head.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

class TrainingDiagnosticsController;

// Final LayerNorm, then the LM-head GEMM (no bias), each as its own op.
class UnfusedOutputHead final : public IOutputHead {
public:
  UnfusedOutputHead(const Config &cfg, TensorStore &tensor_store,
                    GradientStore *gradient_store, const Ops &ops);

  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &x, TensorView &logits,
               TensorView *last_hidden) override;
  void backward(const TensorView &dlogits, TensorView &dx) override;

private:
  void check_contract() const;

  const Config &cfg_;
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  const Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_xn_;
  ITrainingObserver *observer_ = &default_training_observer();
};
