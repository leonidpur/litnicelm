#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

class TrainingDiagnosticsController;

class OutputHead final {
public:
  OutputHead(const Config &cfg, TensorStore &tensor_store,
             GradientStore *gradient_store, Ops &ops);

  void set_observer(ITrainingObserver *observer);
  void set_diagnostics(TrainingDiagnosticsController *diagnostics);

  void forward(const TensorView &x, TensorView &logits,
               TensorView *last_hidden = nullptr);
  void backward(const TensorView &dlogits, TensorView &dx);

private:
  void validate_contract() const;

  const Config &cfg_;
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_xn_;
  ITrainingObserver *observer_ = &default_training_observer();
};
