#pragma once

#include <config.hpp>
#include "gradient_factory.hpp"
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "training_observer.hpp"

#include <functional>
#include <string>

class TrainingDiagnosticsController;

// Feed-Forward Network for one transformer layer:
// out = W2 * relu(W1*x + b1) + b2   (applied over [B, S, D] logical activations)
//
// Parameter naming assumed from the named parameter layout:
//   layer{idx}.ffn_w1 : [D, F]
//   layer{idx}.ffn_b1 : [1, F]
//   layer{idx}.ffn_w2 : [F, D]
//   layer{idx}.ffn_b2 : [1, D]
class FFN {
public:
  FFN(int layer_index, const Config &cfg, TensorFactory &tensor_factory,
      GradientFactory *gradient_factory, Ops &ops);
  void set_observer(ITrainingObserver *observer);
  void set_diagnostics(TrainingDiagnosticsController *diagnostics);

  // x:   [B, S, D]
  // out: [B, S, D]
  void forward(const TensorView &x, TensorView &out);
  void backward(const TensorView &dout, TensorView &dx);

private:
  void validate_contract() const;

  int idx_;
  const Config &cfg_;
  TensorFactory &tensorFactory_;
  GradientFactory *gradientFactory_ = nullptr;
  Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_h_;
  TensorView cache_a_;
  bool has_cache_ = false;
  ITrainingObserver *observer_ = &default_training_observer();
};
