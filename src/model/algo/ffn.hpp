#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "i_ffn.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
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
class FFN : public IFFN {
public:
  FFN(int layer_index, const Config &cfg, TensorStore &tensor_store,
      GradientStore *gradient_store, Ops &ops);
  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  // x:   [B, S, D]
  // out: [B, S, D]
  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

protected:
  void validate_contract() const;

  int idx_;
  const Config &cfg_;
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_h_;
  TensorView cache_a_;
  ITrainingObserver *observer_ = &default_training_observer();
};
