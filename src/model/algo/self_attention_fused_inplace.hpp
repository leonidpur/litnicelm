#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "i_position_encoding.hpp"
#include "i_self_attention.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

#include <string>

class TrainingDiagnosticsController;

class SelfAttentionFusedInplace final : public ISelfAttention {
public:
  SelfAttentionFusedInplace(int layer_index, const Config &cfg,
                            TensorStore &tensor_store,
                            GradientStore *gradient_store, Ops &ops,
                            IPositionEncoding &position_encoding);
  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

private:
  void validate_contract() const;

  int idx_;
  const Config &cfg_;
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  Ops &ops_;
  IPositionEncoding &positionEncoding_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_qkv_;
  TensorView cache_context_;
  ITrainingObserver *observer_ = &default_training_observer();
};
