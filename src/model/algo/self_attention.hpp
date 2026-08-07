#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "i_self_attention.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

#include <functional>
#include <string>

class TrainingDiagnosticsController;

// Multi-head causal self-attention for one transformer layer.
// Uses packed QKV projection:
//   qkv = x * Wqkv + bqkv             where Wqkv: [D, 3D], bqkv: [1, 3D]
//   split qkv -> Q,K,V each [B, S, D]
//   per-head and per-batch block:
//     scores_bh = Qbh * Kbh^T / sqrt(dh)   scores_bh: [S, S]
//     causal mask + softmax rows -> weights_bh: [S, S]
//     head_bh = weights_bh * Vbh           head_bh: [S, dh]
//   concat heads -> context [B, S, D]
//   out = context * Wo + bo                Wo: [D, D], bo: [1, D]
class SelfAttention final : public ISelfAttention {
public:
  SelfAttention(int layer_index, const Config &cfg,
                TensorStore &tensor_store,
                GradientStore *gradient_store, Ops &ops);
  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *diagnostics) override;

  // x:   [B, S, D]
  // out: [B, S, D]
  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;

private:
  void validate_contract() const;

  int idx_;
  const Config &cfg_;
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  TensorView cache_x_;
  TensorView cache_qkv_;
  TensorView cache_context_;
  bool has_cache_ = false;
  ITrainingObserver *observer_ = &default_training_observer();
};
