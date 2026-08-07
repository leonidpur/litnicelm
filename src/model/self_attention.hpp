#pragma once

#include <config.hpp>
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "training_observer.hpp"

#include <functional>
#include <string>

// Multi-head causal self-attention for one transformer layer.
// Uses packed QKV projection:
//   qkv = x * Wqkv + bqkv          where Wqkv: [D, 3D], bqkv: [1, 3D]
//   split qkv -> Q,K,V each [T, D]
//   per-head:
//     scores = Qh * Kh^T / sqrt(dh)   scores: [T, T]
//     causal mask + softmax rows -> weights: [T, T]
//     head = weights * Vh            head: [T, dh]
//   concat heads -> context [T, D]
//   out = context * Wo + bo          Wo: [D, D], bo: [1, D]
class SelfAttention {
public:
  using ParamUpdater =
      std::function<void(const std::string &, TensorView &, const TensorView &, bool)>;

  SelfAttention(int layer_index, const Config &cfg, TensorFactory &tensor_factory,
                Ops &ops);
  void set_observer(ITrainingObserver *observer);

  // x:   [T, D]
  // out: [T, D]
  void forward(const TensorView &x, TensorView &out);
  void backward(const TensorView &dout, TensorView &dx,
                const ParamUpdater &update_param);

private:
  void validate_contract() const;

  int idx_;
  const Config &cfg_;
  TensorFactory &tensorFactory_;
  Ops &ops_;
  TensorView cache_x_;
  TensorView cache_qkv_;
  TensorView cache_context_;
  bool has_cache_ = false;
  ITrainingObserver *observer_ = &default_training_observer();
};
