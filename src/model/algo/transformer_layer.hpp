#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "i_ffn.hpp"
#include "i_self_attention.hpp"
#include "model_algo_factory.hpp"
#include "ops.hpp"
#include "training_observer.hpp"
#include "tensor_store.hpp"

#include <functional>
#include <memory>
#include <string>

class TrainingDiagnosticsController;

// One GPT-style (decoder) transformer block:
//
//   y  = x + SelfAttention(LN1(x))
//   out = y + FFN(LN2(y))
//
// Parameters assumed from the named parameter layout (per layer i):
//   layer{i}.ln1_gamma  [1, D]
//   layer{i}.ln1_beta   [1, D]
//   layer{i}.attn_qkv_w [D, 3D]
//   layer{i}.attn_qkv_b [1, 3D]
//   layer{i}.attn_out_w [D, D]
//   layer{i}.attn_out_b [1, D]
//   layer{i}.ln2_gamma  [1, D]
//   layer{i}.ln2_beta   [1, D]
//   layer{i}.ffn_w1     [D, F]
//   layer{i}.ffn_b1     [1, F]
//   layer{i}.ffn_w2     [F, D]
//   layer{i}.ffn_b2     [1, D]
//
// NOTE: This layer is backend-agnostic and never checks CPU vs GPU.
// Backend/device checks belong in Ops / TensorStore implementations.
class TransformerLayer {
public:
  TransformerLayer(int layer_index, const Config &cfg,
                   TensorStore &tensor_store,
                   GradientStore *gradient_store, Ops &ops,
                   const ModelAlgoFactory &algo_factory);
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
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;

  std::unique_ptr<ISelfAttention> attn_;
  std::unique_ptr<IFFN> ffn_;
  TensorView cache_x_;
  TensorView cache_y_;
  TensorView cache_ln1_;
  TensorView cache_ln2_;
  bool has_cache_ = false;
  ITrainingObserver *observer_ = &default_training_observer();
};
