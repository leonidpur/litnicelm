#pragma once

#include <config.hpp>
#include "gradient_factory.hpp"
#include "training_observer.hpp"
#include <report_interface.hpp>
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "transformer_layer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class TrainingDiagnosticsController;

// Decoder-only GPT-style transformer:
//
//  X = tok_embed(ids[B,S]) + pos_embed(0..S-1), broadcast over batch
//  for l in layers: X = layer_l(X)
//  X = LN_f(X)
//  logits = X * lm_head_w            // [B, S, V]
//
// Notes:
// - This is "pure math": no CPU/GPU branching here.
// - Device-specific checks belong inside Ops / TensorFactory.
// - Parameter names assumed from the named parameter layout:
//     tok_embedding  [V, D]
//     pos_embedding  [S, D]
//     lnf_gamma      [1, D]
//     lnf_beta       [1, D]
//     lm_head_w      [D, V]
class Transformer {
public:
  Transformer(const Config &cfg, TensorFactory &tensor_factory,
              GradientFactory *gradient_factory, Ops &ops,
              ReportSink *sink = nullptr);
  void set_observer(ITrainingObserver *observer);
  void set_diagnostics(TrainingDiagnosticsController *diagnostics);

  // ids: [B, S] semantically. Inference may use batch size 1.
  // logits: [B, S, V]
  // last_hidden (optional): receives final normalized hidden [B, S, D].
  void forward(const TensorView &ids, TensorView &logits,
               TensorView *last_hidden = nullptr);
  void backward(const TensorView &ids, const TensorView &dlogits,
                const RuntimeFlags::ProbeFlags &probe);

private:
  void validate_contract() const;

  const Config &cfg_;
  TensorFactory &tensorFactory_;
  GradientFactory *gradientFactory_ = nullptr;
  Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;

  std::vector<TransformerLayer> layers_;
  TensorView cache_x0_;
  TensorView cache_x_last_;
  TensorView cache_xn_;
  bool has_cache_ = false;
  ReportSink *sink_ = nullptr;
  ITrainingObserver *observer_ = &default_training_observer();
};
