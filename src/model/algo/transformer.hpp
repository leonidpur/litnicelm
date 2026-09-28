#pragma once

#include <config.hpp>
#include "gradient_store.hpp"
#include "i_position_encoding.hpp"
#include "model_algo_config.hpp"
#include "model_algo_factory.hpp"
#include "output_head.hpp"
#include "training_observer.hpp"
#include <report_interface.hpp>
#include "ops.hpp"
#include "tensor_store.hpp"
#include "transformer_layer.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class TrainingDiagnosticsController;

// Decoder-only GPT-style transformer:
//
//  X = tok_embed(ids[B,S]); position_encoding.apply_to_input(X)
//  for l in layers: X = layer_l(X)
//  X = LN_f(X)
//  logits = X * lm_head_w            // [B, S, V]
//
// Notes:
// - This is "pure math": no CPU/GPU branching here.
// - Device-specific checks belong inside Ops / TensorStore.
// - Parameter names assumed from the named parameter layout:
//     tok_embedding  [V, D]
//     lnf_gamma      [1, D]
//     lnf_beta       [1, D]
//     lm_head_w      [D, V]
class Transformer {
public:
  // Creates its attention, FFN and position-encoding objects with algo.
  Transformer(const Config &cfg, const ModelAlgoFactory &algo,
              TensorStore &tensor_store, GradientStore *gradient_store,
              const Ops &ops, ReportSink *sink = nullptr);
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
  // Declared before layers_, which hold references to it.
  std::unique_ptr<IPositionEncoding> positionEncoding_;
  TensorStore &tensorStore_;
  GradientStore *gradientStore_ = nullptr;
  const Ops &ops_;
  TrainingDiagnosticsController *diagnostics_ = nullptr;
  OutputHead outputHead_;

  std::vector<TransformerLayer> layers_;
  TensorView cache_x0_;
  ReportSink *sink_ = nullptr;
  ITrainingObserver *observer_ = &default_training_observer();
};
