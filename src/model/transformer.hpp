#pragma once

#include <config.hpp>
#include <report_interface.hpp>
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "transformer_layer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Decoder-only GPT-style transformer:
//
//  X = tok_embed(ids) + pos_embed(0..T-1)
//  for l in layers: X = layer_l(X)
//  X = LN_f(X)
//  logits = X * lm_head_w            // [T, V]
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
  using ParamUpdater =
      std::function<void(const std::string &, TensorView &, const TensorView &, bool)>;

  Transformer(const Config &cfg, TensorFactory &tensors, Ops &ops,
              ReportSink *sink = nullptr);

  // ids: [T] int32/int64 (whatever your TensorFactory/Ops embedding expects)
  // logits: [T, V] (same device/dtype as weights)
  // last_hidden (optional): receives final normalized hidden [T, D].
  void forward(const TensorView &ids, TensorView &logits,
               TensorView *last_hidden = nullptr);
  void backward(const TensorView &ids, const TensorView &dlogits,
                const ParamUpdater &update_param, bool do_probe = false);

private:
  const Config &cfg_;
  TensorFactory &tensorFactory_;
  Ops &ops_;

  std::vector<TransformerLayer> layers_;
  TensorView cache_x0_;
  TensorView cache_x_last_;
  TensorView cache_xn_;
  bool has_cache_ = false;
  ReportSink *sink_ = nullptr;
};
