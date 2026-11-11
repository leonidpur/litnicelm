// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "attention_dims.hpp"
#include "i_ffn.hpp"
#include "i_output_head.hpp"
#include "i_output_projection.hpp"
#include "i_qkv_projection.hpp"
#include "linear_params.hpp"
#include "i_position_encoding.hpp"
#include "i_self_attention.hpp"
#include "model_algo_config.hpp"
#include "slice_request.hpp"

#include <memory>
#include <vector>

class Config;
class GradientStore;
class Ops;
class TensorStore;

class ModelAlgoFactory {
public:
  explicit ModelAlgoFactory(ModelAlgoConfig cfg);

  std::unique_ptr<ISelfAttention>
  create_attention(int layer_index, const Config &cfg, TensorStore &tensor_store,
                   GradientStore *gradient_store, const Ops &ops,
                   IPositionEncoding &position_encoding) const;

  std::unique_ptr<IFFN>
  create_ffn(int layer_index, const Config &cfg, TensorStore &tensor_store,
             GradientStore *gradient_store, const Ops &ops) const;

  std::unique_ptr<IOutputHead>
  create_output_head(const Config &cfg, TensorStore &tensor_store,
                     GradientStore *gradient_store, const Ops &ops) const;

  // Position encodings are cheap and their queries depend only on the
  // config, so callers that only need an answer create one on demand.
  std::unique_ptr<IPositionEncoding>
  create_position_encoding(const Config &cfg) const;

  // Memory needs of the chosen implementations.
  // The slices one layer's attention requests: the composite's and its
  // core's, or multistream's; each followed by a guard slice under
  // model_algo.debug_guards.
  std::vector<SliceRequest> attention_slice_requests(const AttentionDims &dims) const;
  // Non in-place FFNs keep the activation and its gradient in own buffers.
  bool ffn_needs_activation_buffers() const;

private:
  std::vector<SliceRequest>
  unguarded_attention_slice_requests(const AttentionDims &dims) const;
  std::unique_ptr<IQkvProjection> make_qkv_projection(int layer_index,
                                                      const LinearParams &params,
                                                      const Ops &ops) const;
  std::unique_ptr<IOutputProjection>
  make_output_projection(int layer_index, const LinearParams &params,
                         const Ops &ops) const;

  ModelAlgoConfig cfg_;
};
