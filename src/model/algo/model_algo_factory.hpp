#pragma once

#include "i_ffn.hpp"
#include "i_position_encoding.hpp"
#include "i_self_attention.hpp"
#include "model_algo_config.hpp"

#include <memory>

class Config;
class GradientStore;
class Ops;
class TensorStore;

class ModelAlgoFactory {
public:
  explicit ModelAlgoFactory(ModelAlgoConfig cfg);

  std::unique_ptr<ISelfAttention>
  create_attention(int layer_index, const Config &cfg, TensorStore &tensor_store,
                   GradientStore *gradient_store, Ops &ops,
                   IPositionEncoding &position_encoding) const;

  std::unique_ptr<IFFN>
  create_ffn(int layer_index, const Config &cfg, TensorStore &tensor_store,
             GradientStore *gradient_store, Ops &ops) const;

  // Position encodings are cheap and their queries depend only on the
  // config, so callers that only need an answer create one on demand.
  std::unique_ptr<IPositionEncoding>
  create_position_encoding(const Config &cfg) const;

  // Memory needs of the chosen implementations.
  // Reference attention keeps softmax weights apart from scores; the fused
  // variants compute them in place.
  bool attention_needs_weights_buffer() const;
  // Non in-place FFNs keep the activation and its gradient in own buffers.
  bool ffn_needs_activation_buffers() const;

private:
  ModelAlgoConfig cfg_;
};
