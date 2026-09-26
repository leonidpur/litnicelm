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

  // Created before memory is allocated: memory modules query it for layout.
  std::unique_ptr<IPositionEncoding>
  create_position_encoding(const Config &cfg) const;

private:
  ModelAlgoConfig cfg_;
};
