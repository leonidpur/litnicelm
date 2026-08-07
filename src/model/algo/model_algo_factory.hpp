#pragma once

#include "i_ffn.hpp"
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
                   GradientStore *gradient_store, Ops &ops) const;

  std::unique_ptr<IFFN>
  create_ffn(int layer_index, const Config &cfg, TensorStore &tensor_store,
             GradientStore *gradient_store, Ops &ops) const;

private:
  ModelAlgoConfig cfg_;
};
