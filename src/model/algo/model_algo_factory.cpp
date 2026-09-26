#include "model_algo_factory.hpp"

#include "ffn.hpp"
#include "fused_bias_relu_ffn.hpp"
#include "inplace_fused_bias_relu_ffn.hpp"
#include "position_encoding.hpp"
#include "self_attention.hpp"
#include "self_attention_fused_inplace.hpp"
#include "self_attention_fused_inplace_multistream.hpp"

#include <stdexcept>

ModelAlgoFactory::ModelAlgoFactory(ModelAlgoConfig cfg) : cfg_(cfg) {}

std::unique_ptr<ISelfAttention> ModelAlgoFactory::create_attention(
    int layer_index, const Config &cfg, TensorStore &tensor_store,
    GradientStore *gradient_store, Ops &ops) const {
  switch (cfg_.attention_impl) {
  case AttentionImplKind::Reference:
    return std::make_unique<SelfAttention>(layer_index, cfg, tensor_store,
                                           gradient_store, ops);
  case AttentionImplKind::FusedInplace:
    return std::make_unique<SelfAttentionFusedInplace>(
        layer_index, cfg, tensor_store, gradient_store, ops);
  case AttentionImplKind::FusedInplaceMultistream:
    return std::make_unique<SelfAttentionFusedInplaceMultistream>(
        layer_index, cfg, tensor_store, gradient_store, ops);
  }
  throw std::runtime_error("ModelAlgoFactory: unknown attention implementation");
}

std::unique_ptr<IFFN> ModelAlgoFactory::create_ffn(
    int layer_index, const Config &cfg, TensorStore &tensor_store,
    GradientStore *gradient_store, Ops &ops) const {
  switch (cfg_.ffn_impl) {
  case FFNImplKind::Reference:
    return std::make_unique<FFN>(layer_index, cfg, tensor_store, gradient_store,
                                 ops);
  case FFNImplKind::FusedBiasRelu:
    return std::make_unique<FusedBiasReluFFN>(
        layer_index, cfg, tensor_store, gradient_store, ops);
  case FFNImplKind::InplaceFusedBiasRelu:
    return std::make_unique<InplaceFusedBiasReluFFN>(
        layer_index, cfg, tensor_store, gradient_store, ops);
  }
  throw std::runtime_error("ModelAlgoFactory: unknown FFN implementation");
}

std::unique_ptr<IPositionEncoding>
ModelAlgoFactory::create_position_encoding(const Config &cfg) const {
  switch (cfg_.position_encoding) {
  case PositionEncodingKind::Learned:
    return std::make_unique<LearnedPositionEncoding>(cfg);
  case PositionEncodingKind::None:
    return std::make_unique<NoPositionEncoding>();
  }
  throw std::runtime_error(
      "ModelAlgoFactory: unknown position encoding implementation");
}
