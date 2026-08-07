#include "model_algo_config.hpp"

#include <stdexcept>
#include <string>

namespace {
AttentionImplKind parse_attention_impl(const std::string &value) {
  if (value == "reference") {
    return AttentionImplKind::Reference;
  }
  if (value == "fused_inplace") {
    return AttentionImplKind::FusedInplace;
  }
  if (value == "fused_inplace_multistream") {
    return AttentionImplKind::FusedInplaceMultistream;
  }
  throw std::runtime_error("ModelAlgoConfig: unsupported model_algo.attention: " +
                           value);
}

FFNImplKind parse_ffn_impl(const std::string &value) {
  if (value == "reference") {
    return FFNImplKind::Reference;
  }
  if (value == "fused_bias_relu") {
    return FFNImplKind::FusedBiasRelu;
  }
  if (value == "inplace_fused_bias_relu") {
    return FFNImplKind::InplaceFusedBiasRelu;
  }
  throw std::runtime_error("ModelAlgoConfig: unsupported model_algo.ffn: " +
                           value);
}
} // namespace

ModelAlgoConfig ModelAlgoConfig::from_config(const Config &cfg) {
  ModelAlgoConfig out;
  out.attention_impl = parse_attention_impl(cfg.model_algo.attention);
  out.ffn_impl = parse_ffn_impl(cfg.model_algo.ffn);
  return out;
}
