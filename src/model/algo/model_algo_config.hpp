#pragma once

#include <config.hpp>

enum class AttentionImplKind {
  Reference,
  FusedInplace,
};

enum class FFNImplKind {
  Reference,
  FusedBiasRelu,
  InplaceFusedBiasRelu,
};

struct ModelAlgoConfig {
  AttentionImplKind attention_impl = AttentionImplKind::Reference;
  FFNImplKind ffn_impl = FFNImplKind::Reference;

  static ModelAlgoConfig from_config(const Config &cfg);
};
