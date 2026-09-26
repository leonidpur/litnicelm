#pragma once

#include <config.hpp>

enum class AttentionImplKind {
  Reference,
  FusedInplace,
  FusedInplaceMultistream,
};

enum class FFNImplKind {
  Reference,
  FusedBiasRelu,
  InplaceFusedBiasRelu,
};

enum class PositionEncodingKind {
  Learned,
  None,
};

struct ModelAlgoConfig {
  AttentionImplKind attention_impl = AttentionImplKind::Reference;
  FFNImplKind ffn_impl = FFNImplKind::Reference;
  PositionEncodingKind position_encoding = PositionEncodingKind::Learned;

  static ModelAlgoConfig from_config(const Config &cfg);
};
