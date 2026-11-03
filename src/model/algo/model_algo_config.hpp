// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

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
  Rope,
};

struct ModelAlgoConfig {
  AttentionImplKind attention_impl = AttentionImplKind::Reference;
  FFNImplKind ffn_impl = FFNImplKind::Reference;
  PositionEncodingKind position_encoding = PositionEncodingKind::Learned;

  static ModelAlgoConfig from_config(const Config &cfg);
};
