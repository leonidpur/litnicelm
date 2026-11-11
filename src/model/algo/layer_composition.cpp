// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "layer_composition.hpp"

#include <stdexcept>

LayerComposition LayerComposition::preset(const std::string &name) {
  if (name == "unfused") {
    return LayerComposition{ProjectionKind::Gemm,
                            AttentionCoreKind::Unfused,
                            FFNImplKind::Reference, OutputHeadKind::Unfused};
  }
  if (name == "portable_fused") {
    return LayerComposition{ProjectionKind::FusedLinear,
                            AttentionCoreKind::FusedInplace,
                            FFNImplKind::FusedLinear,
                            OutputHeadKind::Unfused};
  }
  if (name == "max_fused") {
    return LayerComposition{ProjectionKind::FusedLinear,
                            AttentionCoreKind::CudnnFlash,
                            FFNImplKind::FusedLinear,
                            OutputHeadKind::Unfused};
  }
  throw std::runtime_error("LayerComposition: unknown preset " + name +
                           " (unfused, portable_fused, max_fused)");
}

const std::vector<std::string> &LayerComposition::preset_names() {
  static const std::vector<std::string> names{"unfused", "portable_fused",
                                              "max_fused"};
  return names;
}
