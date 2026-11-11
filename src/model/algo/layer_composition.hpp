// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "model_algo_config.hpp"

#include <string>
#include <vector>

// Variants per attention slot. One flat level: a variant never derives from
// another.
enum class AttentionCoreKind {
  Unfused,       // per-head GEMMs; scale, mask, softmax and weight copy apart
  FusedInplace,  // per-head GEMMs; fused scaled causal softmax into the cache
  CudnnFlash,    // the backend's fused causal attention (cuDNN); bf16_mixed
};

// The variant in every slot of the model (each transformer layer's and the
// output head), chosen by the factory from config. The named presets are the tested combinations.
struct LayerComposition {
  ProjectionKind projections = ProjectionKind::Gemm;  // QKV and output
  AttentionCoreKind core = AttentionCoreKind::Unfused;
  FFNImplKind ffn = FFNImplKind::Reference;
  OutputHeadKind output_head = OutputHeadKind::Unfused;

  // "unfused", "portable_fused" or "max_fused"; throws on any other name.
  // portable_fused and max_fused use fused_linear projections and FFN (on every
  // backend; Ops composes them where the backend has no kFusedOpLinear).
  // max_fused also uses the backend's fused attention, so it needs
  // bf16_mixed and a backend with kFusedOpCausalAttentionBf16Act.
  static LayerComposition preset(const std::string &name);
  static const std::vector<std::string> &preset_names();
};
