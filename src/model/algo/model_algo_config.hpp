// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>

#include <string>
#include <vector>

enum class AttentionImplKind {
  Unfused,
  FusedInplace,
  FusedInplaceMultistream,
  CudnnFlash,
};

enum class FFNImplKind {
  Reference,
  FusedBiasRelu,
  InplaceFusedBiasRelu,
  FusedLinear,
};

// The variant in both attention projection slots (QKV and output).
enum class ProjectionKind {
  Gemm,         // GEMM, then the bias as its own op
  FusedLinear,  // Ops::linear_forward / linear_backward_params
};

// The variant in the output-head slot (final norm and LM head).
enum class OutputHeadKind {
  Unfused,  // LayerNorm, then the LM-head GEMM, as separate ops
};

enum class PositionEncodingKind {
  Learned,
  None,
  Rope,
};

enum class AttentionCoreKind;

struct ModelAlgoConfig {
  AttentionImplKind attention_impl = AttentionImplKind::Unfused;
  FFNImplKind ffn_impl = FFNImplKind::Reference;
  PositionEncodingKind position_encoding = PositionEncodingKind::Learned;
  ProjectionKind projections = ProjectionKind::Gemm;
  OutputHeadKind output_head = OutputHeadKind::Unfused;
  // Wrap attention in GuardedSelfAttention (model_algo.debug_guards).
  bool debug_guards = false;

  // model_algo.preset, when set, chooses attention, projections, FFN and
  // output head; it then requires model_algo.attention,
  // model_algo.projections, model_algo.ffn and model_algo.output_head at
  // their defaults.
  static ModelAlgoConfig from_config(const Config &cfg);

  // The config names, in listing order; parsing accepts exactly these.
  static const std::vector<std::string> &attention_names();
  // Attention names outside the presets and tier-1 tests.
  static bool attention_is_experimental(const std::string &name);
  // The precision an attention name is restricted to ("bf16_mixed"), or ""
  // when it runs in every precision.
  static std::string attention_precision(const std::string &name);
  static const std::vector<std::string> &ffn_names();
  static const std::vector<std::string> &position_encoding_names();
  static const std::vector<std::string> &projection_names();
  static const std::vector<std::string> &output_head_names();
  static const std::string &attention_name(AttentionImplKind kind);
  static const std::string &ffn_name(FFNImplKind kind);
  static const std::string &projection_name(ProjectionKind kind);
  static const std::string &output_head_name(OutputHeadKind kind);
  // The attention a composition's core corresponds to.
  static AttentionImplKind attention_of(AttentionCoreKind core);
};
