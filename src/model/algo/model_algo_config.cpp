// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "model_algo_config.hpp"
#include "layer_composition.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace {
template <typename Kind>
using NameTable = std::vector<std::pair<std::string, Kind>>;

const NameTable<AttentionImplKind> &attention_table() {
  static const NameTable<AttentionImplKind> table{
      {"unfused", AttentionImplKind::Unfused},
      {"fused_inplace", AttentionImplKind::FusedInplace},
      {"fused_inplace_multistream", AttentionImplKind::FusedInplaceMultistream},
      {"cudnn_flash", AttentionImplKind::CudnnFlash},
  };
  return table;
}

const NameTable<FFNImplKind> &ffn_table() {
  static const NameTable<FFNImplKind> table{
      {"reference", FFNImplKind::Reference},
      {"fused_bias_relu", FFNImplKind::FusedBiasRelu},
      {"inplace_fused_bias_relu", FFNImplKind::InplaceFusedBiasRelu},
      {"fused_linear", FFNImplKind::FusedLinear},
  };
  return table;
}

const NameTable<ProjectionKind> &projection_table() {
  static const NameTable<ProjectionKind> table{
      {"gemm", ProjectionKind::Gemm},
      {"fused_linear", ProjectionKind::FusedLinear},
  };
  return table;
}

const NameTable<OutputHeadKind> &output_head_table() {
  static const NameTable<OutputHeadKind> table{
      {"unfused", OutputHeadKind::Unfused},
  };
  return table;
}

const NameTable<PositionEncodingKind> &position_encoding_table() {
  static const NameTable<PositionEncodingKind> table{
      {"learned", PositionEncodingKind::Learned},
      {"none", PositionEncodingKind::None},
      {"rope", PositionEncodingKind::Rope},
  };
  return table;
}

template <typename Kind>
Kind parse(const NameTable<Kind> &table, const std::string &value,
           const char *key) {
  for (const auto &[name, kind] : table) {
    if (name == value) {
      return kind;
    }
  }
  throw std::runtime_error(std::string("ModelAlgoConfig: unsupported ") + key +
                           ": " + value);
}

template <typename Kind>
std::vector<std::string> names_of(const NameTable<Kind> &table) {
  std::vector<std::string> names;
  for (const auto &entry : table) {
    names.push_back(entry.first);
  }
  return names;
}

template <typename Kind>
const std::string &name_of(const NameTable<Kind> &table, Kind kind) {
  for (const auto &[name, entry] : table) {
    if (entry == kind) {
      return name;
    }
  }
  throw std::runtime_error("ModelAlgoConfig: unnamed variant");
}
} // namespace

AttentionImplKind ModelAlgoConfig::attention_of(AttentionCoreKind core) {
  switch (core) {
  case AttentionCoreKind::Unfused:
    return AttentionImplKind::Unfused;
  case AttentionCoreKind::FusedInplace:
    return AttentionImplKind::FusedInplace;
  case AttentionCoreKind::CudnnFlash:
    return AttentionImplKind::CudnnFlash;
  }
  throw std::runtime_error("ModelAlgoConfig: unknown attention core");
}

ModelAlgoConfig ModelAlgoConfig::from_config(const Config &cfg) {
  ModelAlgoConfig out;
  out.position_encoding = parse(position_encoding_table(),
                                cfg.model_algo.position_encoding,
                                "model_algo.position_encoding");
  out.debug_guards = cfg.model_algo.debug_guards;
  if (cfg.model_algo.preset.empty()) {
    out.attention_impl = parse(attention_table(), cfg.model_algo.attention,
                               "model_algo.attention");
    out.ffn_impl = parse(ffn_table(), cfg.model_algo.ffn, "model_algo.ffn");
    out.projections = parse(projection_table(), cfg.model_algo.projections,
                            "model_algo.projections");
    out.output_head = parse(output_head_table(), cfg.model_algo.output_head,
                            "model_algo.output_head");
    return out;
  }
  const ModelAlgoRuntimeConfig defaults;
  if (cfg.model_algo.attention != defaults.attention ||
      cfg.model_algo.ffn != defaults.ffn ||
      cfg.model_algo.projections != defaults.projections ||
      cfg.model_algo.output_head != defaults.output_head) {
    throw std::runtime_error(
        "ModelAlgoConfig: model_algo.preset " + cfg.model_algo.preset +
        " chooses attention, projections, ffn and output_head; remove"
        " model_algo.attention (" +
        cfg.model_algo.attention + "), model_algo.projections (" +
        cfg.model_algo.projections + "), model_algo.ffn (" +
        cfg.model_algo.ffn + ") and model_algo.output_head (" +
        cfg.model_algo.output_head + ")");
  }
  const LayerComposition composition =
      LayerComposition::preset(cfg.model_algo.preset);
  out.attention_impl = attention_of(composition.core);
  out.ffn_impl = composition.ffn;
  out.projections = composition.projections;
  out.output_head = composition.output_head;
  return out;
}

const std::vector<std::string> &ModelAlgoConfig::attention_names() {
  static const std::vector<std::string> names = names_of(attention_table());
  return names;
}

bool ModelAlgoConfig::attention_is_experimental(const std::string &name) {
  return name == "fused_inplace_multistream";
}

std::string ModelAlgoConfig::attention_precision(const std::string &name) {
  return name == "cudnn_flash" ? "bf16_mixed" : "";
}

const std::vector<std::string> &ModelAlgoConfig::ffn_names() {
  static const std::vector<std::string> names = names_of(ffn_table());
  return names;
}

const std::vector<std::string> &ModelAlgoConfig::position_encoding_names() {
  static const std::vector<std::string> names =
      names_of(position_encoding_table());
  return names;
}

const std::string &ModelAlgoConfig::attention_name(AttentionImplKind kind) {
  return name_of(attention_table(), kind);
}

const std::string &ModelAlgoConfig::ffn_name(FFNImplKind kind) {
  return name_of(ffn_table(), kind);
}

const std::vector<std::string> &ModelAlgoConfig::projection_names() {
  static const std::vector<std::string> names = names_of(projection_table());
  return names;
}

const std::string &ModelAlgoConfig::projection_name(ProjectionKind kind) {
  return name_of(projection_table(), kind);
}

const std::vector<std::string> &ModelAlgoConfig::output_head_names() {
  static const std::vector<std::string> names = names_of(output_head_table());
  return names;
}

const std::string &ModelAlgoConfig::output_head_name(OutputHeadKind kind) {
  return name_of(output_head_table(), kind);
}
