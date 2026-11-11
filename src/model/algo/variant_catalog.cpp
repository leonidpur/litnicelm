// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "variant_catalog.hpp"
#include "layer_composition.hpp"
#include "model_algo_config.hpp"

#include <sstream>

std::string render_variant_catalog() {
  std::ostringstream out;
  for (const std::string &name : ModelAlgoConfig::attention_names()) {
    out << "attention " << name
        << (ModelAlgoConfig::attention_is_experimental(name) ? " experimental" : "");
    const std::string precision = ModelAlgoConfig::attention_precision(name);
    if (!precision.empty()) {
      out << " precision=" << precision;
    }
    out << "\n";
  }
  for (const std::string &name : ModelAlgoConfig::ffn_names()) {
    out << "ffn " << name << "\n";
  }
  for (const std::string &name : ModelAlgoConfig::position_encoding_names()) {
    out << "position_encoding " << name << "\n";
  }
  for (const std::string &name : ModelAlgoConfig::projection_names()) {
    out << "projections " << name << "\n";
  }
  for (const std::string &name : ModelAlgoConfig::output_head_names()) {
    out << "output_head " << name << "\n";
  }
  out << "slot attention.qkv_projection gemm\n"
      << "slot attention.qkv_projection fused_linear\n"
      << "slot attention.core unfused\n"
      << "slot attention.core fused_inplace\n"
      << "slot attention.core cudnn_flash\n"
      << "slot attention.output_projection gemm\n"
      << "slot attention.output_projection fused_linear\n";
  for (const std::string &name : LayerComposition::preset_names()) {
    const LayerComposition composition = LayerComposition::preset(name);
    const std::string &attention =
        ModelAlgoConfig::attention_name(ModelAlgoConfig::attention_of(composition.core));
    out << "preset " << name << " attention=" << attention
        << " projections=" << ModelAlgoConfig::projection_name(composition.projections)
        << " ffn=" << ModelAlgoConfig::ffn_name(composition.ffn)
        << " output_head="
        << ModelAlgoConfig::output_head_name(composition.output_head);
    const std::string precision = ModelAlgoConfig::attention_precision(attention);
    if (!precision.empty()) {
      out << " precision=" << precision;
    }
    out << "\n";
  }
  return out.str();
}
