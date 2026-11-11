// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <string>

// The registered variants and presets, one per line, for people and tools
// (vs_bench reads the "attention" lines):
//   <config key> <name> [experimental]
//   slot <slot> <variant>
//   preset <name> attention=<name> projections=<name> ffn=<name>
//          output_head=<name> [precision=<native precision>]
std::string render_variant_catalog();
