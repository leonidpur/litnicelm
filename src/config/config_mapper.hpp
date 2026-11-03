// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <types.hpp>

#include <vector>

#include "yaml_parser.hpp"

void map_config_entries(const std::vector<YamlEntry> &entries, Config &cfg);
Config make_default_config();
void validate_config(const Config &cfg);
