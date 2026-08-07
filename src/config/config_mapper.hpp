#pragma once

#include <types.hpp>

#include <vector>

#include "yaml_parser.hpp"

void map_config_entries(const std::vector<YamlEntry> &entries, Config &cfg);
void validate_config(const Config &cfg);
