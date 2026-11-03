// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct YamlEntry {
  std::string key;
  std::string value;
  uint64_t lineno = 0;
};

std::vector<YamlEntry> parse_yaml_file_flat(const std::string &path);
