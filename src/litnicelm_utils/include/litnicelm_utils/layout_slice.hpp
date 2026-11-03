// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "litnicelm_utils/dtype.hpp"

#include <cstdint>
#include <string>

// One named tensor's place in an arena: byte offset, size and element type.
struct LayoutSlice {
  std::string name;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  DType dtype = DType::F32;
};
