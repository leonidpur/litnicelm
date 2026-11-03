// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <stdexcept>

#define REQUIRE_DEBUG(cond, func)                                               \
  if (!(cond)) [[unlikely]] {                                                   \
    throw std::runtime_error(func());                                           \
  }
