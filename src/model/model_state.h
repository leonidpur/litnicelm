// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "arena.h"
#include "named_layout.hpp"

struct ModelState {
  Arena *params_arena = nullptr;
  const NamedLayout *params_layout = nullptr;
};
