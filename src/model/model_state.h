#pragma once

#include "arena.h"
#include "named_layout.hpp"

struct ModelState {
  Arena *params_arena = nullptr;
  const NamedLayout *params_layout = nullptr;
};
