#pragma once

#include <stdexcept>

#define REQUIRE_DEBUG(cond, func)                                               \
  if (!(cond)) [[unlikely]] {                                                   \
    throw std::runtime_error(func());                                           \
  }
