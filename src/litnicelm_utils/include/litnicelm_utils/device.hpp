#pragma once

#include <cstdint>

// Where a tensor's memory lives.
enum class Device : uint8_t {
  CPU = 0,
  GPU = 1,
};
