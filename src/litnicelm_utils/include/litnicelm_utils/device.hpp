// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <cstdint>

// Where a tensor's memory lives.
enum class Device : uint8_t {
  CPU = 0,
  GPU = 1,
};

inline constexpr const char *device_name(Device d) {
  switch (d) {
  case Device::CPU:
    return "cpu";
  case Device::GPU:
    return "gpu";
  default:
    return "unknown";
  }
}
