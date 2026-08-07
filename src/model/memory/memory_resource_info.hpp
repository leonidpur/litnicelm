#pragma once

#include <cstdint>

struct DeviceMemoryInfo {
  bool available = false;
  uint64_t free_bytes = 0;
  uint64_t total_bytes = 0;
};

struct TrainingMemoryUsage {
  uint64_t param_bytes = 0;
  uint64_t grad_bytes = 0;
  uint64_t adam_bytes = 0;
  uint64_t temp_bytes = 0;
  uint64_t total_managed_bytes = 0;
  DeviceMemoryInfo device_before_alloc;
  DeviceMemoryInfo device_after_alloc;
};
