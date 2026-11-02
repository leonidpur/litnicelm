#pragma once

#include <cstdint>

inline constexpr uint32_t kBackendTimingMaxOps = 64;

struct BackendTimingTotal {
  uint32_t op_id = 0;
  uint32_t reserved = 0;
  uint64_t count = 0;
  uint64_t total_nanoseconds = 0;
};

struct BackendTimingReport {
  BackendTimingTotal totals[kBackendTimingMaxOps]{};
  uint32_t total_count = 0;
  uint32_t reserved = 0;
  uint64_t marked_span_nanoseconds = 0;
  uint64_t dropped_marks = 0;
};
