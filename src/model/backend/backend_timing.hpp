#pragma once

#include <array>
#include <chrono>
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

// Allocation-free wall-clock implementation used by host backends. CUDA
// overrides the DeviceBackend timing methods with stream event markers.
class HostBackendTimer {
public:
  void mark_begin(uint32_t op_id) {
    if (active_) {
      ++dropped_marks_;
      return;
    }
    active_ = true;
    active_op_id_ = op_id;
    active_start_ = Clock::now();
    if (!has_span_) {
      span_start_ = active_start_;
      has_span_ = true;
    }
  }

  void mark_end() {
    if (!active_) {
      ++dropped_marks_;
      return;
    }
    const TimePoint now = Clock::now();
    if (active_op_id_ < totals_.size()) {
      Total &total = totals_[active_op_id_];
      ++total.count;
      total.nanoseconds += static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              now - active_start_).count());
    } else {
      ++dropped_marks_;
    }
    span_end_ = now;
    active_ = false;
  }

  BackendTimingReport report() const {
    BackendTimingReport result{};
    for (uint32_t op_id = 0; op_id < totals_.size(); ++op_id) {
      const Total &source = totals_[op_id];
      if (source.count == 0 || result.total_count == kBackendTimingMaxOps) {
        continue;
      }
      BackendTimingTotal &target = result.totals[result.total_count++];
      target.op_id = op_id;
      target.count = source.count;
      target.total_nanoseconds = source.nanoseconds;
    }
    if (has_span_) {
      result.marked_span_nanoseconds = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              span_end_ - span_start_).count());
    }
    result.dropped_marks = dropped_marks_;
    return result;
  }

private:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  struct Total {
    uint64_t count = 0;
    uint64_t nanoseconds = 0;
  };

  std::array<Total, kBackendTimingMaxOps> totals_{};
  TimePoint active_start_{};
  TimePoint span_start_{};
  TimePoint span_end_{};
  uint64_t dropped_marks_ = 0;
  uint32_t active_op_id_ = 0;
  bool active_ = false;
  bool has_span_ = false;
};
