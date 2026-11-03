// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "gpu_op_timer.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <utility>

namespace {
constexpr uint64_t kMarksPerOptimizerStep = 4096;
}

GpuOpTimer::GpuOpTimer(uint32_t skip_batches, uint32_t requested_steps,
                       std::string attention_mode)
    : skip_batches_(skip_batches), requested_steps_(requested_steps),
      attention_mode_(std::move(attention_mode)) {
  if (requested_steps_ == 0 ||
      requested_steps_ > 1'000'000 / kMarksPerOptimizerStep) {
    throw std::invalid_argument("invalid backend profiling step count");
  }
  mark_capacity_ = kMarksPerOptimizerStep * requested_steps_;
}

void GpuOpTimer::on_backend_ready(DeviceBackend &backend) {
  backend_ = &backend;
  backend_->timing_prepare(mark_capacity_);
}

void GpuOpTimer::start() {
  if (backend_ == nullptr) {
    throw std::logic_error("GpuOpTimer::start without an attached backend");
  }
  enabled_ = true;
}

void GpuOpTimer::stop() { enabled_ = false; }

void GpuOpTimer::on_batch_end() { ++completed_batches_; }

void GpuOpTimer::on_optimizer_step_start() {
  if (!enabled_ && recorded_steps_ < requested_steps_ &&
      completed_batches_ >= skip_batches_) {
    start();
  }
}

void GpuOpTimer::on_optimizer_step_end() {
  if (!enabled_) return;
  ++recorded_steps_;
  if (recorded_steps_ == requested_steps_) stop();
}

void GpuOpTimer::on_op_begin(OpId op) {
  if (enabled_) backend_->timing_mark_begin(static_cast<uint32_t>(op));
}

void GpuOpTimer::on_op_end(OpId op) {
  (void)op;
  if (enabled_) backend_->timing_mark_end();
}

std::string GpuOpTimer::summary() {
  if (backend_ == nullptr) return {};
  const BackendTimingReport report = backend_->timing_report();
  if (recorded_steps_ != requested_steps_) {
    throw std::runtime_error(
        "backend profiling window incomplete: requested " +
        std::to_string(requested_steps_) + " optimizer steps, recorded " +
        std::to_string(recorded_steps_));
  }
  if (report.dropped_marks != 0) {
    throw std::runtime_error(
        "backend profiling event pool exhausted: capacity " +
        std::to_string(mark_capacity_) + ", dropped " +
        std::to_string(report.dropped_marks) + " marks");
  }
  std::vector<BackendTimingTotal> totals(
      report.totals, report.totals + report.total_count);
  std::sort(totals.begin(), totals.end(),
            [](const BackendTimingTotal &a, const BackendTimingTotal &b) {
              return a.total_nanoseconds > b.total_nanoseconds;
            });

  uint64_t operation_ns = 0;
  for (const BackendTimingTotal &total : totals) {
    operation_ns += total.total_nanoseconds;
  }
  const uint64_t idle_ns = report.marked_span_nanoseconds > operation_ns
                               ? report.marked_span_nanoseconds - operation_ns
                               : 0;
  const double denominator =
      report.marked_span_nanoseconds == 0
          ? 1.0
          : static_cast<double>(report.marked_span_nanoseconds);

  const bool multistream = attention_mode_.find("multistream") != std::string::npos;
  const bool overlap = operation_ns > report.marked_span_nanoseconds;
  const bool shares_available = !multistream && !overlap;
  std::ostringstream out;
  out << "backend operations: window_skip_batches=" << skip_batches_
      << " window_optimizer_steps=" << recorded_steps_
      << " mark_capacity=" << mark_capacity_
      << " attention_mode=" << attention_mode_
      << " stream_mode=" << (multistream ? "multistream" : "single_stream")
      << "\n"
      << std::fixed << std::setprecision(3);
  for (const BackendTimingTotal &total : totals) {
    out << "- " << op_name(static_cast<OpId>(total.op_id))
        << " total_ms=" << static_cast<double>(total.total_nanoseconds) / 1.0e6
        << " count=" << total.count;
    if (shares_available) {
      out << " share=" << (100.0 * total.total_nanoseconds / denominator) << "%";
    }
    out << "\n";
  }
  if (multistream) {
    out << "- concurrent_streams operation_total_ms="
        << static_cast<double>(operation_ns) / 1.0e6
        << " span_ms=" << static_cast<double>(report.marked_span_nanoseconds) / 1.0e6
        << " shares=unavailable idle_gaps=unavailable\n";
  } else if (overlap) {
    out << "- overlap_detected operation_total_ms="
        << static_cast<double>(operation_ns) / 1.0e6
        << " span_ms=" << static_cast<double>(report.marked_span_nanoseconds) / 1.0e6
        << " shares=unavailable idle_gaps=unavailable\n";
  } else {
    out << "- idle_gaps total_ms=" << static_cast<double>(idle_ns) / 1.0e6
        << " share=" << (100.0 * idle_ns / denominator) << "%\n";
  }
  out << "- dropped_marks count=" << report.dropped_marks;
  return out.str();
}
