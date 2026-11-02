#include "gpu_op_timer.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

void GpuOpTimer::on_backend_ready(DeviceBackend &backend) {
  backend_ = &backend;
}

void GpuOpTimer::start() {
  if (backend_ == nullptr) {
    throw std::logic_error("GpuOpTimer::start without an attached backend");
  }
  enabled_ = true;
}

void GpuOpTimer::stop() { enabled_ = false; }

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

  std::ostringstream out;
  out << "backend operations:\n" << std::fixed << std::setprecision(3);
  for (const BackendTimingTotal &total : totals) {
    out << "- " << op_name(static_cast<OpId>(total.op_id))
        << " total_ms=" << static_cast<double>(total.total_nanoseconds) / 1.0e6
        << " count=" << total.count
        << " share=" << (100.0 * total.total_nanoseconds / denominator)
        << "%\n";
  }
  out << "- idle_gaps total_ms=" << static_cast<double>(idle_ns) / 1.0e6
      << " share=" << (100.0 * idle_ns / denominator) << "%\n"
      << "- dropped_marks count=" << report.dropped_marks;
  return out.str();
}
