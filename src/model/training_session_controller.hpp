#pragma once

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "tensor_factory.hpp"

#include <config.hpp>
#include <report_interface.hpp>

#include <chrono>
#include <cstdint>
#include <string>

struct TrainingState {
  uint64_t global_step = 0;
  uint32_t epoch = 0;
};

class TrainingSessionController {
public:
  explicit TrainingSessionController(const Config &cfg, const Command &cmd);

  bool training_loop_start(TrainingState &state, TensorFactory &tensor_factory,
                           uint64_t steps_per_epoch, DeviceBackend &device_backend,
                           ReportSink *sink,
                           const ArenaView &data_arena,
                           const AdamStateView &adam_state);
  bool should_continue(uint32_t epoch) const;
  void on_epoch_start(uint32_t epoch);
  void on_epoch_end(uint32_t epoch, float mean_loss, TrainingState &state,
                    uint32_t epoch_report_every, DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state);
  void training_loop_end(const TrainingState &state, ReportSink *sink);

  bool maybe_resume(TrainingState &state, DeviceBackend &device_backend,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state);
  void maybe_save(const TrainingState &state, bool force,
                  DeviceBackend &device_backend,
                  const ArenaView &data_arena,
                  const AdamStateView &adam_state);
  std::string get_eta_report(uint32_t current_epoch) const;

private:
  const Config &cfg_;
  const Command &cmd_;
  uint64_t steps_per_epoch_ = 0;
  std::chrono::steady_clock::time_point start_time_{};
  int64_t ms_per_epoch_avg_ = 0;

  bool is_estimation_mode() const;
  uint32_t total_epochs() const;
  std::string format_duration(int64_t ms) const;
};
