#pragma once

#include "training_observer.hpp"

#include <config.hpp>

#include <chrono>
#include <cstdint>

class ModelConvergenceAndCheckpointListener;

class JournalListener final : public ITrainingObserver {
public:
  JournalListener(const Config &cfg, const Command &cmd,
                  const ModelConvergenceAndCheckpointListener &convergence);

  void on_training_start(TrainingState &state,
                         TensorFactory &tensor_factory,
                         uint64_t steps_per_epoch,
                         DeviceBackend &device_backend,
                         ReportSink *sink,
                         const ArenaView &data_arena,
                         const AdamStateView &adam_state) override;
  void on_epoch_start(uint32_t epoch) override;
  bool on_epoch_end(uint32_t epoch, float mean_loss,
                    TrainingState &state,
                    DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;
  void on_training_end(const TrainingState &state, ReportSink *sink) override;

private:
  using Clock = std::chrono::steady_clock;

  const Config &cfg_;
  const ModelConvergenceAndCheckpointListener &convergence_;
  Clock::time_point training_started_at_{};
  Clock::time_point epoch_started_at_{};
  int64_t total_epoch_ms_ = 0;
  uint32_t measured_epochs_ = 0;
  bool training_started_ = false;
  bool epoch_started_ = false;
};
