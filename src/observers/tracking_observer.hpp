#pragma once

#include "experiment_tracking_sink.hpp"
#include "training_observer.hpp"

#include <config.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

class ModelConvergenceAndCheckpointListener;

// Turns training callbacks into experiment-tracking events: RUN_START,
// METRICS (per epoch, optionally every N steps), CHECKPOINT and RUN_END (with
// the run summary that the convergence listener holds).
//
// Each saved checkpoint gets a "<checkpoint>.run" sidecar naming the run that
// wrote it; a run resuming from that checkpoint reports it as parent_run_id.
class TrackingObserver final : public ITrainingObserver {
public:
  TrackingObserver(const Config &cfg, const Command &cmd,
                   const ModelConvergenceAndCheckpointListener &convergence);

  void on_training_start(TrainingState &state,
                         TensorStore &tensor_store,
                         uint64_t steps_per_epoch,
                         DeviceBackend &device_backend,
                         ReportSink *sink,
                         const ArenaView &data_arena,
                         const AdamStateView &adam_state) override;
  void on_epoch_start(uint32_t epoch) override;
  void on_train_step_end(uint64_t global_step, double loss) override;
  bool on_epoch_end(uint32_t epoch, float mean_loss,
                    TrainingState &state,
                    DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;
  void on_checkpoint_save_start(uint64_t global_step, uint32_t epoch) override;
  void on_checkpoint_save_end(bool ok) override;
  void on_checkpoint_load_end(bool ok) override;
  void on_training_end(const TrainingState &state, ReportSink *sink) override;

private:
  using Clock = std::chrono::steady_clock;

  const Config &cfg_;
  const std::string configPath_;
  const ModelConvergenceAndCheckpointListener &convergence_;
  std::unique_ptr<ExperimentTrackingSink> tracking_;
  Clock::time_point training_started_at_{};
  Clock::time_point epoch_started_at_{};
  int64_t total_epoch_ms_ = 0;
  uint32_t measured_epochs_ = 0;
  bool training_started_ = false;
  bool epoch_started_ = false;
  uint64_t pending_save_step_ = 0;
  uint32_t pending_save_epoch_ = 0;
  std::string parentRunId_;
  uint32_t current_epoch_ = 0;
};
