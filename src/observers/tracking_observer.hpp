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

  void init_topology_ready(const NamedLayout &param_layout, void *param_base,
                           uint64_t param_size, void *grad_base,
                           uint64_t grad_size, void *adam_base,
                           uint64_t adam_size, void *temp_base,
                           uint64_t temp_size) override;
  void on_training_start(const TrainingPosition &training_position,
                         uint64_t steps_per_epoch, ReportSink *sink) override;
  void on_epoch_start(uint32_t epoch) override;
  void on_train_step_end(uint64_t optimizer_steps, double loss) override;
  ContinueTrainingDecision on_epoch_end(uint32_t epoch,
                                        const EpochMetrics &metrics,
                    TrainingPosition &training_position,
                    DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;
  void on_checkpoint_save_start(uint64_t optimizer_steps, uint32_t epoch) override;
  void on_checkpoint_save_end(bool ok) override;
  void on_checkpoint_load_end(bool ok) override;
  void on_training_end(const TrainingPosition &training_position, ReportSink *sink) override;

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
  uint64_t paramBytes_ = 0;
  uint32_t current_epoch_ = 0;
};
