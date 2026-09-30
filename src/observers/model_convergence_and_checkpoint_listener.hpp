#pragma once

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "reporters.hpp"
#include "training_observer.hpp"

#include <config.hpp>

#include <cstdint>
#include <string>

class ModelAlgoFactory;

class ReportSink;
class TensorStore;
struct TrainingPosition;

class ModelConvergenceAndCheckpointListener final : public ITrainingObserver {
public:
  // reporters must outlive the listener; checkpoint save/load are reported
  // through it.
  ModelConvergenceAndCheckpointListener(const Config &cfg, const Command &cmd,
                                        const ModelAlgoFactory &algo,
                                        const Reporters &reporters);

  // Resumes from the latest checkpoint when training.incremental is set and
  // it loads; otherwise initializes parameters and optimizer state. Returns
  // the position training starts from.
  TrainingPosition restore_or_initialize(TensorStore &tensor_store,
                                        DeviceBackend &device_backend,
                                        const ArenaView &data_arena,
                                        const AdamStateView &adam_state,
                                        uint64_t steps_per_epoch);
  void on_training_start(const TrainingPosition &training_position,
                         uint64_t steps_per_epoch, ReportSink *sink) override;
  void on_training_end(const TrainingPosition &training_position, ReportSink *sink) override;
  ContinueTrainingDecision on_epoch_end(uint32_t epoch,
                                        const EpochMetrics &metrics,
                    TrainingPosition &training_position,
                    DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;

  bool is_estimation_mode() const;
  uint32_t last_epoch() const;
  bool should_stop() const;
  std::string early_stop_message() const;
  bool has_best() const;
  float best_loss() const;
  uint32_t best_epoch() const;
  std::string best_checkpoint_path() const;

private:
  enum class StopReason : uint32_t {
    None = 0,
    TargetLoss = 1,
    Patience = 2,
    NonFiniteLoss = 3,
  };

  bool maybe_resume(TrainingPosition &training_position, DeviceBackend &device_backend,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state, uint64_t steps_per_epoch,
                    const Reporters &reporters);
  void maybe_save(const TrainingPosition &training_position, DeviceBackend &device_backend,
                  const ArenaView &data_arena,
                  const AdamStateView &adam_state,
                  const Reporters &reporters);
  void clear_persisted_stop_for_resume();
  void reset_convergence_state();
  ContinueTrainingDecision epoch_end_decision() const;
  void restore_convergence_state(const CheckpointConvergenceState &state);
  CheckpointConvergenceState checkpoint_convergence_state() const;
  std::string stop_reason_text() const;
  bool loss_improved(float mean_loss) const;
  void request_stop(StopReason reason, const std::string &message);
  bool copy_latest_checkpoint_to_best(const std::string &latest_path,
                                      const std::string &best_path) const;
  bool save_checkpoint_file(const std::string &path, DeviceBackend &device_backend,
                            const ArenaView &data_arena,
                            const AdamStateView &adam_state,
                            uint64_t optimizer_steps, uint32_t epoch,
                            bool notify_observers,
                            const Reporters &reporters);

  const Config &cfg_;
  const Command &cmd_;
  // Checkpoints record and verify the position encoding it creates.
  const ModelAlgoFactory &algo_;
  const Reporters &reporters_;
  float best_loss_ = 0.0f;
  float last_epoch_loss_ = -1.0f;
  uint32_t best_epoch_ = 0;
  uint32_t epochs_without_improvement_ = 0;
  bool has_best_ = false;
  bool stop_requested_ = false;
  bool best_checkpoint_requested_ = false;
  StopReason stop_reason_ = StopReason::None;
  std::string stop_message_;
};
