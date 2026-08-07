#pragma once

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "training_observer.hpp"

#include <config.hpp>

#include <cstdint>
#include <string>

class ReportSink;
class TensorFactory;
struct TrainingState;

class ModelConvergenceAndCheckpointListener final : public ITrainingObserver {
public:
  ModelConvergenceAndCheckpointListener(const Config &cfg, const Command &cmd);
  void set_observer_relay(ITrainingObserver &observer_relay);

  void on_training_start(TrainingState &state,
                         TensorFactory &tensor_factory,
                         uint64_t steps_per_epoch,
                         DeviceBackend &device_backend,
                         ReportSink *sink,
                         const ArenaView &data_arena,
                         const AdamStateView &adam_state) override;
  void on_training_end(const TrainingState &state, ReportSink *sink) override;
  bool on_epoch_end(uint32_t epoch, float mean_loss,
                    TrainingState &state,
                    DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;

  bool is_estimation_mode() const;
  uint32_t total_epochs() const;
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

  bool maybe_resume(TrainingState &state, DeviceBackend &device_backend,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state, uint64_t steps_per_epoch,
                    ITrainingObserver &observer_relay);
  void maybe_save(const TrainingState &state, DeviceBackend &device_backend,
                  const ArenaView &data_arena,
                  const AdamStateView &adam_state,
                  ITrainingObserver &observer_relay);
  void clear_persisted_stop_for_resume();
  void reset_convergence_state();
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
                            uint64_t global_step, uint32_t epoch,
                            bool notify_observers,
                            ITrainingObserver &observer_relay);

  const Config &cfg_;
  const Command &cmd_;
  ITrainingObserver *observer_relay_ = &default_training_observer();
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
