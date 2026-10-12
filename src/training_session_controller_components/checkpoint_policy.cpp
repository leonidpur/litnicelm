#include "checkpoint_policy.hpp"
#include <litnicelm_utils/log.hpp>
#include "training_session_controller_components_events.hpp"

#include <cmath>

CheckpointPolicy::CheckpointPolicy(const Config &cfg, RunMode mode)
    : cfg_(cfg), mode_(mode) {}

void CheckpointPolicy::apply(const ConvergenceVerdict &verdict,
                             const ConvergenceMonitor &monitor,
                             const TrainingPosition &training_position,
                             const CheckpointStore &store,
                             DeviceBackend &device_backend,
                             const ArenaView &data_arena,
                             const AdamStateView &adam_state) const {
  if (mode_ == RunMode::DryRun) {
    return;
  }

  const bool epoch_milestone =
      (training_position.epoch > 0 &&
       (training_position.epoch % cfg_.training.save_interval_epochs == 0));
  if (!epoch_milestone && !verdict.new_best && !monitor.should_stop()) {
    return;
  }

  if (cfg_.training.stop_on_nonfinite_loss &&
      !std::isfinite(monitor.last_epoch_loss())) {
    logger().log(training_session_controller_components_events::kCheckpointSaveSkippedNonFinite);
    return;
  }

  const bool latest_saved =
      store.save_latest(device_backend, data_arena, adam_state,
                        training_position, monitor.state(),
                        monitor.last_epoch_loss(), monitor.last_val_loss());
  if (latest_saved && verdict.new_best) {
    store.promote_latest_to_best(monitor.best_loss(), monitor.judged_loss_name());
  }
}
