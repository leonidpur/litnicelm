#pragma once

#include "checkpoint.hpp"
#include "checkpoint_store.hpp"
#include "convergence_monitor.hpp"
#include "training_plan.hpp"
#include "training_observer.hpp"

#include <config.hpp>

class DeviceBackend;

// When checkpoints are written: the latest one every
// training.save_interval_epochs epochs, on a new best and when training
// stops (never in a dry run, never for a non-finite epoch loss when
// training.stop_on_nonfinite_loss is set); it becomes the best one on a new
// best.
class CheckpointPolicy {
public:
  CheckpointPolicy(const Config &cfg, RunMode mode);

  void apply(const ConvergenceVerdict &verdict,
             const ConvergenceMonitor &monitor,
             const TrainingPosition &training_position,
             const CheckpointStore &store, DeviceBackend &device_backend,
             const ArenaView &data_arena,
             const AdamStateView &adam_state) const;

private:
  const Config &cfg_;
  const RunMode mode_;
};
