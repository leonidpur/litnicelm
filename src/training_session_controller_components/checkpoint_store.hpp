#pragma once

#include "checkpoint.hpp"
#include "reporters.hpp"
#include "training_observer.hpp"

#include <config.hpp>

#include <string>

class DeviceBackend;
class ModelAlgoFactory;

// Result of loading the latest checkpoint.
struct CheckpointLoad {
  bool ok = false;
  TrainingPosition position;
  CheckpointConvergenceState convergence;
  std::string error;  // why loading failed, when it did
};

// Reads and writes the training checkpoints (paths.model_file_latest and
// paths.model_file_best) with their tokenizer-fingerprint and run sidecars.
// Decides nothing about when to save or whether to resume.
class CheckpointStore {
public:
  // algo and reporters must outlive the store; save and load are reported
  // through reporters.
  CheckpointStore(const Config &cfg, const ModelAlgoFactory &algo,
                  const Reporters &reporters);

  const std::string &latest_path() const;
  const std::string &best_path() const;
  bool latest_exists() const;

  // Loads parameters and optimizer state into the arenas and checks the
  // tokenizer fingerprint.
  CheckpointLoad load_latest(DeviceBackend &device_backend,
                             const ArenaView &data_arena,
                             const AdamStateView &adam_state) const;
  // epoch_loss is only reported.
  bool save_latest(DeviceBackend &device_backend, const ArenaView &data_arena,
                   const AdamStateView &adam_state,
                   const TrainingPosition &training_position,
                   const CheckpointConvergenceState &convergence,
                   float epoch_loss) const;
  // Copies the latest checkpoint and its sidecars to the best path.
  // best_loss is only reported.
  bool promote_latest_to_best(float best_loss) const;

private:
  const Config &cfg_;
  const ModelAlgoFactory &algo_;
  const Reporters &reporters_;
};
