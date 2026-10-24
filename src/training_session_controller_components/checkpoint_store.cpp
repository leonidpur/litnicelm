#include "checkpoint_store.hpp"

#include "backend/device_backend.hpp"
#include <litnicelm_utils/log.hpp>
#include "training_session_controller_components_events.hpp"
#include "model_algo_factory.hpp"

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

CheckpointStore::CheckpointStore(const Config &cfg, const ModelAlgoFactory &algo,
                                 const Reporters &reporters)
    : cfg_(cfg), algo_(algo), reporters_(reporters) {}

const std::string &CheckpointStore::latest_path() const {
  return cfg_.paths.model_file_latest;
}

const std::string &CheckpointStore::best_path() const {
  return cfg_.paths.model_file_best;
}

bool CheckpointStore::latest_exists() const {
  return std::filesystem::exists(latest_path());
}

CheckpointLoad CheckpointStore::load_latest(DeviceBackend &device_backend,
                                            const ArenaView &data_arena,
                                            const AdamStateView &adam_state) const {
  CheckpointLoad result;
  try {
    reporters_.on_checkpoint_load_start();
    const LoadedCheckpointState state =
        CheckpointIO(cfg_, algo_, device_backend, data_arena, adam_state).load(latest_path());
    result.position.optimizer_steps = state.step;
    result.position.epoch = state.epoch;
    result.convergence = state.convergence;
    reporters_.on_checkpoint_load_end(true);
    result.ok = true;
  } catch (const std::exception &e) {
    reporters_.on_checkpoint_load_end(false);
    result = CheckpointLoad{};
    result.error = e.what();
  }
  return result;
}

bool CheckpointStore::save_latest(DeviceBackend &device_backend,
                                  const ArenaView &data_arena,
                                  const AdamStateView &adam_state,
                                  const TrainingPosition &training_position,
                                  const CheckpointConvergenceState &convergence,
                                  float train_loss,
                                  std::optional<float> val_loss) const {
  const std::string &path = latest_path();
  const uint64_t optimizer_steps = training_position.optimizer_steps;
  const uint32_t epoch = training_position.epoch;
  try {
    reporters_.on_checkpoint_save_start(optimizer_steps, epoch);
    CheckpointIO(cfg_, algo_, device_backend, data_arena, adam_state)
        .save(path, optimizer_steps, epoch, convergence);
    std::ostringstream losses;
    losses << std::fixed << std::setprecision(4) << "train " << train_loss;
    if (val_loss) {
      losses << ", val " << *val_loss;
    }
    reporters_.on_checkpoint_save_end(true);
    LogFields fields = LogFields()
                           .str("path", path)
                           .u64("epoch", epoch)
                           .u64("step", optimizer_steps)
                           .f64("loss", train_loss);
    if (val_loss) {
      fields.f64("val_loss", *val_loss);
    }
    logger().log(training_session_controller_components_events::kCheckpointSaved,
                path + " (epoch " + std::to_string(epoch) +
                    ", step " + std::to_string(optimizer_steps) + ", " +
                    losses.str() + ")",
                fields);
    return true;
  } catch (const std::exception &e) {
    reporters_.on_checkpoint_save_end(false);
    logger().log(training_session_controller_components_events::kCheckpointSaveFailed, e.what());
    return false;
  }
}

bool CheckpointStore::promote_latest_to_best(float best_loss,
                                             const std::string &best_metric) const {
  const std::string &latest = latest_path();
  const std::string &best = best_path();
  try {
    std::ostringstream loss;
    loss << std::fixed << std::setprecision(4) << best_loss;
    const std::string reason = " (" + best_metric + " loss " + loss.str() + ")";
    const LogFields fields = LogFields()
                                 .str("path", best)
                                 .f64("loss", best_loss)
                                 .str("metric", best_metric);
    if (latest == best) {
      logger().log(training_session_controller_components_events::kBestCheckpointPromoted, "Latest is also best: " + best + reason,
                  fields);
      return true;
    }
    std::filesystem::path best_fs(best);
    if (best_fs.has_parent_path()) {
      std::filesystem::create_directories(best_fs.parent_path());
    }
    std::filesystem::copy_file(latest, best,
                               std::filesystem::copy_options::overwrite_existing);
    // Sidecars: tokenizer fingerprint and the run that wrote the checkpoint.
    for (const char *suffix : {".tokfp", ".run"}) {
      const std::string latest_sidecar = latest + suffix;
      if (std::filesystem::exists(latest_sidecar)) {
        std::filesystem::copy_file(
            latest_sidecar, best + suffix,
            std::filesystem::copy_options::overwrite_existing);
      }
    }
    logger().log(training_session_controller_components_events::kBestCheckpointPromoted, "Copied latest → best: " + best + reason,
                fields);
    return true;
  } catch (const std::exception &e) {
    logger().log(training_session_controller_components_events::kBestCheckpointCopyFailed, e.what());
    return false;
  }
}
