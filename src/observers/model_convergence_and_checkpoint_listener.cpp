#include "model_convergence_and_checkpoint_listener.hpp"

#include "tensor_factory.hpp"

#include <report_interface.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

ModelConvergenceAndCheckpointListener::ModelConvergenceAndCheckpointListener(
    const Config &cfg, const Command &cmd)
    : cfg_(cfg), cmd_(cmd) {}

void ModelConvergenceAndCheckpointListener::set_observer_relay(
    ITrainingObserver &observer_relay) {
  observer_relay_ = &observer_relay;
}

namespace {
constexpr const char *kAnsiGreen = "\033[32m";
constexpr const char *kAnsiYellow = "\033[33m";
constexpr const char *kAnsiReset = "\033[0m";

void zero_buffer(DeviceBackend &device_backend, void *dst, uint64_t bytes) {
  if (dst == nullptr || bytes == 0) {
    return;
  }
  std::vector<uint8_t> zeros(static_cast<size_t>(bytes), 0u);
  device_backend.copy_host2device(dst, zeros.data(), bytes);
}

void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  if (sink == nullptr) {
    return;
  }
  sink->report(event, std::string("[MC&CListener][") +
                          report_utils::event_name(event) + "] step=" +
                          std::to_string(step) + " val=" +
                          std::to_string(value) + " " + message);
}
} // namespace

void ModelConvergenceAndCheckpointListener::on_training_start(
    TrainingState &state, TensorFactory &tensor_factory,
    uint64_t steps_per_epoch, DeviceBackend &device_backend, ReportSink *sink,
    const ArenaView &data_arena,
    const AdamStateView &adam_state) {
  reset_convergence_state();
  const bool estimate = is_estimation_mode();
  std::ostringstream oss;
  if (estimate) {
    oss << "Dry-run starting for " << cfg_.training.num_epochs_dry_run
        << " epoch(s)"
        << " (real training: " << cfg_.training.num_epochs_train
        << " epoch(s)" << ")";
  } else {
    oss << "Training starting for " << cfg_.training.num_epochs_train
        << " epoch(s)";
  }
  report_if(sink, ReportEvent::START,
            static_cast<uint32_t>(state.global_step), 0.0f, oss.str());

  const bool resumed = maybe_resume(state, device_backend, data_arena,
                                    adam_state, steps_per_epoch,
                                    *observer_relay_);
  if (!resumed) {
    tensor_factory.initialize_parameters_deterministic(device_backend);
    state.global_step = 0;
    state.epoch = 0;
    zero_buffer(device_backend, adam_state.base, adam_state.bytes);
    std::cout << "[MC&CListener] Fresh initialization completed for parameters and optimizer state.\n";
  } else if (stop_requested_) {
    clear_persisted_stop_for_resume();
  }
}

void ModelConvergenceAndCheckpointListener::on_training_end(
    const TrainingState &state, ReportSink *sink) {
  const bool estimate = is_estimation_mode();
  const bool early_stopped = stop_requested_ && stop_reason_ != StopReason::None;
  const std::string end_message =
      early_stopped ? ("Training stopped early: " + stop_reason_text())
                    : (estimate ? "Dry-run complete" : "Training complete");

  report_if(sink, ReportEvent::END, static_cast<uint32_t>(state.global_step),
            0.0f, end_message);
}

bool ModelConvergenceAndCheckpointListener::maybe_resume(
    TrainingState &state, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t steps_per_epoch, ITrainingObserver &observer_relay) {
  if (!cfg_.training.incremental) {
    return false;
  }

  if (!std::filesystem::exists(cfg_.paths.model_file_latest)) {
    std::cout << "[MC&CListener] Incremental enabled, but checkpoint not found at: "
              << cfg_.paths.model_file_latest << ". Starting fresh.\n";
    return false;
  }

  std::cout << "[MC&CListener] Resuming from " << cfg_.paths.model_file_latest
            << "...\n";

  try {
    observer_relay.on_checkpoint_load_start();
    std::string error_detail;
    CheckpointConvergenceState restored_convergence_state;
    if (!load_checkpoint(cfg_.paths.model_file_latest, cfg_.model,
                         cfg_.conf_version, cfg_.memory.alignment_bytes,
                         device_backend, data_arena, adam_state,
                         state.global_step, state.epoch,
                         &restored_convergence_state, &error_detail)) {
      throw std::runtime_error("Failed to load checkpoint: " +
                               cfg_.paths.model_file_latest + " | " +
                               error_detail);
    }
    restore_convergence_state(restored_convergence_state);
    observer_relay.on_checkpoint_load_end(true);
    const uint64_t total_steps =
        steps_per_epoch * static_cast<uint64_t>(total_epochs());
    std::cout << "  -> Success. Resumed at Step [" << state.global_step << "/"
              << total_steps << "], epoch[" << state.epoch << "/"
              << total_epochs() << "]\n";
    return true;
  } catch (const std::exception &e) {
    observer_relay.on_checkpoint_load_end(false);
    std::cerr << "[MC&CListener] ERROR during resume: " << e.what()
              << "\n[MC&CListener] Falling back to fresh initialization.\n";
    return false;
  }
}

void ModelConvergenceAndCheckpointListener::maybe_save(
    const TrainingState &state, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    ITrainingObserver &observer_relay) {
  if (is_estimation_mode()) {
    return;
  }

  const bool epoch_milestone =
      (state.epoch > 0 &&
       (state.epoch % cfg_.training.save_interval_epochs == 0));
  if (!epoch_milestone && !best_checkpoint_requested_ && !stop_requested_) {
    return;
  }

  if (cfg_.training.stop_on_nonfinite_loss &&
      !std::isfinite(last_epoch_loss_)) {
    std::cerr << "[MC&CListener] Skipping latest checkpoint save because epoch loss is non-finite.\n";
    best_checkpoint_requested_ = false;
    return;
  }

  const bool latest_saved =
      save_checkpoint_file(cfg_.paths.model_file_latest, device_backend,
                           data_arena, adam_state, state.global_step,
                           state.epoch, /*notify_observers=*/true,
                           observer_relay);

  if (latest_saved && best_checkpoint_requested_) {
    best_checkpoint_requested_ = false;
    copy_latest_checkpoint_to_best(cfg_.paths.model_file_latest,
                                   best_checkpoint_path());
  } else if (!latest_saved) {
    best_checkpoint_requested_ = false;
  }
}

bool ModelConvergenceAndCheckpointListener::on_epoch_end(
    uint32_t epoch, float mean_loss, TrainingState &state,
    DeviceBackend &device_backend, ReportSink *sink,
    const ArenaView &data_arena, const AdamStateView &adam_state) {
  (void)sink;
  best_checkpoint_requested_ = false;
  last_epoch_loss_ = mean_loss;

  if (!std::isfinite(mean_loss)) {
    if (cfg_.training.stop_on_nonfinite_loss) {
      request_stop(StopReason::NonFiniteLoss,
                   "epoch loss became non-finite at epoch " +
                       std::to_string(epoch));
    }
    maybe_save(state, device_backend, data_arena, adam_state,
               *observer_relay_);
    return !stop_requested_;
  }

  if (!has_best_ || loss_improved(mean_loss)) {
    has_best_ = true;
    best_loss_ = mean_loss;
    best_epoch_ = epoch;
    epochs_without_improvement_ = 0;
    best_checkpoint_requested_ = true;
    std::cout << "[MC&CListener] New best loss " << best_loss_
              << " at epoch " << best_epoch_
              << ". Best checkpoint path: " << best_checkpoint_path() << "\n";
  } else {
    if (epoch >= cfg_.training.min_epochs) {
      epochs_without_improvement_ += 1;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "[MC&CListener] No new best at epoch " << epoch
        << ": mean_loss=" << mean_loss
        << ", best=" << best_loss_
        << ", min_delta=" << cfg_.training.min_delta;
    if (cfg_.training.patience_epochs > 0) {
      if (epoch >= cfg_.training.min_epochs) {
        oss << " | patience " << epochs_without_improvement_ << "/"
            << cfg_.training.patience_epochs;
      } else {
        oss << " | patience starts at epoch "
            << cfg_.training.min_epochs;
      }
    }
    std::cout << kAnsiYellow << oss.str() << kAnsiReset << "\n";
  }

  if (cfg_.training.target_loss >= 0.0f &&
      mean_loss <= cfg_.training.target_loss) {
    request_stop(StopReason::TargetLoss,
                 "target loss reached at epoch " + std::to_string(epoch) +
                     " (loss=" + std::to_string(mean_loss) + ")");
  }

  if (cfg_.training.patience_epochs > 0 &&
      epoch >= cfg_.training.min_epochs &&
      epochs_without_improvement_ >= cfg_.training.patience_epochs) {
    request_stop(StopReason::Patience,
                 "no improvement for " +
                     std::to_string(epochs_without_improvement_) +
                     " epoch(s); best loss=" + std::to_string(best_loss_) +
                     " at epoch " + std::to_string(best_epoch_));
  }
  maybe_save(state, device_backend, data_arena, adam_state,
             *observer_relay_);
  return !stop_requested_;
}

bool ModelConvergenceAndCheckpointListener::is_estimation_mode() const {
  return cmd_.target == Command::Target::DRY_RUN || cfg_.training.dry_run;
}

uint32_t ModelConvergenceAndCheckpointListener::total_epochs() const {
  return is_estimation_mode() ? cfg_.training.num_epochs_dry_run
                              : cfg_.training.num_epochs_train;
}

bool ModelConvergenceAndCheckpointListener::should_stop() const {
  return stop_requested_;
}

std::string ModelConvergenceAndCheckpointListener::early_stop_message() const {
  return stop_reason_text();
}

bool ModelConvergenceAndCheckpointListener::has_best() const {
  return has_best_;
}

float ModelConvergenceAndCheckpointListener::best_loss() const {
  return best_loss_;
}

uint32_t ModelConvergenceAndCheckpointListener::best_epoch() const {
  return best_epoch_;
}

void ModelConvergenceAndCheckpointListener::clear_persisted_stop_for_resume() {
  const std::string restored_stop_reason = stop_reason_text();
  stop_requested_ = false;
  stop_reason_ = StopReason::None;
  stop_message_.clear();
  epochs_without_improvement_ = 0;
  std::cout << kAnsiGreen
            << "[MC&CListener] Resumed checkpoint carries a prior early-stop state ("
            << restored_stop_reason
            << "). Current run explicitly requested incremental resume, so the "
               "persisted stop state and no-improvement counter are cleared "
               "and training will continue."
            << kAnsiReset << "\n";
}

void ModelConvergenceAndCheckpointListener::reset_convergence_state() {
  best_loss_ = 0.0f;
  last_epoch_loss_ = -1.0f;
  best_epoch_ = 0;
  epochs_without_improvement_ = 0;
  has_best_ = false;
  stop_requested_ = false;
  best_checkpoint_requested_ = false;
  stop_reason_ = StopReason::None;
  stop_message_.clear();
}

void ModelConvergenceAndCheckpointListener::restore_convergence_state(
    const CheckpointConvergenceState &state) {
  best_loss_ = state.best_loss;
  last_epoch_loss_ = state.last_epoch_loss;
  best_epoch_ = state.best_epoch;
  epochs_without_improvement_ = state.epochs_without_improvement;
  has_best_ = (state.flags & CheckpointConvergenceState::kHasBest) != 0;
  stop_requested_ =
      (state.flags & CheckpointConvergenceState::kStopRequested) != 0;
  best_checkpoint_requested_ = false;
  stop_reason_ = static_cast<StopReason>(state.stop_reason);
  stop_message_.clear();
}

CheckpointConvergenceState
ModelConvergenceAndCheckpointListener::checkpoint_convergence_state() const {
  CheckpointConvergenceState state;
  state.best_loss = best_loss_;
  state.last_epoch_loss = last_epoch_loss_;
  state.best_epoch = best_epoch_;
  state.epochs_without_improvement = epochs_without_improvement_;
  if (has_best_) {
    state.flags |= CheckpointConvergenceState::kHasBest;
  }
  if (stop_requested_) {
    state.flags |= CheckpointConvergenceState::kStopRequested;
  }
  state.stop_reason = static_cast<uint32_t>(stop_reason_);
  return state;
}

std::string ModelConvergenceAndCheckpointListener::best_checkpoint_path() const {
  return cfg_.paths.model_file_best;
}

std::string ModelConvergenceAndCheckpointListener::stop_reason_text() const {
  if (!stop_message_.empty()) {
    return stop_message_;
  }
  switch (stop_reason_) {
  case StopReason::TargetLoss:
    return "target loss reached";
  case StopReason::Patience:
    return "patience exhausted";
  case StopReason::NonFiniteLoss:
    return "non-finite loss";
  case StopReason::None:
  default:
    return "none";
  }
}

bool ModelConvergenceAndCheckpointListener::loss_improved(float mean_loss) const {
  return (best_loss_ - mean_loss) > cfg_.training.min_delta;
}

void ModelConvergenceAndCheckpointListener::request_stop(
    StopReason reason, const std::string &message) {
  if (stop_requested_) {
    return;
  }
  stop_requested_ = true;
  stop_reason_ = reason;
  stop_message_ = message;
  std::cout << "[MC&CListener] Early stop requested: " << stop_message_ << "\n";
}

bool ModelConvergenceAndCheckpointListener::save_checkpoint_file(
    const std::string &path, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t global_step, uint32_t epoch, bool notify_observers,
    ITrainingObserver &observer_relay) {
  try {
    if (notify_observers) {
      observer_relay.on_checkpoint_save_start(global_step, epoch);
    }
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }

    const CheckpointConvergenceState convergence_state =
        checkpoint_convergence_state();
    const bool ok = save_checkpoint(path, cfg_.model, cfg_.conf_version,
                                    cfg_.memory.alignment_bytes, device_backend,
                                    data_arena, adam_state, global_step, epoch,
                                    &convergence_state);
    if (!ok) {
      throw std::runtime_error("save_checkpoint failed");
    }
    if (notify_observers) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(4) << last_epoch_loss_;
      observer_relay.on_checkpoint_save_end(true);
      std::cout << "[MC&CListener] latest to " << path << " at epoch=" << epoch
                << ", step=" << global_step << ", loss=" << oss.str()
                << "\n";
    }
    return true;
  } catch (const std::exception &e) {
    if (notify_observers) {
      observer_relay.on_checkpoint_save_end(false);
      std::cerr << "[MC&CListener] CRITICAL: Failed to save checkpoint: " << e.what()
                << "\n";
    }
    return false;
  }
}

bool ModelConvergenceAndCheckpointListener::copy_latest_checkpoint_to_best(
    const std::string &latest_path, const std::string &best_path) const {
  try {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << best_loss_;
    if (latest_path == best_path) {
      std::cout << "[MC&CListener] best to " << best_path << ", loss="
                << oss.str() << "\n";
      return true;
    }
    std::filesystem::path best(best_path);
    if (best.has_parent_path()) {
      std::filesystem::create_directories(best.parent_path());
    }
    std::filesystem::copy_file(latest_path, best_path,
                               std::filesystem::copy_options::overwrite_existing);
    std::cout << "[MC&CListener] best to " << best_path << ", loss=" << oss.str()
              << "\n";
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MC&CListener] Failed to copy best checkpoint: " << e.what()
              << "\n";
    return false;
  }
}
