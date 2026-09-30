#include "model_convergence_and_checkpoint_listener.hpp"

#include "model_algo_factory.hpp"
#include "tensor_store.hpp"

#include <report_interface.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

ModelConvergenceAndCheckpointListener::ModelConvergenceAndCheckpointListener(
    const Config &cfg, const Command &cmd, const ModelAlgoFactory &algo,
    const Reporters &reporters)
    : cfg_(cfg), cmd_(cmd), algo_(algo), reporters_(reporters),
      monitor_(cfg, reporters) {}

namespace {
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

TrainingPosition ModelConvergenceAndCheckpointListener::restore_or_initialize(
    TensorStore &tensor_store, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t steps_per_epoch) {
  // Before resuming, which restores the saved convergence state.
  monitor_.reset();
  best_checkpoint_requested_ = false;
  TrainingPosition training_position{};
  const bool resumed = maybe_resume(training_position, device_backend, data_arena,
                                    adam_state, steps_per_epoch,
                                    reporters_);
  if (!resumed) {
    tensor_store.initialize_parameters_deterministic(device_backend);
    training_position.optimizer_steps = 0;
    training_position.epoch = 0;
    zero_buffer(device_backend, adam_state.base, adam_state.bytes);
    std::cout << "[MC&CListener] Fresh initialization completed for parameters and optimizer state.\n";
  } else if (monitor_.should_stop()) {
    monitor_.clear_persisted_stop_for_resume();
  }
  return training_position;
}

void ModelConvergenceAndCheckpointListener::on_training_start(
    const TrainingPosition &training_position, uint64_t steps_per_epoch,
    ReportSink *sink) {
  (void)steps_per_epoch;
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
            static_cast<uint32_t>(training_position.optimizer_steps), 0.0f, oss.str());
}

void ModelConvergenceAndCheckpointListener::on_training_end(
    const TrainingPosition &training_position, ReportSink *sink) {
  const bool estimate = is_estimation_mode();
  const std::string end_message =
      monitor_.early_stopped()
          ? ("Training stopped early: " + monitor_.stop_reason_text())
                    : (estimate ? "Dry-run complete" : "Training complete");

  report_if(sink, ReportEvent::END, static_cast<uint32_t>(training_position.optimizer_steps),
            0.0f, end_message);
}

bool ModelConvergenceAndCheckpointListener::maybe_resume(
    TrainingPosition &training_position, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t steps_per_epoch, const Reporters &reporters) {
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
    reporters.on_checkpoint_load_start();
    std::string error_detail;
    CheckpointConvergenceState restored_convergence_state;
    if (!load_checkpoint(cfg_.paths.model_file_latest, cfg_.model,
                         *algo_.create_position_encoding(cfg_),
                         cfg_.conf_version, cfg_.memory.alignment_bytes,
                         device_backend, data_arena, adam_state,
                         training_position.optimizer_steps, training_position.epoch,
                         &restored_convergence_state, &error_detail)) {
      throw std::runtime_error("Failed to load checkpoint: " +
                               cfg_.paths.model_file_latest + " | " +
                               error_detail);
    }
    const std::string tokfp = checkpoint_tokenizer_fingerprint(cfg_);
    if (!verify_checkpoint_tokenizer_fingerprint(cfg_.paths.model_file_latest,
                                                 tokfp, &error_detail)) {
      throw std::runtime_error("Failed to load checkpoint: " +
                               cfg_.paths.model_file_latest + " | " +
                               error_detail);
    }
    monitor_.restore(restored_convergence_state);
    reporters.on_checkpoint_load_end(true);
    const uint64_t total_steps =
        steps_per_epoch * static_cast<uint64_t>(last_epoch());
    std::cout << "  -> Success. Resumed at Step [" << training_position.optimizer_steps << "/"
              << total_steps << "], epoch[" << training_position.epoch << "/"
              << last_epoch() << "]\n";
    return true;
  } catch (const std::exception &e) {
    reporters.on_checkpoint_load_end(false);
    std::cerr << "[MC&CListener] ERROR during resume: " << e.what()
              << "\n[MC&CListener] Falling back to fresh initialization.\n";
    return false;
  }
}

void ModelConvergenceAndCheckpointListener::maybe_save(
    const TrainingPosition &training_position, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    const Reporters &reporters) {
  if (is_estimation_mode()) {
    return;
  }

  const bool epoch_milestone =
      (training_position.epoch > 0 &&
       (training_position.epoch % cfg_.training.save_interval_epochs == 0));
  if (!epoch_milestone && !best_checkpoint_requested_ &&
      !monitor_.should_stop()) {
    return;
  }

  if (cfg_.training.stop_on_nonfinite_loss &&
      !std::isfinite(monitor_.last_epoch_loss())) {
    std::cerr << "[MC&CListener] Skipping latest checkpoint save because epoch loss is non-finite.\n";
    best_checkpoint_requested_ = false;
    return;
  }

  const bool latest_saved =
      save_checkpoint_file(cfg_.paths.model_file_latest, device_backend,
                           data_arena, adam_state, training_position.optimizer_steps,
                           training_position.epoch, /*notify_observers=*/true,
                           reporters);

  if (latest_saved && best_checkpoint_requested_) {
    best_checkpoint_requested_ = false;
    copy_latest_checkpoint_to_best(cfg_.paths.model_file_latest,
                                   best_checkpoint_path());
  } else if (!latest_saved) {
    best_checkpoint_requested_ = false;
  }
}

ContinueTrainingDecision ModelConvergenceAndCheckpointListener::on_epoch_end(
    uint32_t epoch, const EpochMetrics &metrics, TrainingPosition &training_position,
    DeviceBackend &device_backend, ReportSink *sink,
    const ArenaView &data_arena, const AdamStateView &adam_state) {
  (void)sink;
  const ConvergenceVerdict verdict = monitor_.observe(epoch, metrics);
  best_checkpoint_requested_ = verdict.new_best;
  maybe_save(training_position, device_backend, data_arena, adam_state,
             reporters_);
  return verdict.decision;
}

bool ModelConvergenceAndCheckpointListener::is_estimation_mode() const {
  return cmd_.target == Command::Target::DRY_RUN || cfg_.training.dry_run;
}

uint32_t ModelConvergenceAndCheckpointListener::last_epoch() const {
  return is_estimation_mode() ? cfg_.training.num_epochs_dry_run
                              : cfg_.training.num_epochs_train;
}

bool ModelConvergenceAndCheckpointListener::should_stop() const {
  return monitor_.should_stop();
}

std::string ModelConvergenceAndCheckpointListener::early_stop_message() const {
  return monitor_.stop_reason_text();
}

bool ModelConvergenceAndCheckpointListener::has_best() const {
  return monitor_.has_best();
}

float ModelConvergenceAndCheckpointListener::best_loss() const {
  return monitor_.best_loss();
}

uint32_t ModelConvergenceAndCheckpointListener::best_epoch() const {
  return monitor_.best_epoch();
}





std::string ModelConvergenceAndCheckpointListener::best_checkpoint_path() const {
  return cfg_.paths.model_file_best;
}




bool ModelConvergenceAndCheckpointListener::save_checkpoint_file(
    const std::string &path, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t optimizer_steps, uint32_t epoch, bool notify_observers,
    const Reporters &reporters) {
  try {
    if (notify_observers) {
      reporters.on_checkpoint_save_start(optimizer_steps, epoch);
    }
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }

    const CheckpointConvergenceState convergence_state =
        monitor_.state();
    const bool ok = save_checkpoint(path, cfg_.model,
                                    *algo_.create_position_encoding(cfg_),
                                    cfg_.conf_version,
                                    cfg_.memory.alignment_bytes, device_backend,
                                    data_arena, adam_state, optimizer_steps, epoch,
                                    &convergence_state);
    if (!ok) {
      throw std::runtime_error("save_checkpoint failed");
    }
    std::string fingerprint_error;
    const std::string tokfp = checkpoint_tokenizer_fingerprint(cfg_);
    if (!write_checkpoint_tokenizer_fingerprint(path, tokfp,
                                                &fingerprint_error)) {
      throw std::runtime_error("write tokenizer fingerprint failed: " +
                               fingerprint_error);
    }
    if (notify_observers) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(4) << monitor_.last_epoch_loss();
      reporters.on_checkpoint_save_end(true);
      std::cout << "[MC&CListener] latest to " << path << " at epoch=" << epoch
                << ", step=" << optimizer_steps << ", loss=" << oss.str()
                << "\n";
    }
    return true;
  } catch (const std::exception &e) {
    if (notify_observers) {
      reporters.on_checkpoint_save_end(false);
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
    oss << std::fixed << std::setprecision(4) << monitor_.best_loss();
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
    // Sidecars: tokenizer fingerprint and the run that wrote the checkpoint.
    for (const char *suffix : {".tokfp", ".run"}) {
      const std::string latest_sidecar = latest_path + suffix;
      if (std::filesystem::exists(latest_sidecar)) {
        std::filesystem::copy_file(
            latest_sidecar, best_path + suffix,
            std::filesystem::copy_options::overwrite_existing);
      }
    }
    std::cout << "[MC&CListener] best to " << best_path << ", loss=" << oss.str()
              << "\n";
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[MC&CListener] Failed to copy best checkpoint: " << e.what()
              << "\n";
    return false;
  }
}
