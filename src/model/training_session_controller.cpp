#include "training_session_controller.hpp"

#include "operation_journal.hpp"

#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TRAINING, event, step, value, message);
}
} // namespace

TrainingSessionController::TrainingSessionController(const Config &cfg,
                                                     const Command &cmd)
    : cfg_(cfg), cmd_(cmd) {}

void TrainingSessionController::training_loop_start(
    TrainingState &state, ReportSink *sink, const ArenaView &data_arena,
    const AdamStateView &adam_state) {
  const bool estimate = is_estimation_mode();
  std::ostringstream oss;
  if (estimate) {
    oss << "Dry-run starting for " << cfg_.training.num_epochs_dry_run << " epoch(s)"
        << " (real training: " << cfg_.training.num_epochs_train << " epoch(s)" << ")";
  } else {
    oss << "Training starting for " << cfg_.training.num_epochs_train << " epoch(s)";
  }
  report_if(sink, ReportEvent::START, static_cast<uint32_t>(state.global_step),
            0.0f, oss.str());

  maybe_resume(state, data_arena, adam_state);
}

bool TrainingSessionController::is_estimation_mode() const {
  return cmd_.target == Command::Target::DRY_RUN || cfg_.training.dry_run;
}

bool TrainingSessionController::should_continue(uint32_t epoch) const {
  if (is_estimation_mode()) {
    return epoch <= cfg_.training.num_epochs_dry_run;
  }
  return epoch <= cfg_.training.num_epochs_train;
}

void TrainingSessionController::on_epoch_start(uint32_t epoch) {
  if (epoch == 1) {
    start_time_ = std::chrono::steady_clock::now();
  }
}

void TrainingSessionController::on_epoch_end(uint32_t epoch, float mean_loss,
                                             TrainingState &state,
                                             uint32_t print_mod,
                                             ReportSink *sink,
                                             const ArenaView &data_arena,
                                             const AdamStateView &adam_state) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
  ms_per_epoch_avg_ = (epoch > 0) ? (elapsed / epoch) : 0;
  std::cout << "#";
  if ((epoch % print_mod) == 0) {
    std::ostringstream oss;
    oss << "Epoch " << epoch << " mean_loss=" << mean_loss
        << " " << get_eta_report(epoch);
    report_if(sink, ReportEvent::STEP_COMPLETE, epoch, mean_loss, oss.str());
  }
  state.epoch = epoch;
  maybe_save(state, true, data_arena, adam_state);
}

void TrainingSessionController::training_loop_end(const TrainingState &state,
                                                  ReportSink *sink) {
  const bool estimate = is_estimation_mode();
  if (estimate) {
    report_if(sink, ReportEvent::END,
              static_cast<uint32_t>(state.global_step), 0.0f,
              get_eta_report(state.epoch));
  }

  // 1. Final report message
  report_if(sink, ReportEvent::END,
            static_cast<uint32_t>(state.global_step),
            0.0f,
            estimate ? "Dry-run complete" : "Training complete");

  // 2. Build journal details
  std::ostringstream oss;
  oss << "Status: SUCCESS\n\n"
      << "Mode: " << (estimate ? "DRY_RUN" : "TRAIN") << "\n\n"
      << "Dataset: " << cfg_.tokenization.output_binary << "\n\n"
      << "Epochs completed: " << state.epoch << "\n\n"
      << "Global step: " << state.global_step;

  if (!estimate) {
    oss << "\n\nCheckpoint: " << cfg_.paths.model_file;
  }

  // 3. Log operation (mode-aware)
  log_operation(cfg_.paths.journal_file,
                estimate ? "DRY_RUN" : "TRAIN_RUN",
                oss.str());
}

bool TrainingSessionController::maybe_resume(TrainingState &state,
                                             const ArenaView &data_arena,
                                             const AdamStateView &adam_state) {
  if (!cfg_.training.incremental) {
    return false;
  }

  if (!std::filesystem::exists(cfg_.paths.model_file)) {
    std::cout << "[Trainer] Incremental enabled, but checkpoint not found at: "
              << cfg_.paths.model_file << ". Starting fresh.\n";
    return false;
  }

  std::cout << "[Trainer] Resuming from " << cfg_.paths.model_file << "...\n";

  try {
    std::ifstream in(cfg_.paths.model_file, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Failed to open checkpoint file");
    }

    in.read(reinterpret_cast<char *>(&state.global_step), sizeof(state.global_step));
    in.read(reinterpret_cast<char *>(&state.epoch), sizeof(state.epoch));
    in.read(reinterpret_cast<char *>(data_arena.base), data_arena.bytes);
    in.read(reinterpret_cast<char *>(adam_state.base), adam_state.bytes);

    if (in.gcount() == 0 && data_arena.bytes > 0) {
      throw std::runtime_error("Checkpoint file is truncated or empty");
    }

    std::cout << "  -> Success. Resumed at Epoch " << state.epoch
              << ", Global Step " << state.global_step << "\n";
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[Trainer] ERROR during resume: " << e.what()
              << "\n[Trainer] Falling back to fresh initialization.\n";
    return false;
  }
}

void TrainingSessionController::maybe_save(const TrainingState &state,
                                           bool force,
                                           const ArenaView &data_arena,
                                           const AdamStateView &adam_state) {
  if (is_estimation_mode()) {
    return;
  }

  const bool epoch_milestone =
      (state.epoch > 0 &&
       (state.epoch % cfg_.training.save_interval_epochs == 0));
  if (!force && !epoch_milestone) {
    return;
  }

  std::cout << "[Trainer] Saving checkpoint to " << cfg_.paths.model_file
            << "...\n";

  try {
    std::filesystem::path p(cfg_.paths.model_file);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }

    const bool ok =
        save_checkpoint(cfg_.paths.model_file, cfg_.model, cfg_.conf_version,
                        cfg_.memory.alignment_bytes, data_arena, adam_state,
                        state.global_step);
    if (!ok) {
      throw std::runtime_error("save_checkpoint failed");
    }
    std::cout << "  -> Checkpoint saved successfully at Step "
              << state.global_step << "\n";
  } catch (const std::exception &e) {
    std::cerr << "[Trainer] CRITICAL: Failed to save checkpoint: " << e.what()
              << "\n";
  }
}

std::string TrainingSessionController::get_eta_report(uint32_t current_epoch) const {
  if (ms_per_epoch_avg_ == 0) {
    return "Calculating...";
  }

  if (is_estimation_mode()) {
    const uint32_t dry_total = cfg_.training.num_epochs_dry_run;
    const uint32_t train_total = cfg_.training.num_epochs_train;
    const int64_t projected_total =
        static_cast<int64_t>(train_total) * ms_per_epoch_avg_;
    const int64_t projected_remaining =
        std::max<int64_t>(
            0, static_cast<int64_t>(train_total) -
                   static_cast<int64_t>(current_epoch)) *
        ms_per_epoch_avg_;

    return "[ESTIMATE] Dry-run " + std::to_string(current_epoch) + "/" +
           std::to_string(dry_total) + " | Projected full training: " +
           format_duration(projected_total) + " | Remaining if continued: " +
           format_duration(projected_remaining);
  }

  const uint32_t train_total = cfg_.training.num_epochs_train;
  const int64_t remaining_epochs =
      std::max<int64_t>(0, static_cast<int64_t>(train_total) -
                               static_cast<int64_t>(current_epoch));
  const int64_t remaining = remaining_epochs * ms_per_epoch_avg_;
  return "[TRAIN] Epoch " + std::to_string(current_epoch) + "/" +
         std::to_string(train_total) + " | Rem: " + format_duration(remaining);
}

std::string TrainingSessionController::format_duration(int64_t ms) const {
  if (ms < 0) {
    ms = 0;
  }

  const int64_t total_seconds = ms / 1000;
  const int64_t hours = total_seconds / 3600;
  const int64_t minutes = (total_seconds % 3600) / 60;
  const int64_t seconds = total_seconds % 60;

  std::ostringstream oss;
  if (hours > 0) {
    oss << hours << "h " << minutes << "m " << seconds << "s";
  } else if (minutes > 0) {
    oss << minutes << "m " << seconds << "s";
  } else {
    oss << seconds << "s";
  }
  return oss.str();
}
