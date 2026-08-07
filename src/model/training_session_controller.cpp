#include "training_session_controller.hpp"

#include "operation_journal.hpp"

#include <filesystem>
#include <fstream>
#include <cstring>
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

void TrainingSessionController::add_observer(
    std::unique_ptr<ITrainingObserver> observer) {
  observers_.push_back(std::move(observer));
}

void TrainingSessionController::runtime_cfg_ready(const Config &cfg) {
  for (const auto &observer : observers_) {
    observer->runtime_cfg_ready(cfg);
  }
}

void TrainingSessionController::init_config_ready(const Config &cfg,
                                                  const NamedLayout &param_layout,
                                                  const NamedLayout &temp_layout) {
  for (const auto &observer : observers_) {
    observer->init_config_ready(cfg, param_layout, temp_layout);
  }
}

void TrainingSessionController::init_topology_ready(const NamedLayout &param_layout,
                                                    void *param_base,
                                                    uint64_t param_size,
                                                    void *adam_base,
                                                    uint64_t adam_size,
                                                    void *temp_base,
                                                    uint64_t temp_size) {
  for (const auto &observer : observers_) {
    observer->init_topology_ready(param_layout, param_base, param_size, adam_base,
                                  adam_size, temp_base, temp_size);
  }
}

void TrainingSessionController::tensor_factory_topology_ready(
    const Config &cfg, const TensorFactory &tensor_factory) {
  for (const auto &observer : observers_) {
    observer->tensor_factory_topology_ready(cfg, tensor_factory);
  }
}

void TrainingSessionController::batch_step_ready(uint32_t batch_size,
                                                 uint32_t window_training,
                                                 uint32_t token_rows,
                                                 uint32_t vocab_size) {
  for (const auto &observer : observers_) {
    observer->batch_step_ready(batch_size, window_training, token_rows, vocab_size);
  }
}

void TrainingSessionController::probe_loss_ready(const TensorView &loss_scalar,
                                                 const TensorView &logits,
                                                 const TensorView &targets) {
  for (const auto &observer : observers_) {
    observer->probe_loss_ready(loss_scalar, logits, targets);
  }
}

void TrainingSessionController::probe_output_head_ready(
    const TensorView &lm_head_w, const TensorView &d_lm_w) {
  for (const auto &observer : observers_) {
    observer->probe_output_head_ready(lm_head_w, d_lm_w);
  }
}

void TrainingSessionController::init_tensors_xy_ready(int64_t x_rows,
                                                      int64_t x_cols,
                                                      int64_t y_rows,
                                                      int64_t y_cols,
                                                      const TensorView &tok_emb,
                                                      const TensorView &pos_emb) {
  for (const auto &observer : observers_) {
    observer->init_tensors_xy_ready(x_rows, x_cols, y_rows, y_cols, tok_emb, pos_emb);
  }
}

bool TrainingSessionController::training_loop_start(
    TrainingState &state, TensorFactory &tensor_factory, uint64_t steps_per_epoch,
    DeviceBackend &device_backend, ReportSink *sink,
    const ArenaView &data_arena,
    const AdamStateView &adam_state) {
  steps_per_epoch_ = steps_per_epoch;
  const bool estimate = is_estimation_mode();
  std::ostringstream oss;
  if (estimate) {
    oss << "Dry-run starting for " << cfg_.training.num_epochs_dry_run << " epoch(s)"
        << " (real training: " << cfg_.training.num_epochs_train << " epoch(s)" << ")";
  } else {
    oss << "Training starting for " << cfg_.training.num_epochs_train << " epoch(s)";
  }
  on_training_start();
  report_if(sink, ReportEvent::START, static_cast<uint32_t>(state.global_step),
            0.0f, oss.str());

  const bool resumed = maybe_resume(state, device_backend, data_arena, adam_state);
  if (!resumed) {
    tensor_factory.initialize_parameters_deterministic();
    state.global_step = 0;
    state.epoch = 0;
    if (adam_state.base != nullptr && adam_state.bytes > 0) {
      std::memset(adam_state.base, 0, static_cast<size_t>(adam_state.bytes));
    }
    std::cout << "[Trainer] Fresh initialization completed for parameters and optimizer state.\n";
  }
  return resumed;
}

uint32_t TrainingSessionController::total_epochs() const {
  return is_estimation_mode() ? cfg_.training.num_epochs_dry_run
                              : cfg_.training.num_epochs_train;
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
  for (const auto &observer : observers_) {
    observer->on_epoch_start(epoch);
  }
}

void TrainingSessionController::on_epoch_end(uint32_t epoch, float mean_loss,
                                             TrainingState &state,
                                             uint32_t epoch_report_every,
                                             DeviceBackend &device_backend,
                                             ReportSink *sink,
                                             const ArenaView &data_arena,
                                             const AdamStateView &adam_state) {
  (void)epoch_report_every;
  (void)sink;
  state.epoch = epoch;
  this->on_epoch_end(epoch, mean_loss, state.global_step);
  maybe_save(state, true, device_backend, data_arena, adam_state);
}

void TrainingSessionController::training_loop_end(const TrainingState &state,
                                                  ReportSink *sink) {
  const bool estimate = is_estimation_mode();

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
  on_training_end(state.global_step, state.epoch);
  finalize(state.global_step, state.epoch);
}

bool TrainingSessionController::maybe_resume(TrainingState &state,
                                             DeviceBackend &device_backend,
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
    on_checkpoint_load_start();
    std::string error_detail;
    if (!load_checkpoint(cfg_.paths.model_file, cfg_.model, cfg_.conf_version,
                         cfg_.memory.alignment_bytes, device_backend,
                         data_arena, adam_state,
                         state.global_step, state.epoch, &error_detail)) {
      throw std::runtime_error("Failed to load checkpoint: " +
                               cfg_.paths.model_file + " | " + error_detail);
    }
    on_checkpoint_load_end(true);
    const uint64_t total_steps =
        steps_per_epoch_ * static_cast<uint64_t>(total_epochs());
    std::cout << "  -> Success. Resumed at Step [" << state.global_step << "/"
              << total_steps << "], epoch[" << state.epoch << "/"
              << total_epochs() << "]\n";
    return true;
  } catch (const std::exception &e) {
    on_checkpoint_load_end(false);
    std::cerr << "[Trainer] ERROR during resume: " << e.what()
              << "\n[Trainer] Falling back to fresh initialization.\n";
    return false;
  }
}

void TrainingSessionController::maybe_save(const TrainingState &state,
                                           bool force,
                                           DeviceBackend &device_backend,
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
    on_checkpoint_save_start(state.global_step, state.epoch);
    std::filesystem::path p(cfg_.paths.model_file);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }

    const bool ok =
        save_checkpoint(cfg_.paths.model_file, cfg_.model, cfg_.conf_version,
                        cfg_.memory.alignment_bytes, device_backend,
                        data_arena, adam_state,
                        state.global_step, state.epoch);
    if (!ok) {
      throw std::runtime_error("save_checkpoint failed");
    }
    on_checkpoint_save_end(true);
    const uint32_t total_epochs = this->total_epochs();
    const uint64_t total_steps =
        steps_per_epoch_ * static_cast<uint64_t>(total_epochs);
    std::cout << "  -> Checkpoint saved successfully at Step ["
              << state.global_step << "/" << total_steps << "]"
              << ", epoch[" << state.epoch << "/" << total_epochs << "]\n";
  } catch (const std::exception &e) {
    on_checkpoint_save_end(false);
    std::cerr << "[Trainer] CRITICAL: Failed to save checkpoint: " << e.what()
              << "\n";
  }
}

void TrainingSessionController::on_training_start() {
  for (const auto &observer : observers_) {
    observer->on_training_start();
  }
}

void TrainingSessionController::on_training_end(uint64_t global_step,
                                                uint32_t epoch) {
  for (const auto &observer : observers_) {
    observer->on_training_end(global_step, epoch);
  }
}

void TrainingSessionController::on_epoch_end(uint32_t epoch, float mean_loss,
                                             uint64_t global_step) {
  for (const auto &observer : observers_) {
    observer->on_epoch_end(epoch, mean_loss, global_step);
  }
}

void TrainingSessionController::on_batch_start(uint64_t global_step) {
  for (const auto &observer : observers_) {
    observer->on_batch_start(global_step);
  }
}

void TrainingSessionController::on_batch_end(uint64_t global_step, double loss) {
  for (const auto &observer : observers_) {
    observer->on_batch_end(global_step, loss);
  }
}

void TrainingSessionController::on_forward_start() {
  for (const auto &observer : observers_) {
    observer->on_forward_start();
  }
}

void TrainingSessionController::on_forward_end() {
  for (const auto &observer : observers_) {
    observer->on_forward_end();
  }
}

void TrainingSessionController::on_backward_start() {
  for (const auto &observer : observers_) {
    observer->on_backward_start();
  }
}

void TrainingSessionController::on_backward_end() {
  for (const auto &observer : observers_) {
    observer->on_backward_end();
  }
}

void TrainingSessionController::on_layer_start(int layer_idx) {
  for (const auto &observer : observers_) {
    observer->on_layer_start(layer_idx);
  }
}

void TrainingSessionController::on_layer_end(int layer_idx) {
  for (const auto &observer : observers_) {
    observer->on_layer_end(layer_idx);
  }
}

void TrainingSessionController::on_attention_start(int layer_idx) {
  for (const auto &observer : observers_) {
    observer->on_attention_start(layer_idx);
  }
}

void TrainingSessionController::on_attention_end(int layer_idx) {
  for (const auto &observer : observers_) {
    observer->on_attention_end(layer_idx);
  }
}

void TrainingSessionController::on_ffn_start(int layer_idx) {
  for (const auto &observer : observers_) {
    observer->on_ffn_start(layer_idx);
  }
}

void TrainingSessionController::on_ffn_end(int layer_idx) {
  for (const auto &observer : observers_) {
    observer->on_ffn_end(layer_idx);
  }
}

void TrainingSessionController::on_checkpoint_load_start() {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_load_start();
  }
}

void TrainingSessionController::on_checkpoint_load_end(bool ok) {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_load_end(ok);
  }
}

void TrainingSessionController::on_checkpoint_save_start(uint64_t global_step,
                                                         uint32_t epoch) {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_save_start(global_step, epoch);
  }
}

void TrainingSessionController::on_checkpoint_save_end(bool ok) {
  for (const auto &observer : observers_) {
    observer->on_checkpoint_save_end(ok);
  }
}

void TrainingSessionController::finalize(uint64_t global_step, uint32_t epoch) {
  for (const auto &observer : observers_) {
    observer->finalize(global_step, epoch);
  }
}
