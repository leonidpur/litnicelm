#include "training_session_controller.hpp"

#include "console_reporter.hpp"
#include "eta_observer.hpp"
#include "profiling_observer.hpp"
#include "reporting_observer.hpp"
#include "tracking_observer.hpp"
#include "training_report_sink.hpp"

#include "tensor_store.hpp"

#include <report_interface.hpp>

#include <iostream>
#include <sstream>
#include <vector>

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

TrainingSessionController::TrainingSessionController(const Config &cfg,
                                                     const Command &cmd,
                                                     const ModelAlgoFactory &algo,
                                                     TrainingReportSink &training_sink)
    : cfg_(cfg),
      training_plan_(TrainingPlan::from(cfg, cmd)),
      convergence_monitor_(cfg, reporters_),
      checkpoint_store_(cfg, algo, reporters_),
      checkpoint_policy_(cfg, training_plan_.mode) {
  add_observer(std::make_unique<ConsoleReporter>());
  add_observer(std::make_unique<TrackingObserver>(cfg, cmd, *this));
  add_observer(
      std::make_unique<ReportingObserver>(training_sink, cmd.runtime_flags.probe));
  add_observer(std::make_unique<ProfilingObserver>());
  add_observer(std::make_unique<EtaObserver>(
      cfg, training_plan_, &training_sink, cmd.runtime_flags.epoch_report_every == 0
                                    ? cfg.logging.epoch_report_every
                                    : cmd.runtime_flags.epoch_report_every));
}

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
                                                    void *grad_base,
                                                    uint64_t grad_size,
                                                    void *adam_base,
                                                    uint64_t adam_size,
                                                    void *temp_base,
                                                    uint64_t temp_size) {
  for (const auto &observer : observers_) {
    observer->init_topology_ready(param_layout, param_base, param_size, grad_base,
                                  grad_size, adam_base, adam_size, temp_base,
                                  temp_size);
  }
}

void TrainingSessionController::memory_usage_ready(
    const TrainingMemoryUsage &usage) {
  for (const auto &observer : observers_) {
    observer->memory_usage_ready(usage);
  }
}

void TrainingSessionController::tensor_store_topology_ready(
    const Config &cfg, const TensorStore &tensor_store) {
  for (const auto &observer : observers_) {
    observer->tensor_store_topology_ready(cfg, tensor_store);
  }
}

void TrainingSessionController::batch_step_ready(uint32_t batch_size,
                                                 uint32_t seq_len,
                                                 uint32_t token_rows,
                                                 uint32_t vocab_size) {
  for (const auto &observer : observers_) {
    observer->batch_step_ready(batch_size, seq_len, token_rows, vocab_size);
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
                                                      const TensorView &tok_emb) {
  for (const auto &observer : observers_) {
    observer->init_tensors_xy_ready(x_rows, x_cols, y_rows, y_cols, tok_emb);
  }
}

uint32_t TrainingSessionController::last_epoch() const {
  return training_plan_.last_epoch;
}

std::string TrainingSessionController::early_stop_message() const {
  return convergence_monitor_.stop_reason_text();
}

TrainingPosition TrainingSessionController::restore_or_initialize(
    TensorStore &tensor_store, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t steps_per_epoch) {
  // Before resuming, which restores the saved convergence state.
  convergence_monitor_.reset();
  TrainingPosition training_position{};
  const bool resumed = resume_from_latest(training_position, device_backend,
                                          data_arena, adam_state,
                                          steps_per_epoch);
  if (!resumed) {
    tensor_store.initialize_parameters_deterministic(device_backend);
    training_position.optimizer_steps = 0;
    training_position.epoch = 0;
    zero_buffer(device_backend, adam_state.base, adam_state.bytes);
    std::cout << "[MC&CListener] Fresh initialization completed for parameters and optimizer state.\n";
  } else if (convergence_monitor_.should_stop()) {
    convergence_monitor_.clear_persisted_stop_for_resume();
  }
  return training_position;
}

bool TrainingSessionController::resume_from_latest(
    TrainingPosition &training_position, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t steps_per_epoch) {
  if (!cfg_.training.incremental) {
    return false;
  }

  if (!checkpoint_store_.latest_exists()) {
    std::cout << "[MC&CListener] Incremental enabled, but checkpoint not found at: "
              << checkpoint_store_.latest_path() << ". Starting fresh.\n";
    return false;
  }

  std::cout << "[MC&CListener] Resuming from " << checkpoint_store_.latest_path()
            << "...\n";

  const CheckpointLoad loaded =
      checkpoint_store_.load_latest(device_backend, data_arena, adam_state);
  if (!loaded.ok) {
    std::cerr << "[MC&CListener] ERROR during resume: " << loaded.error
              << "\n[MC&CListener] Falling back to fresh initialization.\n";
    return false;
  }
  training_position = loaded.position;
  convergence_monitor_.restore(loaded.convergence);
  const uint64_t total_steps =
      steps_per_epoch * static_cast<uint64_t>(training_plan_.last_epoch);
  std::cout << "  -> Success. Resumed at Step [" << training_position.optimizer_steps << "/"
            << total_steps << "], epoch[" << training_position.epoch << "/"
            << training_plan_.last_epoch << "]\n";
  return true;
}

void TrainingSessionController::on_training_start(
    const TrainingPosition &training_position, uint64_t steps_per_epoch,
    ReportSink *sink) {
  std::ostringstream oss;
  if (training_plan_.dry_run()) {
    oss << "Dry-run starting for " << cfg_.training.num_epochs_dry_run
        << " epoch(s)"
        << " (real training: " << cfg_.training.num_epochs_train
        << " epoch(s)" << ")";
  } else {
    oss << "Training starting for " << cfg_.training.num_epochs_train
        << " epoch(s)";
  }
  report_if(sink, ReportEvent::START,
            static_cast<uint32_t>(training_position.optimizer_steps), 0.0f,
            oss.str());
  for (const auto &observer : observers_) {
    observer->on_training_start(training_position, steps_per_epoch, sink);
  }
}

void TrainingSessionController::on_epoch_start(uint32_t epoch) {
  for (const auto &observer : observers_) {
    observer->on_epoch_start(epoch);
  }
}

ContinueTrainingDecision TrainingSessionController::on_epoch_end(uint32_t epoch,
                                             const EpochMetrics &metrics,
                                             TrainingPosition &training_position,
                                             DeviceBackend &device_backend,
                                             ReportSink *sink,
                                             const ArenaView &data_arena,
                                             const AdamStateView &adam_state) {
  training_position.epoch = epoch;
  const ConvergenceVerdict verdict = convergence_monitor_.observe(epoch, metrics);
  checkpoint_policy_.apply(verdict, convergence_monitor_, training_position, checkpoint_store_, device_backend,
                data_arena, adam_state);
  // Every observer runs. Training stops if the monitor or any observer
  // decides to stop (the first such message is kept); undecided observers
  // are ignored.
  ContinueTrainingDecision combined = verdict.decision;
  for (const auto &observer : observers_) {
    const ContinueTrainingDecision decision = observer->on_epoch_end(
        epoch, metrics, training_position, device_backend, sink, data_arena,
        adam_state);
    if (!decision.decided) {
      continue;
    }
    if (combined.decided && !combined.continue_training) {
      continue;
    }
    combined = decision;
  }
  return combined;
}

void TrainingSessionController::on_training_end(const TrainingPosition &training_position,
                                                ReportSink *sink) {
  const std::string end_message =
      convergence_monitor_.early_stopped()
          ? ("Training stopped early: " + convergence_monitor_.stop_reason_text())
          : (training_plan_.dry_run() ? "Dry-run complete" : "Training complete");
  report_if(sink, ReportEvent::END,
            static_cast<uint32_t>(training_position.optimizer_steps), 0.0f,
            end_message);
  for (const auto &observer : observers_) {
    observer->on_training_end(training_position, sink);
  }
}

void TrainingSessionController::on_batch_start(uint64_t optimizer_steps) {
  for (const auto &observer : observers_) {
    observer->on_batch_start(optimizer_steps);
  }
}

void TrainingSessionController::on_batch_end(uint64_t optimizer_steps, double loss) {
  for (const auto &observer : observers_) {
    observer->on_batch_end(optimizer_steps, loss);
  }
}

void TrainingSessionController::on_batch_load_start(uint64_t optimizer_steps) {
  for (const auto &observer : observers_) {
    observer->on_batch_load_start(optimizer_steps);
  }
}

void TrainingSessionController::on_batch_load_end(uint64_t optimizer_steps,
                                                  bool has_batch) {
  for (const auto &observer : observers_) {
    observer->on_batch_load_end(optimizer_steps, has_batch);
  }
}

void TrainingSessionController::on_train_step_start(uint64_t optimizer_steps) {
  for (const auto &observer : observers_) {
    observer->on_train_step_start(optimizer_steps);
  }
}

void TrainingSessionController::on_train_step_end(uint64_t optimizer_steps,
                                                  double loss) {
  for (const auto &observer : observers_) {
    observer->on_train_step_end(optimizer_steps, loss);
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

void TrainingSessionController::on_output_head_start() {
  for (const auto &observer : observers_) {
    observer->on_output_head_start();
  }
}

void TrainingSessionController::on_output_head_end() {
  for (const auto &observer : observers_) {
    observer->on_output_head_end();
  }
}

