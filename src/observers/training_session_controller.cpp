#include "training_session_controller.hpp"
#include "logger.hpp"

#include "eta_observer.hpp"
#include "profiling_observer.hpp"
#include "reporting_observer.hpp"
#include "tracking_observer.hpp"
#include "report_renderer.hpp"

#include "tensor_store.hpp"


#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
void zero_buffer(DeviceBackend &device_backend, void *dst, uint64_t bytes) {
  if (dst == nullptr || bytes == 0) {
    return;
  }
  std::vector<uint8_t> zeros(static_cast<size_t>(bytes), 0u);
  device_backend.copy_host2device(dst, zeros.data(), bytes);
}

} // namespace

TrainingSessionController::TrainingSessionController(const Config &cfg,
                                                     const Command &cmd,
                                                     const ModelAlgoFactory &algo,
                                                     ReportRenderer &report_renderer)
    : cfg_(cfg),
      training_plan_(TrainingPlan::from(cfg, cmd)),
      convergence_monitor_(cfg),
      checkpoint_store_(cfg, algo, reporters_),
      checkpoint_policy_(cfg, training_plan_.mode) {
  add_observer(std::make_unique<TrackingObserver>(cfg, cmd, *this));
  add_observer(
      std::make_unique<ReportingObserver>(report_renderer, cmd.runtime_flags.probe));
  add_observer(std::make_unique<ProfilingObserver>());
  add_observer(std::make_unique<EtaObserver>(
      cfg, training_plan_, cmd.runtime_flags.epoch_report_every == 0
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

void TrainingSessionController::init_tensors_xy_ready(const TensorView &X,
                                                      const TensorView &Y,
                                                      const TensorView &tok_emb) {
  for (const auto &observer : observers_) {
    observer->init_tensors_xy_ready(X, Y, tok_emb);
  }
}

bool TrainingSessionController::validation_due(uint32_t epoch,
                                               uint64_t val_batches) const {
  const bool epoch_due = epoch % cfg_.training.validation_every_epochs == 0 ||
                         epoch == training_plan_.last_epoch;
  return epoch_due && val_batches > 0;
}

uint64_t TrainingSessionController::steps_per_epoch(uint64_t batches_per_epoch) const {
  const uint64_t per_step = cfg_.training.batches_per_step;
  return (batches_per_epoch + per_step - 1) / per_step;
}

bool TrainingSessionController::step_starts_at(uint64_t batch_idx) const {
  return batch_idx % cfg_.training.batches_per_step == 0;
}

bool TrainingSessionController::step_ends_at(uint64_t batch_idx,
                                             uint64_t batches_per_epoch) const {
  return (batch_idx + 1) % cfg_.training.batches_per_step == 0 ||
         batch_idx + 1 == batches_per_epoch;
}

void TrainingSessionController::on_evaluate_end(uint32_t epoch,
                                                const ValidationResult &result,
                                                float train_loss) const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << "epoch " << epoch << ": val "
      << result.loss << " (ppl " << std::setprecision(2) << std::exp(result.loss)
      << ") | train " << std::setprecision(4) << train_loss << ", gap "
      << std::showpos << (result.loss - train_loss) << std::noshowpos << " | "
      << result.batches << " batches, " << result.tokens << " tokens";
  logger().log(EventId::ValidationComplete, oss.str(),
               LogFields()
                   .u64("epoch", epoch)
                   .f64("val_loss", result.loss)
                   .f64("train_loss", train_loss)
                   .u64("batches", result.batches)
                   .u64("tokens", result.tokens));
}

float TrainingSessionController::learning_rate(uint64_t optimizer_steps) const {
  if (!lr_schedule_) {
    throw std::logic_error(
        "TrainingSessionController::learning_rate called before on_training_start");
  }
  return lr_schedule_->at(optimizer_steps);
}

std::string TrainingSessionController::early_stop_message() const {
  return convergence_monitor_.stop_reason_text();
}

TrainingPosition TrainingSessionController::restore_or_initialize(
    TensorStore &tensor_store, DeviceBackend &device_backend,
    const ArenaView &data_arena, const AdamStateView &adam_state,
    uint64_t batches_per_epoch) {
  // Before resuming, which restores the saved convergence state.
  convergence_monitor_.reset();
  TrainingPosition training_position{};
  const bool resumed = resume_from_latest(training_position, device_backend,
                                          data_arena, adam_state,
                                          steps_per_epoch(batches_per_epoch));
  if (!resumed) {
    tensor_store.initialize_parameters_deterministic(device_backend);
    training_position.optimizer_steps = 0;
    training_position.epoch = 0;
    zero_buffer(device_backend, adam_state.base, adam_state.bytes);
    logger().log(EventId::FreshInitialization);
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
    logger().log(EventId::CheckpointNotFound,
                 checkpoint_store_.latest_path() + ". Starting fresh.");
    return false;
  }

  logger().log(EventId::ResumeStart, checkpoint_store_.latest_path() + "...");

  const CheckpointLoad loaded =
      checkpoint_store_.load_latest(device_backend, data_arena, adam_state);
  if (!loaded.ok) {
    logger().log(EventId::ResumeFailed,
                 loaded.error +
                     "\n[TrainingSessionController] Falling back to fresh initialization.");
    return false;
  }
  training_position = loaded.position;
  convergence_monitor_.restore(loaded.convergence, training_position.epoch);
  const uint64_t total_steps =
      steps_per_epoch * static_cast<uint64_t>(training_plan_.last_epoch);
  logger().log(EventId::ResumeSucceeded,
               "[" + std::to_string(training_position.optimizer_steps) + "/" +
                   std::to_string(total_steps) + "], epoch[" +
                   std::to_string(training_position.epoch) + "/" +
                   std::to_string(training_plan_.last_epoch) + "]",
               LogFields()
                   .u64("step", training_position.optimizer_steps)
                   .u64("epoch", training_position.epoch));
  return true;
}

void TrainingSessionController::on_training_start(
    const TrainingPosition &training_position, uint64_t batches_per_epoch) {
  // The plan is complete once the epoch size is known.
  lr_schedule_.emplace(cfg_.training,
                       steps_per_epoch(batches_per_epoch) *
                           static_cast<uint64_t>(training_plan_.last_epoch));
  // With no epochs left, nothing_to_do() explains the run instead.
  if (has_epochs_left(training_position)) {
    const uint32_t first_epoch = training_position.epoch + 1;
    const uint32_t last_epoch = training_plan_.last_epoch;
    std::ostringstream oss;
    oss << (training_plan_.dry_run() ? "Dry run: " : "Training ");
    if (first_epoch == last_epoch) {
      oss << "epoch " << first_epoch;
    } else {
      oss << "epochs " << first_epoch << "-" << last_epoch;
    }
    oss << ", starting at step " << training_position.optimizer_steps;
    if (cfg_.training.batches_per_step > 1) {
      oss << " (" << cfg_.training.batches_per_step << " batches per step)";
    }
    if (training_plan_.dry_run()) {
      oss << " (real training: " << cfg_.training.num_epochs_train
          << " epochs)";
    }
    logger().log(EventId::TrainingStart, oss.str());
    lr_schedule_->report();
  }
  for (const auto &observer : observers_) {
    observer->on_training_start(training_position, batches_per_epoch);
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
                                             const ArenaView &data_arena,
                                             const AdamStateView &adam_state) {
  training_position.epoch = epoch;
  const ConvergenceVerdict verdict = convergence_monitor_.observe(epoch, metrics);
  checkpoint_policy_.apply(verdict, convergence_monitor_, training_position, checkpoint_store_, device_backend,
                data_arena, adam_state);
  // Every observer runs; any observer that decides to stop stops training.
  bool observer_stop = false;
  for (const auto &observer : observers_) {
    const ContinueTrainingDecision decision = observer->on_epoch_end(
        epoch, metrics, training_position, device_backend, data_arena,
        adam_state);
    observer_stop |= decision.decided && !decision.continue_training;
  }
  const bool plan_done = epoch >= training_plan_.last_epoch;
  const bool monitor_stop = !verdict.decision.continue_training;

  ContinueTrainingDecision result;
  result.decided = true;
  result.continue_training = !plan_done && !monitor_stop && !observer_stop;
  if (!result.continue_training) {
    log_training_end(training_position, plan_done, monitor_stop, observer_stop);
  }
  return result;
}

void TrainingSessionController::log_training_end(
    const TrainingPosition &training_position, bool plan_done,
    bool monitor_stop, bool observer_stop) const {
  std::vector<std::string> reasons;
  if (plan_done) {
    reasons.push_back("reached last epoch " +
                      std::to_string(training_plan_.last_epoch));
  }
  if (monitor_stop) {
    reasons.push_back(convergence_monitor_.stop_reason_text());
  }
  if (observer_stop) {
    reasons.push_back("requested by an observer");
  }
  std::string message =
      plan_done ? (training_plan_.dry_run() ? "Dry run complete" : "Training complete")
                : "Training stopped early";
  message += " at step " + std::to_string(training_position.optimizer_steps) + ": ";
  for (size_t i = 0; i < reasons.size(); ++i) {
    message += (i ? " and " : "") + reasons[i];
  }
  logger().log(EventId::TrainingEnd, message);
}

void TrainingSessionController::nothing_to_do(
    const TrainingPosition &training_position) const {
  const char *epochs_key = training_plan_.dry_run()
                               ? "training.num_epochs_dry_run"
                               : "training.num_epochs_train";
  logger().log(EventId::NothingToTrain,
               "checkpoint is already at epoch " +
                   std::to_string(training_position.epoch) + " of " +
                   std::to_string(training_plan_.last_epoch) + " (step " +
                   std::to_string(training_position.optimizer_steps) +
                   "). Raise " + epochs_key + " or pass --epochs N to continue.");
}

void TrainingSessionController::on_training_end(const TrainingPosition &training_position) {
  for (const auto &observer : observers_) {
    observer->on_training_end(training_position);
  }
}

void TrainingSessionController::on_batch_start(uint64_t batch_idx) {
  for (const auto &observer : observers_) {
    observer->on_batch_start(batch_idx);
  }
}

void TrainingSessionController::on_batch_end(uint64_t batch_idx, double loss) {
  for (const auto &observer : observers_) {
    observer->on_batch_end(batch_idx, loss);
  }
}

void TrainingSessionController::on_batch_load_start(uint64_t batch_idx) {
  for (const auto &observer : observers_) {
    observer->on_batch_load_start(batch_idx);
  }
}

void TrainingSessionController::on_batch_load_end(uint64_t batch_idx,
                                                  bool has_batch) {
  for (const auto &observer : observers_) {
    observer->on_batch_load_end(batch_idx, has_batch);
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

