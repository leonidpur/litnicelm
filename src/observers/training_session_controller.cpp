#include "training_session_controller.hpp"

#include "eta_observer.hpp"
#include "journal_listener.hpp"
#include "model_convergence_and_checkpoint_listener.hpp"
#include "profiling_observer.hpp"
#include "reporting_observer.hpp"
#include "training_report_sink.hpp"

#include <vector>

TrainingSessionController::TrainingSessionController(const Config &cfg,
                                                     const Command &cmd,
                                                     TrainingReportSink &training_sink) {
  auto convergence_listener =
      std::make_unique<ModelConvergenceAndCheckpointListener>(cfg, cmd);
  total_epochs_ = convergence_listener->total_epochs();
  convergence_listener->set_observer_relay(*this);
  convergence_listener_ = convergence_listener.get();
  add_observer(std::move(convergence_listener));
  add_observer(std::make_unique<JournalListener>(cfg, cmd, *convergence_listener_));
  add_observer(
      std::make_unique<ReportingObserver>(training_sink, cmd.runtime_flags.probe));
  add_observer(std::make_unique<ProfilingObserver>());
  add_observer(std::make_unique<EtaObserver>(
      cfg, cmd, &training_sink, cmd.runtime_flags.epoch_report_every == 0
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
                                                      const TensorView &tok_emb,
                                                      const TensorView &pos_emb) {
  for (const auto &observer : observers_) {
    observer->init_tensors_xy_ready(x_rows, x_cols, y_rows, y_cols, tok_emb, pos_emb);
  }
}

uint32_t TrainingSessionController::total_epochs() const {
  return total_epochs_;
}

std::string TrainingSessionController::early_stop_message() const {
  return convergence_listener_->early_stop_message();
}

void TrainingSessionController::on_training_start(
    TrainingState &state, TensorStore &tensor_store,
    uint64_t steps_per_epoch, DeviceBackend &device_backend, ReportSink *sink,
    const ArenaView &data_arena,
    const AdamStateView &adam_state) {
  steps_per_epoch_ = steps_per_epoch;
  for (const auto &observer : observers_) {
    observer->on_training_start(state, tensor_store, steps_per_epoch_,
                                device_backend, sink, data_arena, adam_state);
  }
  if (convergence_listener_->should_stop()) {
    total_epochs_ = state.epoch;
  }
}

void TrainingSessionController::on_epoch_start(uint32_t epoch) {
  for (const auto &observer : observers_) {
    observer->on_epoch_start(epoch);
  }
}

bool TrainingSessionController::on_epoch_end(uint32_t epoch, float mean_loss,
                                             TrainingState &state,
                                             DeviceBackend &device_backend,
                                             ReportSink *sink,
                                             const ArenaView &data_arena,
                                             const AdamStateView &adam_state) {
  state.epoch = epoch;
  bool should_continue = true;
  for (const auto &observer : observers_) {
    should_continue =
        observer->on_epoch_end(epoch, mean_loss, state, device_backend, sink,
                               data_arena, adam_state) &&
        should_continue;
  }
  return should_continue;
}

void TrainingSessionController::on_training_end(const TrainingState &state,
                                                ReportSink *sink) {
  for (const auto &observer : observers_) {
    observer->on_training_end(state, sink);
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

void TrainingSessionController::on_batch_load_start(uint64_t global_step) {
  for (const auto &observer : observers_) {
    observer->on_batch_load_start(global_step);
  }
}

void TrainingSessionController::on_batch_load_end(uint64_t global_step,
                                                  bool has_batch) {
  for (const auto &observer : observers_) {
    observer->on_batch_load_end(global_step, has_batch);
  }
}

void TrainingSessionController::on_train_step_start(uint64_t global_step) {
  for (const auto &observer : observers_) {
    observer->on_train_step_start(global_step);
  }
}

void TrainingSessionController::on_train_step_end(uint64_t global_step,
                                                  double loss) {
  for (const auto &observer : observers_) {
    observer->on_train_step_end(global_step, loss);
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
