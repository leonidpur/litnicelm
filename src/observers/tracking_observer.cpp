#include "tracking_observer.hpp"

#include "model_convergence_and_checkpoint_listener.hpp"

#include "build_info.hpp"
#include "config_yaml.hpp"

#include <filesystem>
#include <fstream>

namespace {
std::string run_sidecar_path(const std::string &checkpoint_path) {
  return checkpoint_path + ".run";
}

int64_t elapsed_ms(std::chrono::steady_clock::time_point start,
                   std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void add_model_config(TrackingEventBuilder &ev, const ModelConfig &model) {
  ev.i64("config.model.max_seq_len", model.max_seq_len)
      .i64("config.model.n_layers", model.n_layers)
      .i64("config.model.n_heads", model.n_heads)
      .i64("config.model.d_model", model.d_model)
      .i64("config.model.d_ff", model.d_ff)
      .i64("config.model.target_vocab_size", model.target_vocab_size);
}

void add_training_config(TrackingEventBuilder &ev,
                         const TrainingConfig &training) {
  ev.f64("config.training.learning_rate", training.learning_rate)
      .f64("config.training.beta1", training.beta1)
      .f64("config.training.beta2", training.beta2)
      .f64("config.training.eps", training.eps)
      .f64("config.training.weight_decay", training.weight_decay)
      .boolean("config.training.incremental", training.incremental)
      .boolean("config.training.dry_run", training.dry_run)
      .i64("config.training.num_epochs_train", training.num_epochs_train)
      .i64("config.training.num_epochs_dry_run", training.num_epochs_dry_run)
      .i64("config.training.save_interval_epochs",
           training.save_interval_epochs)
      .f64("config.training.grad_clip", training.grad_clip)
      .i64("config.training.train_seq_len", training.train_seq_len)
      .i64("config.training.window_stride", training.window_stride)
      .i64("config.training.batch_size", training.batch_size)
      .f64("config.training.target_loss", training.target_loss)
      .f64("config.training.min_delta", training.min_delta)
      .i64("config.training.patience_epochs", training.patience_epochs)
      .i64("config.training.min_epochs", training.min_epochs)
      .boolean("config.training.stop_on_nonfinite_loss",
               training.stop_on_nonfinite_loss);
}
} // namespace

TrackingObserver::TrackingObserver(
    const Config &cfg, const Command &cmd,
    const ModelConvergenceAndCheckpointListener &convergence)
    : cfg_(cfg), configPath_(cmd.config_path), convergence_(convergence),
      tracking_(std::make_unique<ExperimentTrackingSink>(
          cfg, std::filesystem::path(cmd.config_path).stem().string())) {}

void TrackingObserver::on_training_start(TrainingState &state,
                                         TensorStore &tensor_store,
                                         uint64_t steps_per_epoch,
                                         DeviceBackend &device_backend,
                                         ReportSink *sink,
                                         const ArenaView &data_arena,
                                         const AdamStateView &adam_state) {
  (void)tensor_store;
  (void)device_backend;
  (void)sink;
  (void)adam_state;
  training_started_at_ = Clock::now();
  total_epoch_ms_ = 0;
  measured_epochs_ = 0;
  training_started_ = true;
  epoch_started_ = false;

  const bool estimate = convergence_.is_estimation_mode();
  TrackingEventBuilder ev(TRACKING_EVENT_RUN_START,
                          estimate ? "DRY_RUN" : "TRAIN_RUN");
  ev.step(state.global_step)
      .epoch(state.epoch)
      .str("mode", estimate ? "DRY_RUN" : "TRAIN")
      .str("config_path", configPath_)
      .str("git_commit", build_info::kGitCommit)
      .boolean("git_dirty", build_info::kGitDirty)
      .str("backend_library", cfg_.backend.library)
      .str("position_encoding", cfg_.model_algo.position_encoding)
      .i64("vocab_size", cfg_.model.target_vocab_size)
      .i64("param_bytes", static_cast<int64_t>(data_arena.bytes))
      .i64("steps_per_epoch", static_cast<int64_t>(steps_per_epoch))
      .boolean("resumed", !parentRunId_.empty());
  if (!parentRunId_.empty()) {
    ev.str("parent_run_id", parentRunId_);
  }
  ev.str("config_yaml", config_to_yaml(cfg_));
  add_model_config(ev, cfg_.model);
  add_training_config(ev, cfg_.training);
  tracking_->emit(ev);
}

void TrackingObserver::on_epoch_start(uint32_t epoch) {
  current_epoch_ = epoch;
  epoch_started_at_ = Clock::now();
  epoch_started_ = true;
}

bool TrackingObserver::on_epoch_end(uint32_t epoch, float mean_loss,
                                    TrainingState &state,
                                    DeviceBackend &device_backend,
                                    ReportSink *sink,
                                    const ArenaView &data_arena,
                                    const AdamStateView &adam_state) {
  (void)device_backend;
  (void)sink;
  (void)data_arena;
  (void)adam_state;
  int64_t epoch_ms = 0;
  if (epoch_started_) {
    epoch_ms = elapsed_ms(epoch_started_at_, Clock::now());
    total_epoch_ms_ += epoch_ms;
    measured_epochs_ += 1;
    epoch_started_ = false;
  }
  tracking_->emit(TrackingEventBuilder(TRACKING_EVENT_METRICS, "epoch")
                      .step(state.global_step)
                      .epoch(epoch)
                      .f64("train_loss", mean_loss)
                      .i64("epoch_ms", epoch_ms));
  return true;
}

// global_step is the index of the step that just finished; metrics use the
// completed-step count, like the per-epoch events.
void TrackingObserver::on_train_step_end(uint64_t global_step, double loss) {
  const uint32_t every = cfg_.tracking.metrics_every_n_steps;
  const uint64_t completed = global_step + 1;
  if (every == 0 || completed % every != 0) {
    return;
  }
  tracking_->emit(TrackingEventBuilder(TRACKING_EVENT_METRICS, "step")
                      .step(completed)
                      .epoch(current_epoch_)
                      .f64("train_loss", loss));
}

void TrackingObserver::on_checkpoint_save_start(uint64_t global_step,
                                                uint32_t epoch) {
  pending_save_step_ = global_step;
  pending_save_epoch_ = epoch;
}

void TrackingObserver::on_checkpoint_save_end(bool ok) {
  if (ok) {
    std::ofstream(run_sidecar_path(cfg_.paths.model_file_latest),
                  std::ios::trunc)
        << tracking_->run_id() << "\n";
  }
  tracking_->emit(TrackingEventBuilder(TRACKING_EVENT_CHECKPOINT, "SAVE")
                      .step(pending_save_step_)
                      .epoch(pending_save_epoch_)
                      .boolean("ok", ok)
                      .str("path", cfg_.paths.model_file_latest));
}

// Called while the convergence listener resumes, before this observer's
// on_training_start, so RUN_START can name the parent run.
void TrackingObserver::on_checkpoint_load_end(bool ok) {
  TrackingEventBuilder ev(TRACKING_EVENT_CHECKPOINT, "LOAD");
  ev.boolean("ok", ok).str("path", cfg_.paths.model_file_latest);
  if (ok) {
    std::ifstream in(run_sidecar_path(cfg_.paths.model_file_latest));
    std::getline(in, parentRunId_);
    if (!parentRunId_.empty()) {
      ev.str("parent_run_id", parentRunId_);
    }
  }
  tracking_->emit(ev);
}

void TrackingObserver::on_training_end(const TrainingState &state,
                                       ReportSink *sink) {
  (void)sink;
  const bool estimate = convergence_.is_estimation_mode();
  const int64_t total_ms =
      training_started_ ? elapsed_ms(training_started_at_, Clock::now()) : 0;
  const int64_t avg_epoch_ms =
      measured_epochs_ == 0 ? 0 : total_epoch_ms_ / measured_epochs_;

  TrackingEventBuilder ev(TRACKING_EVENT_RUN_END,
                          estimate ? "DRY_RUN" : "TRAIN_RUN");
  ev.step(state.global_step)
      .epoch(state.epoch)
      .str("status", "SUCCESS")
      .str("mode", estimate ? "DRY_RUN" : "TRAIN")
      .str("input_corpus", cfg_.tokenization.input_corpus)
      .str("dataset", cfg_.tokenization.output_binary)
      .i64("time_total_ms", total_ms)
      .i64("avg_epoch_ms", avg_epoch_ms)
      .i64("measured_epochs", measured_epochs_)
      .i64("epochs_completed", state.epoch)
      .i64("global_step", static_cast<int64_t>(state.global_step));
  if (!estimate) {
    ev.str("checkpoint_latest", cfg_.paths.model_file_latest)
        .str("checkpoint_best", convergence_.best_checkpoint_path());
  }
  if (convergence_.has_best()) {
    ev.f64("best_loss", convergence_.best_loss())
        .i64("best_epoch", convergence_.best_epoch());
  }
  if (convergence_.should_stop()) {
    ev.str("stop_reason", convergence_.early_stop_message());
  }
  add_model_config(ev, cfg_.model);
  add_training_config(ev, cfg_.training);
  tracking_->emit(ev);
  tracking_->flush();
}
