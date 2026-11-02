#include "profiling_observer.hpp"
#include <litnicelm_utils/log.hpp>
#include "observers_events.hpp"
#include <utility>

ProfilingObserver::ProfilingObserver(bool profile_backend_ops,
                                     uint32_t skip_batches,
                                     uint32_t requested_steps,
                                     std::string attention_mode)
    : gpu_op_timer_(skip_batches, requested_steps, std::move(attention_mode)),
      profile_backend_ops_(profile_backend_ops) {}

GpuOpTimer *ProfilingObserver::backend_timer() {
  return profile_backend_ops_ ? &gpu_op_timer_ : nullptr;
}

void ProfilingObserver::on_training_start(
    const TrainingPosition &training_position, uint64_t batches_per_epoch) {
  (void)training_position;
  (void)batches_per_epoch;
  profiling_.reset();
}

void ProfilingObserver::on_training_end(const TrainingPosition &training_position) {
  (void)training_position;
  if (profile_backend_ops_) gpu_op_timer_.stop();
  std::string summary = profiling_.summary();
  if (!summary.empty() && summary.back() == '\n') {
    summary.pop_back();
  }
  if (profile_backend_ops_) {
    if (!summary.empty()) summary += '\n';
    summary += gpu_op_timer_.summary();
  }
  logger().log(observers_events::kProfilingSummary, summary);
}

void ProfilingObserver::on_batch_start(uint64_t batch_idx) {
  (void)batch_idx;
  profiling_.enter(Stage::BATCH_STEP);
}

void ProfilingObserver::on_batch_end(uint64_t batch_idx, double loss) {
  (void)batch_idx;
  (void)loss;
  profiling_.leave();
  if (profile_backend_ops_) gpu_op_timer_.on_batch_end();
}

void ProfilingObserver::on_batch_load_start(uint64_t batch_idx) {
  (void)batch_idx;
  profiling_.enter(Stage::BATCH_LOAD);
}

void ProfilingObserver::on_batch_load_end(uint64_t batch_idx, bool has_batch) {
  (void)batch_idx;
  (void)has_batch;
  profiling_.leave();
}

void ProfilingObserver::on_train_step_start(uint64_t optimizer_steps) {
  (void)optimizer_steps;
  if (profile_backend_ops_) gpu_op_timer_.on_optimizer_step_start();
  profiling_.enter(Stage::TRAIN_STEP);
}

void ProfilingObserver::on_train_step_end(uint64_t optimizer_steps, double loss) {
  (void)optimizer_steps;
  (void)loss;
  profiling_.leave();
  if (profile_backend_ops_) gpu_op_timer_.on_optimizer_step_end();
}

void ProfilingObserver::on_forward_start() { profiling_.enter(Stage::FORWARD); }

void ProfilingObserver::on_forward_end() { profiling_.leave(); }

void ProfilingObserver::on_backward_start() { profiling_.enter(Stage::BACKWARD); }

void ProfilingObserver::on_backward_end() { profiling_.leave(); }

void ProfilingObserver::on_layer_start(int layer_idx) {
  profiling_.enter(Stage::LAYER, layer_idx);
}

void ProfilingObserver::on_layer_end(int layer_idx) {
  (void)layer_idx;
  profiling_.leave();
}

void ProfilingObserver::on_attention_start(int layer_idx) {
  profiling_.enter(Stage::SELF_ATTENTION, layer_idx);
}

void ProfilingObserver::on_attention_end(int layer_idx) {
  (void)layer_idx;
  profiling_.leave();
}

void ProfilingObserver::on_ffn_start(int layer_idx) {
  profiling_.enter(Stage::FFN, layer_idx);
}

void ProfilingObserver::on_ffn_end(int layer_idx) {
  (void)layer_idx;
  profiling_.leave();
}

void ProfilingObserver::on_output_head_start() {
  profiling_.enter(Stage::OUTPUT_HEAD);
}

void ProfilingObserver::on_output_head_end() { profiling_.leave(); }

void ProfilingObserver::on_checkpoint_load_start() {
  profiling_.enter(Stage::CHECKPOINT_LOAD);
}

void ProfilingObserver::on_checkpoint_load_end(bool ok) {
  (void)ok;
  profiling_.leave();
}

void ProfilingObserver::on_checkpoint_save_start(uint64_t optimizer_steps,
                                                 uint32_t epoch) {
  (void)optimizer_steps;
  (void)epoch;
  profiling_.enter(Stage::CHECKPOINT_SAVE);
}

void ProfilingObserver::on_checkpoint_save_end(bool ok) {
  (void)ok;
  profiling_.leave();
}
