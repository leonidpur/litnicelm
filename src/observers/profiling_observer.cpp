#include "profiling_observer.hpp"

#include <iostream>

void ProfilingObserver::on_training_start(TrainingState &state,
                                          TensorStore &tensor_store,
                                          uint64_t steps_per_epoch,
                                          DeviceBackend &device_backend,
                                          ReportSink *sink,
                                          const ArenaView &data_arena,
                                          const AdamStateView &adam_state) {
  (void)state;
  (void)tensor_store;
  (void)steps_per_epoch;
  (void)device_backend;
  (void)sink;
  (void)data_arena;
  (void)adam_state;
  profiling_.reset();
}

void ProfilingObserver::on_training_end(const TrainingState &state,
                                        ReportSink *sink) {
  (void)state;
  (void)sink;
  std::cout << profiling_.summary();
}

void ProfilingObserver::on_batch_start(uint64_t global_step) {
  (void)global_step;
  profiling_.enter(Stage::BATCH_STEP);
}

void ProfilingObserver::on_batch_end(uint64_t global_step, double loss) {
  (void)global_step;
  (void)loss;
  profiling_.leave();
}

void ProfilingObserver::on_batch_load_start(uint64_t global_step) {
  (void)global_step;
  profiling_.enter(Stage::BATCH_LOAD);
}

void ProfilingObserver::on_batch_load_end(uint64_t global_step, bool has_batch) {
  (void)global_step;
  (void)has_batch;
  profiling_.leave();
}

void ProfilingObserver::on_train_step_start(uint64_t global_step) {
  (void)global_step;
  profiling_.enter(Stage::TRAIN_STEP);
}

void ProfilingObserver::on_train_step_end(uint64_t global_step, double loss) {
  (void)global_step;
  (void)loss;
  profiling_.leave();
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

void ProfilingObserver::on_checkpoint_save_start(uint64_t global_step,
                                                 uint32_t epoch) {
  (void)global_step;
  (void)epoch;
  profiling_.enter(Stage::CHECKPOINT_SAVE);
}

void ProfilingObserver::on_checkpoint_save_end(bool ok) {
  (void)ok;
  profiling_.leave();
}
