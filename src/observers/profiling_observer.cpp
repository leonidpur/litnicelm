#include "profiling_observer.hpp"

#include <iostream>

void ProfilingObserver::on_training_start() { profiling_.reset(); }

void ProfilingObserver::on_batch_start(uint64_t global_step) {
  (void)global_step;
  profiling_.enter(Stage::TRAIN_STEP);
}

void ProfilingObserver::on_batch_end(uint64_t global_step, double loss) {
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

void ProfilingObserver::finalize(uint64_t global_step, uint32_t epoch) {
  (void)global_step;
  (void)epoch;
  std::cout << profiling_.summary();
}
