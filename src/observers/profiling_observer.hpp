#pragma once

#include "profiling.hpp"
#include "training_observer.hpp"

#include <cstdint>

class ProfilingObserver final : public ITrainingObserver {
public:
  ProfilingObserver() = default;

  void on_training_start() override;
  void on_batch_start(uint64_t global_step) override;
  void on_batch_end(uint64_t global_step, double loss) override;
  void on_forward_start() override;
  void on_forward_end() override;
  void on_backward_start() override;
  void on_backward_end() override;
  void on_layer_start(int layer_idx) override;
  void on_layer_end(int layer_idx) override;
  void on_attention_start(int layer_idx) override;
  void on_attention_end(int layer_idx) override;
  void on_ffn_start(int layer_idx) override;
  void on_ffn_end(int layer_idx) override;
  void on_checkpoint_load_start() override;
  void on_checkpoint_load_end(bool ok) override;
  void on_checkpoint_save_start(uint64_t global_step, uint32_t epoch) override;
  void on_checkpoint_save_end(bool ok) override;
  void finalize(uint64_t global_step, uint32_t epoch) override;

private:
  ProfilingController profiling_;
};
