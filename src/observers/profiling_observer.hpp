#pragma once

#include "profiling.hpp"
#include "training_observer.hpp"

#include <cstdint>

class ProfilingObserver final : public ITrainingObserver {
public:
  ProfilingObserver() = default;

  void on_training_start(const TrainingPosition &training_position,
                         uint64_t steps_per_epoch, ReportSink *sink) override;
  void on_training_end(const TrainingPosition &training_position, ReportSink *sink) override;
  void on_batch_start(uint64_t optimizer_steps) override;
  void on_batch_end(uint64_t optimizer_steps, double loss) override;
  void on_batch_load_start(uint64_t optimizer_steps) override;
  void on_batch_load_end(uint64_t optimizer_steps, bool has_batch) override;
  void on_train_step_start(uint64_t optimizer_steps) override;
  void on_train_step_end(uint64_t optimizer_steps, double loss) override;
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
  void on_output_head_start() override;
  void on_output_head_end() override;
  void on_checkpoint_load_start() override;
  void on_checkpoint_load_end(bool ok) override;
  void on_checkpoint_save_start(uint64_t optimizer_steps, uint32_t epoch) override;
  void on_checkpoint_save_end(bool ok) override;

private:
  ProfilingController profiling_;
};
