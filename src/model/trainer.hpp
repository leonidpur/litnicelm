#pragma once

#include "adam_state_factory.hpp"
#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "gradient_factory.hpp"
#include <config.hpp>
#include "dataset.hpp"
#include "ops.hpp"
#include "optimizer_adamw.hpp"
#include "training_session_controller.hpp"
#include "training_diagnostics_controller.hpp"
#include <report_interface.hpp>
#include "tensor_factory.hpp"
#include <types.hpp>
#include "transformer.hpp"

#include <chrono>
#include <cstdint>
#include <string>

// Trainer does: load/resume, loop epochs, forward, loss, backward (later),
// optimizer step, save.
class Trainer {
public:
  Trainer(const Config &cfg, TensorFactory &tensor_factory, Ops &ops,
          OptimizerAdamW &opt, Transformer &model, const ArenaView &data_arena,
          const ArenaView &grad_arena, GradientFactory &gradient_factory,
          uint64_t decay_bytes,
          const AdamStateView &adam_state, DeviceBackend &device_backend,
          TrainingSessionController &session_controller,
          const RuntimeFlags &runtime_flags,
          TrainingReportSink *sink = nullptr);

  // Train for the epoch count selected by TrainingSessionController.
  void train(IDataLoader &loader);
  static int train_entry_point(const Config &cfg, const Command &cmd);

private:
  const Config &cfg_;
  TensorFactory &tensorFactory_;
  Ops &ops_;
  OptimizerAdamW &opt_;
  Transformer &model_;
  ArenaView data_arena_;
  ArenaView grad_arena_;
  GradientFactory &gradientFactory_;
  uint64_t decay_bytes_ = 0;
  AdamStateView adam_state_;
  RuntimeFlags runtime_flags_{};
  std::chrono::steady_clock::time_point process_start_time_{};
  std::chrono::steady_clock::time_point last_checkpoint_time_{};
  bool has_last_checkpoint_time_ = false;
  TrainingReportSink *sink_ = nullptr;
  DeviceBackend &device_backend_;
  TrainingSessionController &session_controller_;
  TrainingDiagnosticsController diagnostics_;

  void zero_gradients();
  double compute_global_grad_norm() const;
  void clip_gradients();
  void apply_decay_zone_gradients(uint64_t step);
  void apply_no_decay_zone_gradients(uint64_t step);
  double train_one_batch(const TrainBatch &batch, TrainingState &state);
};

