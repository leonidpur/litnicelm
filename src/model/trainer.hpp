#pragma once

#include "adam_state_store.hpp"
#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "gradient_store.hpp"
#include <config.hpp>
#include "dataset.hpp"
#include "ops.hpp"
#include "optimizer_adamw.hpp"
#include "training_session_controller.hpp"
#include "training_diagnostics_controller.hpp"
#include "gradient_accumulator.hpp"
#include "tensor_store.hpp"
#include <types.hpp>
#include "algo/transformer.hpp"

#include <chrono>
#include <cstdint>
#include <string>

// Trainer does: load/resume, loop epochs, forward, loss, backward (later),
// optimizer step, save.
class TrainingMemoryManager;

class Trainer {
public:
  Trainer(const Config &cfg, TensorStore &tensor_store, const Ops &ops,
          OptimizerAdamW &opt, Transformer &transformer, const ArenaView &data_arena,
          const ArenaView &grad_sum_arena, GradientStore &gradient_store,
          AdamStateStore &adam_state_store, const AdamStateView &adam_state,
          DeviceBackend &device_backend, TrainingMemoryManager &memory_manager,
          TrainingSessionController &session_controller,
          const RuntimeFlags &runtime_flags);

  // Train for the epoch count selected by TrainingSessionController.
  // val_loader is evaluated every training.validation_every_epochs epochs and
  // after the last epoch, when it has batches.
  void train(IDataLoader &loader, IDataLoader &val_loader);
  static int train_entry_point(const Config &cfg, const Command &cmd);

private:
  const Config &cfg_;
  TensorStore &tensorStore_;
  const Ops &ops_;
  OptimizerAdamW &opt_;
  Transformer &transformer_;
  ArenaView data_arena_;
  GradientStore &gradientStore_;
  AdamStateStore &adamStateStore_;
  AdamStateView adam_state_;
  RuntimeFlags runtime_flags_{};
  std::chrono::steady_clock::time_point process_start_time_{};
  std::chrono::steady_clock::time_point last_checkpoint_time_{};
  bool has_last_checkpoint_time_ = false;
  DeviceBackend &device_backend_;
  TrainingMemoryManager &memory_manager_;
  TrainingSessionController &session_controller_;
  TrainingDiagnosticsController diagnostics_;
  GradientAccumulator accumulator_;

  void zero_gradients();
  double compute_global_grad_norm() const;
  void clip_gradients();
  // AdamW on one decay zone: parameters, gradients, m and v come from their
  // stores as flat views of that zone.
  void apply_zone_gradients(DecayZone zone, uint64_t step, float learning_rate);
  // Where a batch sits in its epoch.
  struct BatchInEpoch {
    uint64_t index = 0;  // 0-based
    uint64_t count = 0;  // batches in the epoch
  };
  // Forward and backward of one batch; on the last batch of a step, the
  // optimizer step (which advances training_position). Returns the loss.
  double train_one_batch(const TrainBatch &batch, BatchInEpoch in_epoch,
                         TrainingPosition &training_position);
  // Forward, loss and backward; leaves the batch's gradients in the
  // gradient arena and returns the loss.
  double forward_backward(const TrainBatch &batch);
  // Clips and applies the gradients in the gradient arena.
  // batch: the step's last batch, for non-finite-norm diagnostics.
  void apply_optimizer_step(const TrainBatch &batch,
                            const TrainingPosition &training_position);
  // Mean loss over val_loader's batches: forward pass only, parameters and
  // optimizer state untouched.
  ValidationResult evaluate(IDataLoader &val_loader);
};
