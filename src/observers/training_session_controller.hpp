#pragma once

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "tensor_store.hpp"
#include "checkpoint_policy.hpp"
#include "checkpoint_store.hpp"
#include "convergence_monitor.hpp"
#include "learning_rate_schedule.hpp"
#include "reporters.hpp"
#include "training_plan.hpp"
#include "training_observer.hpp"


#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

class ReportRenderer;
class ModelAlgoFactory;

class TrainingSessionController : public ITrainingObserver {
public:
  // algo must outlive the controller (it lives next to the backend).
  TrainingSessionController(const Config &cfg, const Command &cmd,
                            const ModelAlgoFactory &algo,
                            ReportRenderer &report_renderer);

  bool has_epochs_left(const TrainingPosition &training_position) const {
    return training_position.epoch < training_plan_.last_epoch;
  }
  // Explains why a run with no epochs left trains nothing.
  void nothing_to_do(const TrainingPosition &training_position) const;
  // Validation runs every training.validation_every_epochs and on the last
  // epoch, when the validation split has batches.
  bool validation_due(uint32_t epoch, uint64_t val_batches) const;
  // Optimizer steps in an epoch: one per training.batches_per_step batches;
  // the epoch's last batch closes a partial step.
  uint64_t steps_per_epoch(uint64_t batches_per_epoch) const;
  // Whether batch batch_idx (0-based in the epoch) starts / ends a step.
  bool step_starts_at(uint64_t batch_idx) const;
  bool step_ends_at(uint64_t batch_idx, uint64_t batches_per_epoch) const;
  // Rate for the optimizer step after optimizer_steps completed steps; the
  // schedule is built in on_training_start, which must come first.
  float learning_rate(uint64_t optimizer_steps) const;
  // Reports one validation pass against the epoch's training loss.
  void on_evaluate_end(uint32_t epoch, const ValidationResult &result,
                       float train_loss) const;
  std::string early_stop_message() const;
  bool is_dry_run() const { return training_plan_.dry_run(); }
  bool should_stop() const { return convergence_monitor_.should_stop(); }
  bool has_best() const { return convergence_monitor_.has_best(); }
  float best_loss() const { return convergence_monitor_.best_loss(); }
  uint32_t best_epoch() const { return convergence_monitor_.best_epoch(); }
  std::string best_checkpoint_path() const { return checkpoint_store_.best_path(); }
  void add_observer(std::unique_ptr<ITrainingObserver> observer);
  void runtime_cfg_ready(const Config &cfg) override;
  void init_config_ready(const Config &cfg, const TrainingLayouts &layouts) override;
  void init_topology_ready(const NamedLayout &param_layout, void *param_base,
                           uint64_t param_size, void *grad_base,
                           uint64_t grad_size, void *adam_base,
                           uint64_t adam_size, void *temp_base,
                           uint64_t temp_size) override;
  void memory_usage_ready(const TrainingMemoryUsage &usage) override;
  void tensor_store_topology_ready(const Config &cfg,
                                     const TensorStore &tensor_store) override;
  void batch_step_ready(uint32_t batch_size, uint32_t seq_len,
                        uint32_t token_rows, uint32_t vocab_size) override;
  void probe_loss_ready(const TensorView &loss_scalar, const TensorView &logits,
                        const TensorView &targets) override;
  void probe_output_head_ready(const TensorView &lm_head_w,
                               const TensorView &d_lm_w) override;
  void init_tensors_xy_ready(const TensorView &X, const TensorView &Y,
                             const TensorView &tok_emb) override;

  // Resumes from the latest checkpoint or initializes parameters and
  // optimizer state; returns the position training starts from.
  TrainingPosition restore_or_initialize(TensorStore &tensor_store,
                                        DeviceBackend &device_backend,
                                        const ArenaView &data_arena,
                                        const AdamStateView &adam_state,
                                        uint64_t batches_per_epoch);
  void on_training_start(const TrainingPosition &training_position,
                         uint64_t batches_per_epoch) override;
  void on_training_end(const TrainingPosition &training_position) override;
  void on_epoch_start(uint32_t epoch) override;
  ContinueTrainingDecision on_epoch_end(uint32_t epoch,
                                        const EpochMetrics &metrics,
                    TrainingPosition &training_position,
                    DeviceBackend &device_backend,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;
  void on_batch_start(uint64_t batch_idx) override;
  void on_batch_end(uint64_t batch_idx, double loss) override;
  void on_batch_load_start(uint64_t batch_idx) override;
  void on_batch_load_end(uint64_t batch_idx, bool has_batch) override;
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

private:
  // Logs why training ends: every reason that holds, joined.
  void log_training_end(const TrainingPosition &training_position,
                        bool plan_done, bool monitor_stop,
                        bool observer_stop) const;
  bool resume_from_latest(TrainingPosition &training_position,
                          DeviceBackend &device_backend,
                          const ArenaView &data_arena,
                          const AdamStateView &adam_state,
                          uint64_t steps_per_epoch);

  const Config &cfg_;
  const TrainingPlan training_plan_;
  std::vector<std::unique_ptr<ITrainingObserver>> observers_;
  // Report access to observers_, shared with the components below.
  Reporters reporters_{observers_};
  ConvergenceMonitor convergence_monitor_;
  CheckpointStore checkpoint_store_;
  CheckpointPolicy checkpoint_policy_;
  // The run's plan for the rate; built once the step count is known.
  std::optional<LearningRateSchedule> lr_schedule_;
};
