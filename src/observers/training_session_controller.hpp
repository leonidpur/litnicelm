#pragma once

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

#include <report_interface.hpp>

#include <cstdint>
#include <memory>
#include <vector>

class TrainingReportSink;
class ModelAlgoFactory;
class ModelConvergenceAndCheckpointListener;

class TrainingSessionController : public ITrainingObserver {
public:
  // algo must outlive the controller (it lives next to the backend).
  TrainingSessionController(const Config &cfg, const Command &cmd,
                            const ModelAlgoFactory &algo,
                            TrainingReportSink &training_sink);

  uint32_t last_epoch() const;
  std::string early_stop_message() const;
  void add_observer(std::unique_ptr<ITrainingObserver> observer);
  void runtime_cfg_ready(const Config &cfg) override;
  void init_config_ready(const Config &cfg, const NamedLayout &param_layout,
                         const NamedLayout &temp_layout) override;
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
  void init_tensors_xy_ready(int64_t x_rows, int64_t x_cols, int64_t y_rows,
                             int64_t y_cols, const TensorView &tok_emb) override;

  // Resumes from the latest checkpoint or initializes parameters and
  // optimizer state; returns the position training starts from.
  TrainingPosition restore_or_initialize(TensorStore &tensor_store,
                                        DeviceBackend &device_backend,
                                        const ArenaView &data_arena,
                                        const AdamStateView &adam_state,
                                        uint64_t steps_per_epoch);
  void on_training_start(const TrainingPosition &training_position,
                         uint64_t steps_per_epoch, ReportSink *sink) override;
  void on_training_end(const TrainingPosition &training_position, ReportSink *sink) override;
  void on_epoch_start(uint32_t epoch) override;
  ContinueTrainingDecision on_epoch_end(uint32_t epoch,
                                        const EpochMetrics &metrics,
                    TrainingPosition &training_position,
                    DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state) override;
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
  uint64_t steps_per_epoch_ = 0;
  uint32_t last_epoch_ = 0;
  std::vector<std::unique_ptr<ITrainingObserver>> observers_;
  ModelConvergenceAndCheckpointListener *convergence_listener_ = nullptr;
};
