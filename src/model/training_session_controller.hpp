#pragma once

#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include "tensor_factory.hpp"
#include "training_observer.hpp"

#include <config.hpp>
#include <report_interface.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TrainingState {
  uint64_t global_step = 0;
  uint32_t epoch = 0;
};

class TrainingSessionController : public ITrainingObserver {
public:
  explicit TrainingSessionController(const Config &cfg, const Command &cmd);

  bool training_loop_start(TrainingState &state, TensorFactory &tensor_factory,
                           uint64_t steps_per_epoch, DeviceBackend &device_backend,
                           ReportSink *sink,
                           const ArenaView &data_arena,
                           const AdamStateView &adam_state);
  bool should_continue(uint32_t epoch) const;
  void on_epoch_end(uint32_t epoch, float mean_loss, TrainingState &state,
                    uint32_t epoch_report_every, DeviceBackend &device_backend,
                    ReportSink *sink,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state);
  void training_loop_end(const TrainingState &state, ReportSink *sink);
  void add_observer(std::unique_ptr<ITrainingObserver> observer);
  void runtime_cfg_ready(const Config &cfg) override;
  void init_config_ready(const Config &cfg, const NamedLayout &param_layout,
                         const NamedLayout &temp_layout) override;
  void init_topology_ready(const NamedLayout &param_layout, void *param_base,
                           uint64_t param_size, void *adam_base,
                           uint64_t adam_size, void *temp_base,
                           uint64_t temp_size) override;
  void tensor_factory_topology_ready(const Config &cfg,
                                     const TensorFactory &tensor_factory) override;
  void batch_step_ready(uint32_t batch_size, uint32_t window_training,
                        uint32_t token_rows, uint32_t vocab_size) override;
  void probe_loss_ready(const TensorView &loss_scalar, const TensorView &logits,
                        const TensorView &targets) override;
  void probe_output_head_ready(const TensorView &lm_head_w,
                               const TensorView &d_lm_w) override;
  void init_tensors_xy_ready(int64_t x_rows, int64_t x_cols, int64_t y_rows,
                             int64_t y_cols, const TensorView &tok_emb,
                             const TensorView &pos_emb) override;

  void on_training_start() override;
  void on_training_end(uint64_t global_step, uint32_t epoch) override;
  void on_epoch_start(uint32_t epoch) override;
  void on_epoch_end(uint32_t epoch, float mean_loss,
                    uint64_t global_step) override;
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

  bool maybe_resume(TrainingState &state, DeviceBackend &device_backend,
                    const ArenaView &data_arena,
                    const AdamStateView &adam_state);
  void maybe_save(const TrainingState &state, bool force,
                  DeviceBackend &device_backend,
                  const ArenaView &data_arena,
                  const AdamStateView &adam_state);

private:
  const Config &cfg_;
  const Command &cmd_;
  uint64_t steps_per_epoch_ = 0;
  std::vector<std::unique_ptr<ITrainingObserver>> observers_;

  bool is_estimation_mode() const;
  uint32_t total_epochs() const;
};
