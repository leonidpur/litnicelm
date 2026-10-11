#pragma once

#include <cstdint>
#include <optional>
#include <string>

struct Config;
class DeviceBackend;
class NamedLayout;
struct TrainingLayouts;
class TensorStore;
class TensorView;
struct AdamStateView;
struct ArenaView;
struct TrainingMemoryUsage;

struct TrainingPosition {
  uint64_t optimizer_steps = 0;
  uint32_t epoch = 0;
};

// An observer's verdict at the end of an epoch. When decided is false the
// observer has no opinion and continue_training is ignored.
struct ContinueTrainingDecision {
  bool decided = false;
  bool continue_training = true;
};

// Losses of one finished epoch.
struct EpochMetrics {
  float train_loss = 0.0f;  // mean over the epoch's training batches
  std::optional<float> val_loss;  // set when validation ran this epoch
};

// Outcome of one validation pass.
struct ValidationResult {
  double loss = 0.0;       // mean over the validation batches
  uint64_t batches = 0;
  uint64_t tokens = 0;     // predicted tokens: batch_size * seq_len per batch
};

class ITrainingObserver {
public:
  virtual ~ITrainingObserver() = default;

  virtual void runtime_cfg_ready(const Config &cfg) { (void)cfg; }
  virtual void init_config_ready(const Config &cfg, const TrainingLayouts &layouts) {
    (void)cfg;
    (void)layouts;
  }
  virtual void init_topology_ready(const NamedLayout &param_layout, void *param_base,
                                   uint64_t param_size, void *grad_base,
                                   uint64_t grad_size, void *adam_base,
                                   uint64_t adam_size, void *temp_base,
                                   uint64_t temp_size) {
    (void)param_layout;
    (void)param_base;
    (void)param_size;
    (void)grad_base;
    (void)grad_size;
    (void)adam_base;
    (void)adam_size;
    (void)temp_base;
    (void)temp_size;
  }
  virtual void memory_usage_ready(const TrainingMemoryUsage &usage) {
    (void)usage;
  }
  virtual void tensor_store_topology_ready(const Config &cfg,
                                             const TensorStore &tensor_store) {
    (void)cfg;
    (void)tensor_store;
  }
  virtual void batch_step_ready(uint32_t batch_size, uint32_t seq_len,
                                uint32_t token_rows, uint32_t vocab_size) {
    (void)batch_size;
    (void)seq_len;
    (void)token_rows;
    (void)vocab_size;
  }
  virtual void probe_loss_ready(const TensorView &loss_scalar,
                                const TensorView &logits,
                                const TensorView &targets) {
    (void)loss_scalar;
    (void)logits;
    (void)targets;
  }
  virtual void probe_output_head_ready(const TensorView &lm_head_w,
                                       const TensorView &d_lm_w) {
    (void)lm_head_w;
    (void)d_lm_w;
  }
  virtual void init_tensors_xy_ready(const TensorView &X, const TensorView &Y,
                                     const TensorView &tok_emb) {
    (void)X;
    (void)Y;
    (void)tok_emb;
  }

  // Training starts at training_position (restored on resume). Information
  // only: weights and position are already decided.
  virtual void on_training_start(const TrainingPosition &training_position,
                                 uint64_t batches_per_epoch) {
    (void)training_position;
    (void)batches_per_epoch;
  }
  virtual void on_training_end(const TrainingPosition &training_position) {
    (void)training_position;
  }

  virtual void on_epoch_start(uint32_t epoch) { (void)epoch; }
  virtual ContinueTrainingDecision on_epoch_end(uint32_t epoch,
                                                const EpochMetrics &metrics,
                            TrainingPosition &training_position,
                            DeviceBackend &device_backend,
                            const ArenaView &data_arena,
                            const AdamStateView &adam_state) {
    (void)epoch;
    (void)metrics;
    (void)training_position;
    (void)device_backend;
    (void)data_arena;
    (void)adam_state;
    return {};
  }

  // batch_idx: the batch's 0-based index in its epoch.
  virtual void on_batch_start(uint64_t batch_idx) { (void)batch_idx; }
  virtual void on_batch_end(uint64_t batch_idx, double loss) {
    (void)batch_idx;
    (void)loss;
  }
  virtual void on_batch_load_start(uint64_t batch_idx) { (void)batch_idx; }
  virtual void on_batch_load_end(uint64_t batch_idx, bool has_batch) {
    (void)batch_idx;
    (void)has_batch;
  }
  virtual void on_train_step_start(uint64_t optimizer_steps) { (void)optimizer_steps; }
  virtual void on_train_step_end(uint64_t optimizer_steps, double loss) {
    (void)optimizer_steps;
    (void)loss;
  }

  virtual void on_forward_start() {}
  virtual void on_forward_end() {}

  virtual void on_backward_start() {}
  virtual void on_backward_end() {}

  virtual void on_layer_start(int layer_idx) { (void)layer_idx; }
  virtual void on_layer_end(int layer_idx) { (void)layer_idx; }

  virtual void on_attention_start(int layer_idx) { (void)layer_idx; }
  virtual void on_attention_end(int layer_idx) { (void)layer_idx; }

  virtual void on_ffn_start(int layer_idx) { (void)layer_idx; }
  virtual void on_ffn_end(int layer_idx) { (void)layer_idx; }

  virtual void on_output_head_start() {}
  virtual void on_output_head_end() {}

  virtual void on_checkpoint_load_start() {}
  virtual void on_checkpoint_load_end(bool ok) { (void)ok; }

  virtual void on_checkpoint_save_start(uint64_t optimizer_steps, uint32_t epoch) {
    (void)optimizer_steps;
    (void)epoch;
  }
  virtual void on_checkpoint_save_end(bool ok) { (void)ok; }
};

inline ITrainingObserver &default_training_observer() {
  class NullTrainingObserver final : public ITrainingObserver {};
  static NullTrainingObserver observer;
  return observer;
}
