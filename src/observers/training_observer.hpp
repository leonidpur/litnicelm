#pragma once

#include <cstdint>
#include <string>

struct Config;
class NamedLayout;
class TensorFactory;
class TensorView;

class ITrainingObserver {
public:
  virtual ~ITrainingObserver() = default;

  virtual void runtime_cfg_ready(const Config &cfg) { (void)cfg; }
  virtual void init_config_ready(const Config &cfg, const NamedLayout &param_layout,
                                 const NamedLayout &temp_layout) {
    (void)cfg;
    (void)param_layout;
    (void)temp_layout;
  }
  virtual void init_topology_ready(const NamedLayout &param_layout, void *param_base,
                                   uint64_t param_size, void *adam_base,
                                   uint64_t adam_size, void *temp_base,
                                   uint64_t temp_size) {
    (void)param_layout;
    (void)param_base;
    (void)param_size;
    (void)adam_base;
    (void)adam_size;
    (void)temp_base;
    (void)temp_size;
  }
  virtual void tensor_factory_topology_ready(const Config &cfg,
                                             const TensorFactory &tensor_factory) {
    (void)cfg;
    (void)tensor_factory;
  }
  virtual void batch_step_ready(uint32_t batch_size, uint32_t window_training,
                                uint32_t token_rows, uint32_t vocab_size) {
    (void)batch_size;
    (void)window_training;
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
  virtual void init_tensors_xy_ready(int64_t x_rows, int64_t x_cols,
                                     int64_t y_rows, int64_t y_cols,
                                     const TensorView &tok_emb,
                                     const TensorView &pos_emb) {
    (void)x_rows;
    (void)x_cols;
    (void)y_rows;
    (void)y_cols;
    (void)tok_emb;
    (void)pos_emb;
  }

  virtual void on_training_start() {}
  virtual void on_training_end(uint64_t global_step, uint32_t epoch) {
    (void)global_step;
    (void)epoch;
  }

  virtual void on_epoch_start(uint32_t epoch) { (void)epoch; }
  virtual void on_epoch_end(uint32_t epoch, float mean_loss,
                            uint64_t global_step) {
    (void)epoch;
    (void)mean_loss;
    (void)global_step;
  }

  virtual void on_batch_start(uint64_t global_step) { (void)global_step; }
  virtual void on_batch_end(uint64_t global_step, double loss) {
    (void)global_step;
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

  virtual void on_checkpoint_load_start() {}
  virtual void on_checkpoint_load_end(bool ok) { (void)ok; }

  virtual void on_checkpoint_save_start(uint64_t global_step, uint32_t epoch) {
    (void)global_step;
    (void)epoch;
  }
  virtual void on_checkpoint_save_end(bool ok) { (void)ok; }

  virtual void finalize(uint64_t global_step, uint32_t epoch) {
    (void)global_step;
    (void)epoch;
  }
};

inline ITrainingObserver &default_training_observer() {
  class NullTrainingObserver final : public ITrainingObserver {};
  static NullTrainingObserver observer;
  return observer;
}
