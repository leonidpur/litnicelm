#pragma once

#include "training_observer.hpp"

#include <types.hpp>

class ReportRenderer;

class ReportingObserver final : public ITrainingObserver {
public:
  ReportingObserver(ReportRenderer &report_renderer,
                    const RuntimeFlags::ProbeFlags &probe_flags);

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

private:
  ReportRenderer &report_renderer_;
  const RuntimeFlags::ProbeFlags &probe_flags_;
  bool batch_step_reported_ = false;
  bool init_tensors_reported_ = false;
};
