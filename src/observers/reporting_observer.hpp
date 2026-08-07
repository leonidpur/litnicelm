#pragma once

#include "training_observer.hpp"

#include <types.hpp>

class TrainingReportSink;

class ReportingObserver final : public ITrainingObserver {
public:
  ReportingObserver(TrainingReportSink &sink,
                    const RuntimeFlags::ProbeFlags &probe_flags);

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

private:
  TrainingReportSink &sink_;
  const RuntimeFlags::ProbeFlags &probe_flags_;
};
