#include "reporting_observer.hpp"

#include "training_report_sink.hpp"

ReportingObserver::ReportingObserver(
    TrainingReportSink &sink, const RuntimeFlags::ProbeFlags &probe_flags)
    : sink_(sink), probe_flags_(probe_flags) {}

void ReportingObserver::runtime_cfg_ready(const Config &cfg) {
  sink_.report_training_start(cfg);
}

void ReportingObserver::init_config_ready(const Config &cfg,
                                          const NamedLayout &param_layout,
                                          const NamedLayout &temp_layout) {
  sink_.report_init_config(cfg, param_layout, temp_layout);
}

void ReportingObserver::init_topology_ready(const NamedLayout &param_layout,
                                            void *param_base, uint64_t param_size,
                                            void *adam_base, uint64_t adam_size,
                                            void *temp_base, uint64_t temp_size) {
  sink_.report_init_topology(param_layout, param_base, param_size, adam_base,
                             adam_size, temp_base, temp_size);
}

void ReportingObserver::tensor_factory_topology_ready(
    const Config &cfg, const TensorFactory &tensor_factory) {
  sink_.report_tensor_factory_topology(cfg, tensor_factory);
}

void ReportingObserver::batch_step_ready(uint32_t batch_size,
                                         uint32_t window_training,
                                         uint32_t token_rows,
                                         uint32_t vocab_size) {
  sink_.report_batch_step(batch_size, window_training, token_rows, vocab_size);
}

void ReportingObserver::probe_loss_ready(const TensorView &loss_scalar,
                                         const TensorView &logits,
                                         const TensorView &targets) {
  if (!probe_flags_.loss) {
    return;
  }
  sink_.report_probe_loss(loss_scalar, logits, targets);
}

void ReportingObserver::probe_output_head_ready(const TensorView &lm_head_w,
                                                const TensorView &d_lm_w) {
  if (!probe_flags_.output_head) {
    return;
  }
  sink_.report_probe_tensor("output_head", "lm_head_w", lm_head_w);
  sink_.report_probe_tensor("output_head", "d_lm_w", d_lm_w);
}

void ReportingObserver::init_tensors_xy_ready(int64_t x_rows, int64_t x_cols,
                                              int64_t y_rows, int64_t y_cols,
                                              const TensorView &tok_emb,
                                              const TensorView &pos_emb) {
  sink_.init_tensors_X_Y(x_rows, x_cols, y_rows, y_cols, tok_emb, pos_emb);
}
