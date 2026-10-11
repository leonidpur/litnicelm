#include "reporting_observer.hpp"

#include "logger.hpp"
#include "report_renderer.hpp"
#include "tensor_probe.hpp"

#include <config.hpp>

#include <sstream>

ReportingObserver::ReportingObserver(
    ReportRenderer &report_renderer, const RuntimeFlags::ProbeFlags &probe_flags)
    : report_renderer_(report_renderer), probe_flags_(probe_flags) {}

void ReportingObserver::runtime_cfg_ready(const Config &cfg) {
  std::ostringstream oss;
  oss << "Training hyperparams: "
      << "n_layers=" << cfg.model.n_layers
      << ", n_heads=" << cfg.model.n_heads
      << ", d_model=" << cfg.model.d_model
      << ", d_ff=" << cfg.model.d_ff
      << ", vocab_size(actual)=" << cfg.model.vocab_size
      << ", lr=" << cfg.training.learning_rate
      << ", beta1=" << cfg.training.beta1
      << ", beta2=" << cfg.training.beta2
      << ", eps=" << cfg.training.eps
      << ", weight_decay=" << cfg.training.weight_decay
      << ", grad_clip=" << cfg.training.grad_clip
      << ", train_epochs=" << cfg.training.num_epochs_train
      << ", dry_run_epochs=" << cfg.training.num_epochs_dry_run
      << ", batch_size=" << cfg.training.batch_size
      << ", train_seq_len=" << cfg.training.train_seq_len
      << ", window_stride=" << cfg.training.window_stride
      << ", max_seq_len=" << cfg.model.max_seq_len;
  logger().log(EventId::TrainingHyperparams, oss.str());
}

void ReportingObserver::init_config_ready(const Config &cfg,
                                          const TrainingLayouts &layouts) {
  report_renderer_.report_init_config(cfg, layouts);
}

void ReportingObserver::init_topology_ready(const NamedLayout &param_layout,
                                            void *param_base, uint64_t param_size,
                                            void *grad_base, uint64_t grad_size,
                                            void *adam_base, uint64_t adam_size,
                                            void *temp_base, uint64_t temp_size) {
  report_renderer_.report_init_topology(param_layout, param_base, param_size, grad_base,
                             grad_size, adam_base, adam_size, temp_base,
                             temp_size);
}

void ReportingObserver::memory_usage_ready(const TrainingMemoryUsage &usage) {
  report_renderer_.report_memory_usage(usage);
}

void ReportingObserver::tensor_store_topology_ready(
    const Config &cfg, const TensorStore &tensor_store) {
  report_renderer_.report_tensor_store_topology(cfg, tensor_store);
}

void ReportingObserver::batch_step_ready(uint32_t batch_size,
                                         uint32_t seq_len,
                                         uint32_t token_rows,
                                         uint32_t vocab_size) {
  if (batch_step_reported_) {
    return;
  }
  batch_step_reported_ = true;
  logger().log(EventId::BatchShape,
               "batch_size=" + std::to_string(batch_size) +
                   " seq_len=" + std::to_string(seq_len) +
                   " token_rows=" + std::to_string(token_rows) +
                   " vocab_size=" + std::to_string(vocab_size) + "\n");
}

void ReportingObserver::probe_loss_ready(const TensorView &loss_scalar,
                                         const TensorView &logits,
                                         const TensorView &targets) {
  if (!probe_flags_.loss) {
    return;
  }
  TensorProbe::log_loss(loss_scalar, logits, targets);
}

void ReportingObserver::probe_output_head_ready(const TensorView &lm_head_w,
                                                const TensorView &d_lm_w) {
  if (!probe_flags_.output_head) {
    return;
  }
  TensorProbe::log_tensor("output_head", "lm_head_w", lm_head_w);
  TensorProbe::log_tensor("output_head", "d_lm_w", d_lm_w);
}

void ReportingObserver::init_tensors_xy_ready(const TensorView &X,
                                              const TensorView &Y,
                                              const TensorView &tok_emb) {
  if (init_tensors_reported_) {
    return;
  }
  init_tensors_reported_ = true;
  const auto shape_str = [](const TensorView &t) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < t.rank(); ++i) {
      oss << (i ? " × " : "") << t.dim(i);
    }
    oss << "]";
    return oss.str();
  };
  const std::string x_shape = shape_str(X);
  const std::string y_shape = shape_str(Y);
  logger().log(EventId::ProbeInitTensorsXY,
               (x_shape == y_shape ? "X, Y: " + x_shape
                                   : "X: " + x_shape + ", Y: " + y_shape) +
                   " (batch × seq × d_model)");
  if (tok_emb.device() != Device::CPU || tok_emb.dtype() != DType::F32) {
    logger().log(EventId::ProbeTokEmbPreviewSkipped);
  }
}
