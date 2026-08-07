#include "training_report_sink.hpp"

#include <config.hpp>
#include "named_layout.hpp"
#include "tensor_factory.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
const char *layout_type(const TensorView &t) {
  if (t.is_contiguous_row_major()) {
    return "Cont. Row-Major";
  }
  return "Strided Subview";
}

std::string infer_purpose(const std::string &name) {
  if (name == "tok_embedding") return "Token embedding lookup matrix";
  if (name == "pos_embedding") return "Positional embedding lookup matrix";
  if (name.find("attn_qkv_w") != std::string::npos)
    return "Weight matrix for Query/Key/Value projections";
  if (name.find("attn_qkv_b") != std::string::npos)
    return "Bias vector for Query/Key/Value projections";
  if (name.find("attn_out_w") != std::string::npos)
    return "Attention output projection matrix";
  if (name.find("attn_out_b") != std::string::npos)
    return "Attention output projection bias";
  if (name.find("ffn_w1") != std::string::npos)
    return "Feed-forward expansion weight matrix";
  if (name.find("ffn_b1") != std::string::npos)
    return "Feed-forward expansion bias";
  if (name.find("ffn_w2") != std::string::npos)
    return "Feed-forward contraction weight matrix";
  if (name.find("ffn_b2") != std::string::npos)
    return "Feed-forward contraction bias";
  if (name.find("ln1_gamma") != std::string::npos ||
      name.find("ln2_gamma") != std::string::npos || name == "lnf_gamma")
    return "Layer normalization scaling";
  if (name.find("ln1_beta") != std::string::npos ||
      name.find("ln2_beta") != std::string::npos || name == "lnf_beta")
    return "Layer normalization bias";
  if (name == "lm_head_w") return "Language-model logits projection matrix";
  return "Model parameter tensor";
}

std::string infer_temp_purpose(const std::string &name) {
  if (name == "ds.ids") return "Dataset input-token buffer";
  if (name == "ds.targets") return "Dataset target-token buffer";
  if (name == "infer.ids") return "Inference token-id buffer";
  if (name == "infer.logits") return "Inference logits buffer";
  if (name == "tr.logits") return "Training logits buffer";
  if (name == "tr.loss") return "Scalar loss buffer";
  if (name == "tr.X" || name == "tr.Y" || name == "tr.Xn")
    return "Training hidden-state buffer";
  if (name == "bw.XnT") return "Transposed normalized hidden states";
  if (name == "bw.d_lm_w") return "LM head weight gradients";
  if (name == "bw.lm_wT") return "Transposed LM head weights";
  if (name == "bw.d_xn" || name == "bw.d_xlast") return "Backward hidden gradients";
  if (name == "bw.d_lnf_g" || name == "bw.d_lnf_b")
    return "Final layernorm parameter gradients";
  if (name == "bw.d_tok" || name == "bw.d_pos")
    return "Embedding gradient accumulators";
  if (name.find("layer") == 0 && name.find(".ln") != std::string::npos)
    return "Layernorm or residual workspace";
  if (name.find("attn.") != std::string::npos)
    return "Attention workspace";
  if (name.find("ffn.") != std::string::npos)
    return "Feed-forward workspace";
  return "Temporary tensor";
}

std::string tensor_metadata_row(const std::string &name, const TensorView &tv,
                                const std::string &purpose) {
  std::ostringstream addr;
  addr << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(tv.data());
  const std::string shape =
      std::to_string(tv.shape().r) + "x" + std::to_string(tv.shape().c);
  const std::string stride =
      std::to_string(tv.stride_r_bytes()) + "/" + std::to_string(tv.stride_c_bytes());

  std::ostringstream line;
  line << "| " << std::left << std::setw(18) << name
       << " | " << std::setw(7) << shape
       << " | " << std::setw(17) << layout_type(tv)
       << " | " << std::setw(6) << stride
       << " | " << std::setw(14) << addr.str()
       << " | " << std::right << std::setw(7) << tv.bytes()
       << " | " << std::left << purpose;
  return line.str();
}

double l2_norm_f32_cpu(const TensorView &t) {
  if (t.device() != Device::CPU || t.dtype() != DType::F32 ||
      !t.is_contiguous_row_major()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const float *p = reinterpret_cast<const float *>(t.data());
  const int64_t n = t.shape().r * t.shape().c;
  double sum_sq = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(p[i]);
    sum_sq += v * v;
  }
  return std::sqrt(sum_sq);
}

bool all_finite_f32_cpu(const TensorView &t) {
  if (t.device() != Device::CPU || t.dtype() != DType::F32 ||
      !t.is_contiguous_row_major()) {
    return false;
  }
  const float *p = reinterpret_cast<const float *>(t.data());
  const int64_t n = t.shape().r * t.shape().c;
  for (int64_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) {
      return false;
    }
  }
  return true;
}

struct ProbeStats {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
  uint32_t hash = 2166136261u;
};

void hash_bytes(uint32_t &h, const void *ptr, size_t n) {
  const auto *p = reinterpret_cast<const uint8_t *>(ptr);
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint32_t>(p[i]);
    h *= 16777619u;
  }
}

ProbeStats probe_stats_f32(const TensorView &t) {
  if (t.device() != Device::CPU || t.dtype() != DType::F32) {
    return {};
  }
  ProbeStats s;
  const int64_t R = t.shape().r;
  const int64_t C = t.shape().c;
  double sum = 0.0;
  double sum_sq = 0.0;
  bool init = false;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      const float v = t.at_f32(r, c);
      hash_bytes(s.hash, &v, sizeof(float));
      const double x = static_cast<double>(v);
      if (!init) {
        s.min = s.max = x;
        init = true;
      } else {
        if (x < s.min) s.min = x;
        if (x > s.max) s.max = x;
      }
      sum += x;
      sum_sq += x * x;
    }
  }
  const double n = static_cast<double>(R * C);
  if (n > 0.0) {
    s.mean = sum / n;
    double var = (sum_sq / n) - (s.mean * s.mean);
    if (var < 0.0) {
      var = 0.0;
    }
    s.stddev = std::sqrt(var);
  }
  return s;
}
} // namespace

TrainingReportSink::TrainingReportSink(const LoggingConfig &logging)
    : console_(logging) {}

void TrainingReportSink::report(ReportEvent event, const std::string &message) {
  console_.report(event, message);
}

void TrainingReportSink::report_probe_tensor(const std::string &group,
                                             const std::string &name,
                                             const TensorView &tensor) {
  if (tensor.device() != Device::CPU || tensor.dtype() != DType::F32) {
    return;
  }
  const ProbeStats s = probe_stats_f32(tensor);
  std::ostringstream oss;
  oss << "[TRAINING][STEP_COMPLETE] step=0 val=0.000000 [probe] "
      << group << "." << name
      << " shape=[" << tensor.shape().r << "," << tensor.shape().c << "]"
      << std::fixed << std::setprecision(6)
      << " mean=" << s.mean
      << " std=" << s.stddev
      << " min=" << s.min
      << " max=" << s.max
      << " hash=0x" << std::hex << s.hash << std::dec;
  report(ReportEvent::STEP_COMPLETE, oss.str());
}

void TrainingReportSink::report_probe_loss(const TensorView &loss_scalar,
                                           const TensorView &logits,
                                           const TensorView &targets) {
  if (loss_scalar.device() != Device::CPU || loss_scalar.dtype() != DType::F32 ||
      logits.device() != Device::CPU || logits.dtype() != DType::F32) {
    return;
  }
  const ProbeStats logits_stats = probe_stats_f32(logits);
  std::ostringstream oss;
  oss << "[TRAINING][STEP_COMPLETE] step=0 val=0.000000 [probe] loss"
      << " scalar=" << std::fixed << std::setprecision(6) << loss_scalar.at_f32(0, 0)
      << " logits_shape=[" << logits.shape().r << "," << logits.shape().c << "]"
      << " logits_mean=" << logits_stats.mean
      << " logits_std=" << logits_stats.stddev
      << " logits_min=" << logits_stats.min
      << " logits_max=" << logits_stats.max
      << " logits_hash=0x" << std::hex << logits_stats.hash << std::dec
      << " targets_shape=[" << targets.shape().r << "," << targets.shape().c << "]";
  report(ReportEvent::STEP_COMPLETE, oss.str());
}

void TrainingReportSink::init_tensors_X_Y(int64_t x_rows, int64_t x_cols,
                                          int64_t y_rows, int64_t y_cols,
                                          const TensorView &tok_emb,
                                          const TensorView &pos_emb) {
  if (init_tensors_report_count_ >= init_tensors_report_limit_) {
    return;
  }
  init_tensors_report_count_ += 1;

  std::ostringstream oss;
  oss << "[TRAINING][START] init_tensors_X_Y: X=[" << x_rows << "x" << x_cols
      << "], Y=[" << y_rows << "x" << y_cols << "]";
  report(ReportEvent::START, oss.str());

  if (tok_emb.device() != Device::CPU || pos_emb.device() != Device::CPU ||
      tok_emb.dtype() != DType::F32 || pos_emb.dtype() != DType::F32) {
    report(ReportEvent::PROGRESS,
           "[TRAINING][PROGRESS] init_tensors_X_Y: tok_emb/pos_emb preview "
           "requires CPU F32.");
    return;
  }
  /*
  const int64_t tok_limit = std::min<int64_t>(50, tok_emb.shape().r);
  report(ReportEvent::PROGRESS,
         "[TRAINING][PROGRESS] tok_emb first " + std::to_string(tok_limit) +
             " positions (v[0..3]):");
  for (int64_t i = 0; i < tok_limit; ++i) {
    std::ostringstream row;
    row << "  tok_emb[" << i << "] = [";
    const int64_t take = std::min<int64_t>(4, tok_emb.shape().c);
    for (int64_t j = 0; j < take; ++j) {
      if (j != 0) {
        row << ", ";
      }
      row << tok_emb.at_f32(i, j);
    }
    if (tok_emb.shape().c > take) {
      row << ", ...";
    }
    row << "]";
    report(ReportEvent::PROGRESS, row.str());
  }

  const int64_t pos_limit = std::min<int64_t>(50, pos_emb.shape().r);
  report(ReportEvent::PROGRESS,
         "[TRAINING][PROGRESS] pos_emb first " + std::to_string(pos_limit) +
             " positions (v[0..3]):");
  for (int64_t i = 0; i < pos_limit; ++i) {
    std::ostringstream row;
    row << "  pos_emb[" << i << "] = [";
    const int64_t take = std::min<int64_t>(4, pos_emb.shape().c);
    for (int64_t j = 0; j < take; ++j) {
      if (j != 0) {
        row << ", ";
      }
      row << pos_emb.at_f32(i, j);
    }
    if (pos_emb.shape().c > take) {
      row << ", ...";
    }
    row << "]";
    report(ReportEvent::PROGRESS, row.str());
  }
    */
}

void TrainingReportSink::report_training_start(const Config &cfg) {
  std::ostringstream oss;
  oss << "Training hyperparams: "
      << "n_layers=" << cfg.model.n_layers
      << ", n_heads=" << cfg.model.n_heads
      << ", d_model=" << cfg.model.d_model
      << ", d_ff=" << cfg.model.d_ff
      << ", vocab_size(actual)=" << cfg.model.target_vocab_size
      << ", lr=" << cfg.training.learning_rate
      << ", beta1=" << cfg.training.beta1
      << ", beta2=" << cfg.training.beta2
      << ", eps=" << cfg.training.eps
      << ", weight_decay=" << cfg.training.weight_decay
      << ", grad_clip=" << cfg.training.grad_clip
      << ", train_epochs=" << cfg.training.num_epochs_train
      << ", dry_run_epochs=" << cfg.training.num_epochs_dry_run
      << ", batch_size=" << cfg.training.batch_size
      << ", window_training=" << cfg.training.window_training
      << ", window_capacity=" << cfg.model.window_capacity;
  report(ReportEvent::START, "[TRAINING][START] " + oss.str());
}

void TrainingReportSink::report_epoch_complete(uint32_t epoch, float mean_loss) {
  std::ostringstream oss;
  oss << "[TRAINING][STEP_COMPLETE] Epoch " << epoch << " mean_loss=" << mean_loss;
  report(ReportEvent::STEP_COMPLETE, oss.str());
}

void TrainingReportSink::report_training_end(uint32_t global_step) {
  report(ReportEvent::END, "[TRAINING][END] Training complete at global_step=" +
                               std::to_string(global_step));
}

void TrainingReportSink::report_error(const std::string &message) {
  report(ReportEvent::ERROR, "[TRAINING][ERROR] " + message);
}

void TrainingReportSink::report_fetch(const TrainingFetchReportData &data) {
  std::ostringstream oss;
  oss << "[TRAINING][FETCH] start_index=" << data.start_index
      << " batch_size=" << data.batch_size
      << " seq_len=" << data.seq_len
      << " block_span=" << data.total_tokens << "\n";
  report(ReportEvent::PROGRESS, oss.str());
}

void TrainingReportSink::report_batch_step(uint32_t batch_cfg, uint32_t window_cfg,
                                           uint32_t batch_flat_t,
                                           uint32_t logit_dim_v) {
  if (batch_step_report_count_ >= batch_step_report_limit_) {
    return;
  }
  std::ostringstream oss;
  oss << "[TRAINING][BATCH] Batch->Logit: batch_size=" << batch_cfg
      << " window_training=" << window_cfg
      << " token_rows=" << batch_flat_t
      << " vocab_size=" << logit_dim_v << "\n";
  report(ReportEvent::STEP_COMPLETE, oss.str());
  batch_step_report_count_ += 1;
}

void TrainingReportSink::report_optimizer_state(
    int phase, const std::string &name, const TensorView &param,
    const TensorView &grad, const TensorView &m, const TensorView &v,
    uint64_t step, bool use_weight_decay) {
  std::ostringstream oss;
  oss << "[TRAINING][PROGRESS] step=" << step
      << " [OPTDBG] phase=" << phase
      << " name=" << name
      << " shape=" << param.shape().r << "x" << param.shape().c
      << " use_wd=" << (use_weight_decay ? 1 : 0)
      << " grad_norm=" << l2_norm_f32_cpu(grad)
      << " param_norm=" << l2_norm_f32_cpu(param)
      << " m_norm=" << l2_norm_f32_cpu(m)
      << " v_norm=" << l2_norm_f32_cpu(v)
      << " grad_finite=" << (all_finite_f32_cpu(grad) ? 1 : 0)
      << " param_finite=" << (all_finite_f32_cpu(param) ? 1 : 0)
      << " m_finite=" << (all_finite_f32_cpu(m) ? 1 : 0)
      << " v_finite=" << (all_finite_f32_cpu(v) ? 1 : 0);
  report(ReportEvent::PROGRESS, oss.str());
}

void TrainingReportSink::report_init_config(const Config &cfg,
                                            const NamedLayout &param_layout,
                                            const NamedLayout &temp_layout) {
  verbose_init_enabled_ = cfg.reporting.verbose_init;
  alignment_bytes_ = cfg.memory.alignment_bytes;
  if (!verbose_init_enabled_) {
    return;
  }

  uint64_t used_param_bytes = 0;
  for (const auto &slice : param_layout.slices()) {
    used_param_bytes += slice.bytes;
  }
  uint64_t used_temp_bytes = 0;
  for (const auto &slice : temp_layout.slices()) {
    used_temp_bytes += slice.bytes;
  }
  const uint64_t total_param_bytes = param_layout.total_bytes();
  const uint64_t total_temp_bytes = temp_layout.total_bytes();
  const uint64_t param_waste_bytes =
      (total_param_bytes >= used_param_bytes) ? (total_param_bytes - used_param_bytes) : 0;
  const uint64_t temp_waste_bytes =
      (total_temp_bytes >= used_temp_bytes) ? (total_temp_bytes - used_temp_bytes) : 0;
  const uint64_t total_adam_bytes = total_param_bytes * 2;
  const size_t param_slice_count = param_layout.slices().size();
  const size_t temp_slice_count = temp_layout.slices().size();

  const double param_mb =
      static_cast<double>(total_param_bytes) / (1024.0 * 1024.0);
  const double adam_mb = static_cast<double>(total_adam_bytes) / (1024.0 * 1024.0);
  const double temp_mb =
      static_cast<double>(total_temp_bytes) / (1024.0 * 1024.0);

  std::ostringstream oss;
  oss << "\n[Memory Topography] enabled by `reporting.verbose_init`\n";
  oss << "+--------------------------------------+----------------+\n";
  oss << "| Metric                               | Value          |\n";
  oss << "+--------------------------------------+----------------+\n";
  oss << "| Parameter Arena (MB)                 | " << std::fixed
      << std::setprecision(2) << std::setw(14) << param_mb << " |\n";
  oss << "| Adam Optimizer Arena (MB)            | " << std::fixed
      << std::setprecision(2) << std::setw(14) << adam_mb << " |\n";
  oss << "| Temp Arena (MB)                      | " << std::fixed
      << std::setprecision(2) << std::setw(14) << temp_mb << " |\n";
  oss << "| Param Padding Waste (bytes)          | " << std::setw(14)
      << param_waste_bytes << " |\n";
  oss << "| Temp Padding Waste (bytes)           | " << std::setw(14)
      << temp_waste_bytes << " |\n";
  oss << "| Param Tensor Slices Mapped           | " << std::setw(14)
      << param_slice_count << " |\n";
  oss << "| Temp Tensor Slices Mapped            | " << std::setw(14)
      << temp_slice_count << " |\n";
  oss << "| Alignment Bytes                      | " << std::setw(14)
      << alignment_bytes_ << " |\n";
  oss << "+--------------------------------------+----------------+\n";
  oss << "Adam uses exactly 2x parameter memory because it keeps two moments\n"
         "(m and v) for every trainable weight.\n";
  report(ReportEvent::START, oss.str());
}

void TrainingReportSink::report_init_topology(const NamedLayout &param_layout,
                                              void *param_base,
                                              uint64_t param_size,
                                              void *adam_base,
                                              uint64_t adam_size,
                                              void *temp_base,
                                              uint64_t temp_size) {
  if (!verbose_init_enabled_) {
    return;
  }

  const uint64_t total_model_memory = param_size + adam_size + temp_size;
  uint64_t sum_param_tensors = 0;
  for (const auto &slice : param_layout.slices()) {
    sum_param_tensors += slice.bytes;
  }
  const uint64_t param_padding_waste =
      (param_size >= sum_param_tensors) ? (param_size - sum_param_tensors) : 0;

  const std::uintptr_t p0 = reinterpret_cast<std::uintptr_t>(param_base);
  const std::uintptr_t p1 = (param_size == 0) ? p0 : (p0 + param_size - 1);
  const std::uintptr_t a0 = reinterpret_cast<std::uintptr_t>(adam_base);
  const std::uintptr_t a1 = (adam_size == 0) ? a0 : (a0 + adam_size - 1);
  const std::uintptr_t t0 = reinterpret_cast<std::uintptr_t>(temp_base);
  const std::uintptr_t t1 = (temp_size == 0) ? t0 : (t0 + temp_size - 1);

  const double param_mb = static_cast<double>(param_size) / (1024.0 * 1024.0);
  const double adam_mb = static_cast<double>(adam_size) / (1024.0 * 1024.0);
  const double temp_mb = static_cast<double>(temp_size) / (1024.0 * 1024.0);
  const double total_mb =
      static_cast<double>(total_model_memory) / (1024.0 * 1024.0);

  std::ostringstream oss;
  oss << "\n[Memory Topology] enabled by `reporting.verbose_init`\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "| Region           | Hex Range                                 | Size (MB)  |\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "| Parameter Arena  | 0x" << std::hex << p0 << " - 0x" << p1
      << std::dec << " | " << std::fixed << std::setprecision(2) << std::setw(10)
      << param_mb << " |\n";
  oss << "| Optimizer Arena  | 0x" << std::hex << a0 << " - 0x" << a1
      << std::dec << " | " << std::fixed << std::setprecision(2) << std::setw(10)
      << adam_mb << " |\n";
  oss << "| Temp Arena       | 0x" << std::hex << t0 << " - 0x" << t1
      << std::dec << " | " << std::fixed << std::setprecision(2) << std::setw(10)
      << temp_mb << " |\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "| Total Model Mem  |                                           | "
      << std::setw(10) << total_mb << " |\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "Alignment bytes: " << alignment_bytes_
      << ", param padding waste=" << param_padding_waste
      << " bytes, param slices mapped=" << param_layout.slices().size() << "\n\n";
  oss << "Pointer Arithmetic: TensorFactory treats arena base pointers as anchors and\n"
         "adds each LayoutSlice offset to address every tensor block directly.\n";
  oss << "Adam Overhead: Optimizer memory is 2x parameter memory because Adam keeps\n"
         "two state tensors per weight (m and v).\n";
  oss << "Contiguity Benefit: One large contiguous arena improves spatial locality\n"
         "and CPU prefetch efficiency versus many scattered allocations.\n";
  report(ReportEvent::START, oss.str());
}

void TrainingReportSink::report_tensor_factory_topology(
    const Config &cfg, const TensorFactory &tensor_factory) {
  if (!verbose_init_enabled_) {
    return;
  }

  std::vector<std::string> lines;
  lines.push_back("[TensorFactory Parameters] enabled by `reporting.verbose_init`");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back("| Name               | Shape   | Layout                | Stride | Address        | Bytes   | Purpose                      |");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back(tensor_metadata_row("tok_embedding", tensor_factory.param_tok_embedding(),
                                      infer_purpose("tok_embedding")));
  lines.push_back(tensor_metadata_row("pos_embedding", tensor_factory.param_pos_embedding(),
                                      infer_purpose("pos_embedding")));
  for (uint32_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";
    lines.push_back(tensor_metadata_row(p + "ln1_gamma",
                                        tensor_factory.param_ln1_gamma(static_cast<int>(l)),
                                        infer_purpose(p + "ln1_gamma")));
    lines.push_back(tensor_metadata_row(p + "ln1_beta",
                                        tensor_factory.param_ln1_beta(static_cast<int>(l)),
                                        infer_purpose(p + "ln1_beta")));
    lines.push_back(tensor_metadata_row(p + "attn_qkv_w",
                                        tensor_factory.param_attn_qkv_w(static_cast<int>(l)),
                                        infer_purpose(p + "attn_qkv_w")));
    lines.push_back(tensor_metadata_row(p + "attn_qkv_b",
                                        tensor_factory.param_attn_qkv_b(static_cast<int>(l)),
                                        infer_purpose(p + "attn_qkv_b")));
    lines.push_back(tensor_metadata_row(p + "attn_out_w",
                                        tensor_factory.param_attn_out_w(static_cast<int>(l)),
                                        infer_purpose(p + "attn_out_w")));
    lines.push_back(tensor_metadata_row(p + "attn_out_b",
                                        tensor_factory.param_attn_out_b(static_cast<int>(l)),
                                        infer_purpose(p + "attn_out_b")));
    lines.push_back(tensor_metadata_row(p + "ln2_gamma",
                                        tensor_factory.param_ln2_gamma(static_cast<int>(l)),
                                        infer_purpose(p + "ln2_gamma")));
    lines.push_back(tensor_metadata_row(p + "ln2_beta",
                                        tensor_factory.param_ln2_beta(static_cast<int>(l)),
                                        infer_purpose(p + "ln2_beta")));
    lines.push_back(tensor_metadata_row(p + "ffn_w1",
                                        tensor_factory.param_ffn_w1(static_cast<int>(l)),
                                        infer_purpose(p + "ffn_w1")));
    lines.push_back(tensor_metadata_row(p + "ffn_b1",
                                        tensor_factory.param_ffn_b1(static_cast<int>(l)),
                                        infer_purpose(p + "ffn_b1")));
    lines.push_back(tensor_metadata_row(p + "ffn_w2",
                                        tensor_factory.param_ffn_w2(static_cast<int>(l)),
                                        infer_purpose(p + "ffn_w2")));
    lines.push_back(tensor_metadata_row(p + "ffn_b2",
                                        tensor_factory.param_ffn_b2(static_cast<int>(l)),
                                        infer_purpose(p + "ffn_b2")));
  }
  lines.push_back(tensor_metadata_row("lnf_gamma", tensor_factory.param_lnf_gamma(),
                                      infer_purpose("lnf_gamma")));
  lines.push_back(tensor_metadata_row("lnf_beta", tensor_factory.param_lnf_beta(),
                                      infer_purpose("lnf_beta")));
  lines.push_back(tensor_metadata_row("lm_head_w", tensor_factory.param_lm_head_w(),
                                      infer_purpose("lm_head_w")));
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");

  const int64_t T = static_cast<int64_t>(std::max<uint64_t>(
      static_cast<uint64_t>(cfg.training.batch_size) *
          static_cast<uint64_t>(cfg.training.window_training),
      cfg.model.window_capacity));
  const int64_t S = static_cast<int64_t>(cfg.model.window_capacity);

  lines.push_back("");
  lines.push_back("[TensorFactory Temporaries] enabled by `reporting.verbose_init`");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back("| Name               | Shape   | Layout                | Stride | Address        | Bytes   | Purpose                      |");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back(tensor_metadata_row("ds.ids", tensor_factory.temp_ds_ids(T), infer_temp_purpose("ds.ids")));
  lines.push_back(tensor_metadata_row("ds.targets", tensor_factory.temp_ds_targets(T), infer_temp_purpose("ds.targets")));
  lines.push_back(tensor_metadata_row("infer.ids", tensor_factory.temp_infer_ids(S), infer_temp_purpose("infer.ids")));
  lines.push_back(tensor_metadata_row("infer.logits", tensor_factory.temp_infer_logits(S), infer_temp_purpose("infer.logits")));
  lines.push_back(tensor_metadata_row("tr.logits", tensor_factory.temp_tr_logits(T), infer_temp_purpose("tr.logits")));
  lines.push_back(tensor_metadata_row("tr.loss", tensor_factory.temp_tr_loss(), infer_temp_purpose("tr.loss")));
  lines.push_back(tensor_metadata_row("tr.X", tensor_factory.temp_tr_X(T), infer_temp_purpose("tr.X")));
  lines.push_back(tensor_metadata_row("tr.Y", tensor_factory.temp_tr_Y(T), infer_temp_purpose("tr.Y")));
  lines.push_back(tensor_metadata_row("tr.Xn", tensor_factory.temp_tr_Xn(T), infer_temp_purpose("tr.Xn")));
  lines.push_back(tensor_metadata_row("bw.XnT", tensor_factory.temp_bw_XnT(T), infer_temp_purpose("bw.XnT")));
  lines.push_back(tensor_metadata_row("bw.d_lm_w", tensor_factory.temp_bw_d_lm_w(T), infer_temp_purpose("bw.d_lm_w")));
  lines.push_back(tensor_metadata_row("bw.lm_wT", tensor_factory.temp_bw_lm_wT(), infer_temp_purpose("bw.lm_wT")));
  lines.push_back(tensor_metadata_row("bw.d_xn", tensor_factory.temp_bw_d_xn(T), infer_temp_purpose("bw.d_xn")));
  lines.push_back(tensor_metadata_row("bw.d_xlast", tensor_factory.temp_bw_d_xlast(T), infer_temp_purpose("bw.d_xlast")));
  lines.push_back(tensor_metadata_row("bw.d_lnf_g", tensor_factory.temp_bw_d_lnf_g(), infer_temp_purpose("bw.d_lnf_g")));
  lines.push_back(tensor_metadata_row("bw.d_lnf_b", tensor_factory.temp_bw_d_lnf_b(), infer_temp_purpose("bw.d_lnf_b")));
  lines.push_back(tensor_metadata_row("bw.d_tok", tensor_factory.temp_bw_d_tok(), infer_temp_purpose("bw.d_tok")));
  lines.push_back(tensor_metadata_row("bw.d_pos", tensor_factory.temp_bw_d_pos(), infer_temp_purpose("bw.d_pos")));

  for (uint32_t l = 0; l < cfg.model.n_layers; ++l) {
    const int li = static_cast<int>(l);
    const std::string p = "layer" + std::to_string(l) + ".";
    lines.push_back(tensor_metadata_row(p + "ln1", tensor_factory.temp_layer_ln1(li, T), infer_temp_purpose(p + "ln1")));
    lines.push_back(tensor_metadata_row(p + "attn_out", tensor_factory.temp_layer_attn_out(li, T), infer_temp_purpose(p + "attn_out")));
    lines.push_back(tensor_metadata_row(p + "resid1", tensor_factory.temp_layer_resid1(li, T), infer_temp_purpose(p + "resid1")));
    lines.push_back(tensor_metadata_row(p + "ln2", tensor_factory.temp_layer_ln2(li, T), infer_temp_purpose(p + "ln2")));
    lines.push_back(tensor_metadata_row(p + "ffn_out", tensor_factory.temp_layer_ffn_out(li, T), infer_temp_purpose(p + "ffn_out")));
    lines.push_back(tensor_metadata_row(p + "bw.d_prev", tensor_factory.temp_layer_bw_d_prev(li, T), infer_temp_purpose(p + "bw.d_prev")));
    lines.push_back(tensor_metadata_row(p + "dln2", tensor_factory.temp_layer_dln2(li, T), infer_temp_purpose(p + "dln2")));
    lines.push_back(tensor_metadata_row(p + "dy_ln2", tensor_factory.temp_layer_dy_ln2(li, T), infer_temp_purpose(p + "dy_ln2")));
    lines.push_back(tensor_metadata_row(p + "dln2_gamma", tensor_factory.temp_layer_dln2_gamma(li), infer_temp_purpose(p + "dln2_gamma")));
    lines.push_back(tensor_metadata_row(p + "dln2_beta", tensor_factory.temp_layer_dln2_beta(li), infer_temp_purpose(p + "dln2_beta")));
    lines.push_back(tensor_metadata_row(p + "dy_total", tensor_factory.temp_layer_dy_total(li, T), infer_temp_purpose(p + "dy_total")));
    lines.push_back(tensor_metadata_row(p + "dln1", tensor_factory.temp_layer_dln1(li, T), infer_temp_purpose(p + "dln1")));
    lines.push_back(tensor_metadata_row(p + "dx_ln1", tensor_factory.temp_layer_dx_ln1(li, T), infer_temp_purpose(p + "dx_ln1")));
    lines.push_back(tensor_metadata_row(p + "dln1_gamma", tensor_factory.temp_layer_dln1_gamma(li), infer_temp_purpose(p + "dln1_gamma")));
    lines.push_back(tensor_metadata_row(p + "dln1_beta", tensor_factory.temp_layer_dln1_beta(li), infer_temp_purpose(p + "dln1_beta")));

    lines.push_back(tensor_metadata_row(p + "attn.qkv", tensor_factory.temp_attn_qkv(li, T), infer_temp_purpose("attn.qkv")));
    lines.push_back(tensor_metadata_row(p + "attn.context", tensor_factory.temp_attn_context(li, T), infer_temp_purpose("attn.context")));
    lines.push_back(tensor_metadata_row(p + "attn.scores", tensor_factory.temp_attn_scores(li, T), infer_temp_purpose("attn.scores")));
    lines.push_back(tensor_metadata_row(p + "attn.weights", tensor_factory.temp_attn_weights(li, T), infer_temp_purpose("attn.weights")));
    lines.push_back(tensor_metadata_row(p + "attn.head", tensor_factory.temp_attn_head(li, T), infer_temp_purpose("attn.head")));
    lines.push_back(tensor_metadata_row(p + "attn.contextT", tensor_factory.temp_attn_contextT(li, T), infer_temp_purpose("attn.contextT")));
    lines.push_back(tensor_metadata_row(p + "attn.dWo", tensor_factory.temp_attn_dWo(li), infer_temp_purpose("attn.dWo")));
    lines.push_back(tensor_metadata_row(p + "attn.dbo", tensor_factory.temp_attn_dbo(li), infer_temp_purpose("attn.dbo")));
    lines.push_back(tensor_metadata_row(p + "attn.WoT", tensor_factory.temp_attn_WoT(li), infer_temp_purpose("attn.WoT")));
    lines.push_back(tensor_metadata_row(p + "attn.dcontext", tensor_factory.temp_attn_dcontext(li, T), infer_temp_purpose("attn.dcontext")));
    lines.push_back(tensor_metadata_row(p + "attn.dqkv", tensor_factory.temp_attn_dqkv(li, T), infer_temp_purpose("attn.dqkv")));
    lines.push_back(tensor_metadata_row(p + "attn.KhT", tensor_factory.temp_attn_KhT(li, T), infer_temp_purpose("attn.KhT")));
    lines.push_back(tensor_metadata_row(p + "attn.VhT", tensor_factory.temp_attn_VhT(li, T), infer_temp_purpose("attn.VhT")));
    lines.push_back(tensor_metadata_row(p + "attn.dweights", tensor_factory.temp_attn_dweights(li, T), infer_temp_purpose("attn.dweights")));
    lines.push_back(tensor_metadata_row(p + "attn.weightsT", tensor_factory.temp_attn_weightsT(li, T), infer_temp_purpose("attn.weightsT")));
    lines.push_back(tensor_metadata_row(p + "attn.dscores", tensor_factory.temp_attn_dscores(li, T), infer_temp_purpose("attn.dscores")));
    lines.push_back(tensor_metadata_row(p + "attn.dscoresT", tensor_factory.temp_attn_dscoresT(li, T), infer_temp_purpose("attn.dscoresT")));
    lines.push_back(tensor_metadata_row(p + "attn.WqkvT", tensor_factory.temp_attn_WqkvT(li), infer_temp_purpose("attn.WqkvT")));
    lines.push_back(tensor_metadata_row(p + "attn.xT", tensor_factory.temp_attn_xT(li, T), infer_temp_purpose("attn.xT")));
    lines.push_back(tensor_metadata_row(p + "attn.dWqkv", tensor_factory.temp_attn_dWqkv(li), infer_temp_purpose("attn.dWqkv")));
    lines.push_back(tensor_metadata_row(p + "attn.dbqkv", tensor_factory.temp_attn_dbqkv(li), infer_temp_purpose("attn.dbqkv")));

    lines.push_back(tensor_metadata_row(p + "ffn.h", tensor_factory.temp_ffn_h(li, T), infer_temp_purpose("ffn.h")));
    lines.push_back(tensor_metadata_row(p + "ffn.a", tensor_factory.temp_ffn_a(li, T), infer_temp_purpose("ffn.a")));
    lines.push_back(tensor_metadata_row(p + "ffn.aT", tensor_factory.temp_ffn_aT(li, T), infer_temp_purpose("ffn.aT")));
    lines.push_back(tensor_metadata_row(p + "ffn.dW2", tensor_factory.temp_ffn_dW2(li), infer_temp_purpose("ffn.dW2")));
    lines.push_back(tensor_metadata_row(p + "ffn.db2", tensor_factory.temp_ffn_db2(li), infer_temp_purpose("ffn.db2")));
    lines.push_back(tensor_metadata_row(p + "ffn.W2T", tensor_factory.temp_ffn_W2T(li), infer_temp_purpose("ffn.W2T")));
    lines.push_back(tensor_metadata_row(p + "ffn.da", tensor_factory.temp_ffn_da(li, T), infer_temp_purpose("ffn.da")));
    lines.push_back(tensor_metadata_row(p + "ffn.dh", tensor_factory.temp_ffn_dh(li, T), infer_temp_purpose("ffn.dh")));
    lines.push_back(tensor_metadata_row(p + "ffn.xT", tensor_factory.temp_ffn_xT(li, T), infer_temp_purpose("ffn.xT")));
    lines.push_back(tensor_metadata_row(p + "ffn.dW1", tensor_factory.temp_ffn_dW1(li), infer_temp_purpose("ffn.dW1")));
    lines.push_back(tensor_metadata_row(p + "ffn.db1", tensor_factory.temp_ffn_db1(li), infer_temp_purpose("ffn.db1")));
    lines.push_back(tensor_metadata_row(p + "ffn.W1T", tensor_factory.temp_ffn_W1T(li), infer_temp_purpose("ffn.W1T")));
  }
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");

  for (const auto &line : lines) {
    report(ReportEvent::PROGRESS, line);
  }
  report(ReportEvent::PROGRESS, "");
}
