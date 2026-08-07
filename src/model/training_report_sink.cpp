#include "training_report_sink.hpp"

#include <config.hpp>
#include "named_layout.hpp"
#include "tensor_factory.hpp"

#include <iomanip>
#include <sstream>

namespace {
const char *layout_type(const TensorView &t) {
  if (t.is_contiguous_row_major()) {
    return "Contiguous Row-Major";
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

std::string tensor_metadata_row(const std::string &name, const TensorView &tv) {
  std::ostringstream addr;
  addr << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(tv.data());
  const std::string shape =
      std::to_string(tv.shape().r) + "x" + std::to_string(tv.shape().c);
  const std::string stride =
      std::to_string(tv.stride_r_bytes()) + "/" + std::to_string(tv.stride_c_bytes());

  std::ostringstream line;
  line << "| " << std::left << std::setw(26) << name
       << " | " << std::setw(7) << shape
       << " | " << std::setw(21) << layout_type(tv)
       << " | " << std::setw(17) << stride
       << " | " << std::setw(14) << addr.str()
       << " | " << std::right << std::setw(7) << tv.bytes()
       << " | " << std::left << infer_purpose(name);
  return line.str();
}
} // namespace

TrainingReportSink::TrainingReportSink(const LoggingConfig &logging)
    : console_(logging) {}

void TrainingReportSink::report(ReportEvent event, const std::string &message) {
  console_.report(event, message);
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
  oss << "[TRAINING][BATCH] Batch->Logit: B_cfg=" << batch_cfg
      << " T_cfg=" << window_cfg << " T_flat=" << batch_flat_t
      << " V=" << logit_dim_v << "\n";
  report(ReportEvent::STEP_COMPLETE, oss.str());
  batch_step_report_count_ += 1;
}

void TrainingReportSink::report_init_config(const Config &cfg,
                                            const NamedLayout &layout) {
  verbose_init_enabled_ = cfg.reporting.verbose_init;
  alignment_bytes_ = cfg.memory.alignment_bytes;
  if (!verbose_init_enabled_) {
    return;
  }

  uint64_t used_bytes = 0;
  for (const auto &slice : layout.slices()) {
    used_bytes += slice.bytes;
  }
  const uint64_t total_param_bytes = layout.total_bytes();
  const uint64_t waste_bytes =
      (total_param_bytes >= used_bytes) ? (total_param_bytes - used_bytes) : 0;
  const uint64_t total_adam_bytes = total_param_bytes * 2;
  const size_t slice_count = layout.slices().size();

  const double param_mb =
      static_cast<double>(total_param_bytes) / (1024.0 * 1024.0);
  const double adam_mb = static_cast<double>(total_adam_bytes) / (1024.0 * 1024.0);

  std::ostringstream oss;
  oss << "\n[Memory Topography]\n";
  oss << "+--------------------------------------+----------------+\n";
  oss << "| Metric                               | Value          |\n";
  oss << "+--------------------------------------+----------------+\n";
  oss << "| Parameter Arena (MB)                 | " << std::fixed
      << std::setprecision(2) << std::setw(14) << param_mb << " |\n";
  oss << "| Adam Optimizer Arena (MB)            | " << std::fixed
      << std::setprecision(2) << std::setw(14) << adam_mb << " |\n";
  oss << "| Alignment Padding Waste (bytes)      | " << std::setw(14) << waste_bytes
      << " |\n";
  oss << "| Unique Tensor Slices Mapped          | " << std::setw(14) << slice_count
      << " |\n";
  oss << "+--------------------------------------+----------------+\n";
  oss << "Adam uses exactly 2x parameter memory because it keeps two moments\n"
         "(m and v) for every trainable weight.\n";
  report(ReportEvent::START, oss.str());
}

void TrainingReportSink::report_init_topology(const NamedLayout &layout,
                                              void *param_base,
                                              uint64_t param_size,
                                              void *adam_base,
                                              uint64_t adam_size) {
  if (!verbose_init_enabled_) {
    return;
  }

  const uint64_t total_model_memory = param_size + adam_size;
  uint64_t sum_tensors = 0;
  for (const auto &slice : layout.slices()) {
    sum_tensors += slice.bytes;
  }
  const uint64_t padding_waste =
      (param_size >= sum_tensors) ? (param_size - sum_tensors) : 0;

  const std::uintptr_t p0 = reinterpret_cast<std::uintptr_t>(param_base);
  const std::uintptr_t p1 = (param_size == 0) ? p0 : (p0 + param_size - 1);
  const std::uintptr_t a0 = reinterpret_cast<std::uintptr_t>(adam_base);
  const std::uintptr_t a1 = (adam_size == 0) ? a0 : (a0 + adam_size - 1);

  const double param_mb = static_cast<double>(param_size) / (1024.0 * 1024.0);
  const double adam_mb = static_cast<double>(adam_size) / (1024.0 * 1024.0);
  const double total_mb =
      static_cast<double>(total_model_memory) / (1024.0 * 1024.0);

  std::ostringstream oss;
  oss << "\n[Memory Topology]\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "| Region           | Hex Range                                 | Size (MB)  |\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "| Parameter Arena  | 0x" << std::hex << p0 << " - 0x" << p1
      << std::dec << " | " << std::fixed << std::setprecision(2) << std::setw(10)
      << param_mb << " |\n";
  oss << "| Optimizer Arena  | 0x" << std::hex << a0 << " - 0x" << a1
      << std::dec << " | " << std::fixed << std::setprecision(2) << std::setw(10)
      << adam_mb << " |\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "| Total Model Mem  |                                           | "
      << std::setw(10) << total_mb << " |\n";
  oss << "+------------------+-------------------------------------------+------------+\n";
  oss << "Alignment bytes: " << alignment_bytes_
      << ", padding waste=" << padding_waste
      << " bytes, slices mapped=" << layout.slices().size() << "\n\n";
  oss << "Pointer Arithmetic: TensorFactory treats arena base pointers as anchors and\n"
         "adds each LayoutSlice offset to address every tensor block directly.\n";
  oss << "Adam Overhead: Optimizer memory is 2x parameter memory because Adam keeps\n"
         "two state tensors per weight (m and v).\n";
  oss << "Contiguity Benefit: One large contiguous arena improves spatial locality\n"
         "and CPU prefetch efficiency versus many scattered allocations.\n";
  report(ReportEvent::START, oss.str());
}

void TrainingReportSink::report_tensor_factory_topology(
    const Config &cfg, const TensorFactory &tensors) {
  if (!verbose_init_enabled_) {
    return;
  }

  std::vector<std::string> lines;
  lines.push_back("[TensorFactory Parameters]");
  lines.push_back("+----------------------------+---------+-----------------------+-------------------+----------------+---------+------------------------------+");
  lines.push_back("| Name                       | Shape   | Layout                | Stride(r/c bytes) | Address        | Bytes   | Purpose                      |");
  lines.push_back("+----------------------------+---------+-----------------------+-------------------+----------------+---------+------------------------------+");
  lines.push_back(tensor_metadata_row("tok_embedding", tensors.param_tok_embedding()));
  lines.push_back(tensor_metadata_row("pos_embedding", tensors.param_pos_embedding()));
  for (uint32_t l = 0; l < cfg.model.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";
    lines.push_back(tensor_metadata_row(p + "ln1_gamma", tensors.param_ln1_gamma(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ln1_beta", tensors.param_ln1_beta(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "attn_qkv_w", tensors.param_attn_qkv_w(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "attn_qkv_b", tensors.param_attn_qkv_b(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "attn_out_w", tensors.param_attn_out_w(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "attn_out_b", tensors.param_attn_out_b(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ln2_gamma", tensors.param_ln2_gamma(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ln2_beta", tensors.param_ln2_beta(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ffn_w1", tensors.param_ffn_w1(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ffn_b1", tensors.param_ffn_b1(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ffn_w2", tensors.param_ffn_w2(static_cast<int>(l))));
    lines.push_back(tensor_metadata_row(p + "ffn_b2", tensors.param_ffn_b2(static_cast<int>(l))));
  }
  lines.push_back(tensor_metadata_row("lnf_gamma", tensors.param_lnf_gamma()));
  lines.push_back(tensor_metadata_row("lnf_beta", tensors.param_lnf_beta()));
  lines.push_back(tensor_metadata_row("lm_head_w", tensors.param_lm_head_w()));
  lines.push_back("+----------------------------+---------+-----------------------+-------------------+----------------+---------+------------------------------+");

  for (const auto &line : lines) {
    report(ReportEvent::PROGRESS, line);
  }
  report(ReportEvent::PROGRESS, "");
}
