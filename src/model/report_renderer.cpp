#include "report_renderer.hpp"
#include "memory/adam_state_store.hpp"
#include "memory/named_layout.hpp"

#include <config.hpp>
#include "named_layout.hpp"
#include "tensor_probe.hpp"
#include "tensor_store.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
double bytes_to_mib(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double bytes_to_gib(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

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

bool uses_weight_decay(const std::string &name) {
  if (name == "tok_embedding" || name == "pos_embedding" ||
      name == "lm_head_w") {
    return true;
  }
  if (name.find("attn_qkv_w") != std::string::npos ||
      name.find("attn_out_w") != std::string::npos ||
      name.find("ffn_w1") != std::string::npos ||
      name.find("ffn_w2") != std::string::npos) {
    return true;
  }
  return false;
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
  if (name == "bw.lm_wT") return "Transposed LM head weights";
  if (name == "bw.d_xn" || name == "bw.d_xlast") return "Backward hidden gradients";
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
  const std::string shape = TensorProbe::format_shape(tv.shape_nd());
  const std::string stride = tv.rank() >= 2
                                 ? std::to_string(tv.stride_bytes(tv.rank() - 2)) +
                                       "/" +
                                       std::to_string(tv.stride_bytes(tv.rank() - 1))
                                 : tv.rank() == 1
                                       ? std::string("0/") +
                                             std::to_string(tv.stride_bytes(0))
                                       : "0/0";

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

} // namespace

ReportRenderer::ReportRenderer(std::ostream &out) : out_(out) {}

ReportRenderer::ReportRenderer() : out_(std::cout) {}

void ReportRenderer::print(const std::string &text) { out_ << text << "\n"; }

void ReportRenderer::report_init_config(const Config &cfg,
                                        const TrainingLayouts &layouts) {
  const NamedLayout &param_layout = layouts.param;
  const NamedLayout &temp_layout = layouts.temp;
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
  const uint64_t total_grad_bytes = layouts.grad.total_bytes();
  const uint64_t total_temp_bytes = temp_layout.total_bytes();
  const uint64_t param_waste_bytes =
      (total_param_bytes >= used_param_bytes) ? (total_param_bytes - used_param_bytes) : 0;
  const uint64_t temp_waste_bytes =
      (total_temp_bytes >= used_temp_bytes) ? (total_temp_bytes - used_temp_bytes) : 0;
  const uint64_t total_adam_bytes = AdamStateStore::arena_bytes(layouts.optimizer);
  const size_t param_slice_count = param_layout.slices().size();
  const size_t temp_slice_count = temp_layout.slices().size();

  const double param_mb =
      static_cast<double>(total_param_bytes) / (1024.0 * 1024.0);
  const double grad_mb =
      static_cast<double>(total_grad_bytes) / (1024.0 * 1024.0);
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
  oss << "| Gradient Arena (MB)                  | " << std::fixed
      << std::setprecision(2) << std::setw(14) << grad_mb << " |\n";
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
  oss << "Adam keeps two moments (m and v) for every trainable weight, each in\n"
         "the optimizer-state dtype.\n";
  print(oss.str());
}

void ReportRenderer::report_init_topology(const NamedLayout &param_layout,
                                              void *param_base,
                                              uint64_t param_size,
                                              void *grad_base,
                                              uint64_t grad_size,
                                              void *adam_base,
                                              uint64_t adam_size,
                                              void *temp_base,
                                              uint64_t temp_size) {
  if (!verbose_init_enabled_) {
    return;
  }

  const uint64_t total_model_memory =
      param_size + grad_size + adam_size + temp_size;
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
  const std::uintptr_t g0 = reinterpret_cast<std::uintptr_t>(grad_base);
  const std::uintptr_t g1 = (grad_size == 0) ? g0 : (g0 + grad_size - 1);
  const std::uintptr_t t0 = reinterpret_cast<std::uintptr_t>(temp_base);
  const std::uintptr_t t1 = (temp_size == 0) ? t0 : (t0 + temp_size - 1);

  const double param_mb = static_cast<double>(param_size) / (1024.0 * 1024.0);
  const double grad_mb = static_cast<double>(grad_size) / (1024.0 * 1024.0);
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
  oss << "| Gradient Arena   | 0x" << std::hex << g0 << " - 0x" << g1
      << std::dec << " | " << std::fixed << std::setprecision(2)
      << std::setw(10) << grad_mb << " |\n";
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
  oss << "Pointer Arithmetic: TensorStore treats arena base pointers as anchors and\n"
         "adds each LayoutSlice offset to address every tensor block directly.\n";
  oss << "Adam Overhead: Adam keeps two state tensors per weight (m and v), each\n"
         "in the optimizer-state dtype.\n";
  oss << "Contiguity Benefit: One large contiguous arena improves spatial locality\n"
         "and CPU prefetch efficiency versus many scattered allocations.\n";
  print(oss.str());
}

void ReportRenderer::report_memory_usage(const TrainingMemoryUsage &usage) {
  std::ostringstream oss;
  oss << "\n[Memory Usage]\n";
  oss << "+--------------------------+-------------+\n";
  oss << "| Arena                    | Size (MiB)  |\n";
  oss << "+--------------------------+-------------+\n";
  oss << "| Parameters               | " << std::fixed
      << std::setprecision(2) << std::setw(11)
      << bytes_to_mib(usage.param_bytes) << " |\n";
  oss << "| Gradients                | " << std::fixed
      << std::setprecision(2) << std::setw(11)
      << bytes_to_mib(usage.grad_bytes) << " |\n";
  if (usage.grad_sum_bytes > 0) {
    oss << "| Gradient accumulator     | " << std::fixed
        << std::setprecision(2) << std::setw(11)
        << bytes_to_mib(usage.grad_sum_bytes) << " |\n";
  }
  oss << "| Adam optimizer           | " << std::fixed
      << std::setprecision(2) << std::setw(11)
      << bytes_to_mib(usage.adam_bytes) << " |\n";
  oss << "| Temp                     | " << std::fixed
      << std::setprecision(2) << std::setw(11)
      << bytes_to_mib(usage.temp_bytes) << " |\n";
  oss << "| Total managed            | " << std::fixed
      << std::setprecision(2) << std::setw(11)
      << bytes_to_mib(usage.total_managed_bytes) << " |\n";
  oss << "+--------------------------+-------------+\n";

  if (usage.device_after_alloc.available) {
    const uint64_t before_free = usage.device_before_alloc.available
                                     ? usage.device_before_alloc.free_bytes
                                     : 0;
    const uint64_t after_free = usage.device_after_alloc.free_bytes;
    const uint64_t total = usage.device_after_alloc.total_bytes;
    const uint64_t consumed =
        (usage.device_before_alloc.available && before_free >= after_free)
            ? (before_free - after_free)
            : 0;
    oss << "Device memory: total=" << std::fixed << std::setprecision(2)
        << bytes_to_gib(total) << " GiB";
    if (usage.device_before_alloc.available) {
      oss << ", free_before=" << bytes_to_gib(before_free) << " GiB"
          << ", free_after=" << bytes_to_gib(after_free) << " GiB"
          << ", consumed_by_init=" << bytes_to_mib(consumed) << " MiB";
    } else {
      oss << ", free_after=" << bytes_to_gib(after_free) << " GiB";
    }
    oss << "\n";
  } else {
    oss << "Device memory: unavailable for this backend.\n";
  }
  print(oss.str());
}

void ReportRenderer::report_tensor_store_topology(
    const Config &cfg, const TensorStore &tensor_store) {
  if (!verbose_init_enabled_) {
    return;
  }

  auto append_param_row = [](std::vector<std::string> &decay_rows,
                             std::vector<std::string> &no_decay_rows,
                             const std::string &name, const TensorView &tv) {
    const std::string row = tensor_metadata_row(name, tv, infer_purpose(name));
    if (uses_weight_decay(name)) {
      decay_rows.push_back(row);
    } else {
      no_decay_rows.push_back(row);
    }
  };

  std::vector<std::string> decay_rows;
  std::vector<std::string> no_decay_rows;
  append_param_row(decay_rows, no_decay_rows, "tok_embedding",
                   tensor_store.param_tok_embedding());
  if (tensor_store.has_pos_embedding()) {
    append_param_row(decay_rows, no_decay_rows, "pos_embedding",
                     tensor_store.param_pos_embedding());
  }
  for (uint32_t l = 0; l < cfg.model.n_layers; ++l) {
    const int li = static_cast<int>(l);
    const std::string p = "layer" + std::to_string(l) + ".";
    append_param_row(decay_rows, no_decay_rows, p + "ln1_gamma",
                     tensor_store.param_ln1_gamma(li));
    append_param_row(decay_rows, no_decay_rows, p + "ln1_beta",
                     tensor_store.param_ln1_beta(li));
    append_param_row(decay_rows, no_decay_rows, p + "attn_qkv_w",
                     tensor_store.param_attn_qkv_w(li));
    append_param_row(decay_rows, no_decay_rows, p + "attn_qkv_b",
                     tensor_store.param_attn_qkv_b(li));
    append_param_row(decay_rows, no_decay_rows, p + "attn_out_w",
                     tensor_store.param_attn_out_w(li));
    append_param_row(decay_rows, no_decay_rows, p + "attn_out_b",
                     tensor_store.param_attn_out_b(li));
    append_param_row(decay_rows, no_decay_rows, p + "ln2_gamma",
                     tensor_store.param_ln2_gamma(li));
    append_param_row(decay_rows, no_decay_rows, p + "ln2_beta",
                     tensor_store.param_ln2_beta(li));
    append_param_row(decay_rows, no_decay_rows, p + "ffn_w1",
                     tensor_store.param_ffn_w1(li));
    append_param_row(decay_rows, no_decay_rows, p + "ffn_b1",
                     tensor_store.param_ffn_b1(li));
    append_param_row(decay_rows, no_decay_rows, p + "ffn_w2",
                     tensor_store.param_ffn_w2(li));
    append_param_row(decay_rows, no_decay_rows, p + "ffn_b2",
                     tensor_store.param_ffn_b2(li));
  }
  append_param_row(decay_rows, no_decay_rows, "lnf_gamma",
                   tensor_store.param_lnf_gamma());
  append_param_row(decay_rows, no_decay_rows, "lnf_beta",
                   tensor_store.param_lnf_beta());
  append_param_row(decay_rows, no_decay_rows, "lm_head_w",
                   tensor_store.param_lm_head_w());

  std::vector<std::string> lines;
  lines.push_back("[TensorStore Parameters] enabled by `reporting.verbose_init`");
  lines.push_back("[Decay Zone]");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back("| Name               | Shape   | Layout                | Stride | Address        | Bytes   | Purpose                      |");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.insert(lines.end(), decay_rows.begin(), decay_rows.end());
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back("");
  lines.push_back("[No-Decay Zone]");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back("| Name               | Shape   | Layout                | Stride | Address        | Bytes   | Purpose                      |");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.insert(lines.end(), no_decay_rows.begin(), no_decay_rows.end());
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");

  const int64_t T = static_cast<int64_t>(std::max<uint64_t>(
      static_cast<uint64_t>(cfg.training.batch_size) *
          static_cast<uint64_t>(cfg.training.train_seq_len),
      cfg.model.max_seq_len));
  const int64_t S = static_cast<int64_t>(cfg.model.max_seq_len);

  lines.push_back("");
  lines.push_back("[TensorStore Temporaries] enabled by `reporting.verbose_init`");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back("| Name               | Shape   | Layout                | Stride | Address        | Bytes   | Purpose                      |");
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");
  lines.push_back(tensor_metadata_row(
      "ds.ids",
      tensor_store.temp_ds_ids(static_cast<int64_t>(cfg.training.batch_size),
                                 static_cast<int64_t>(cfg.training.train_seq_len)),
      infer_temp_purpose("ds.ids")));
  lines.push_back(tensor_metadata_row(
      "ds.targets",
      tensor_store.temp_ds_targets(static_cast<int64_t>(cfg.training.batch_size),
                                     static_cast<int64_t>(cfg.training.train_seq_len)),
      infer_temp_purpose("ds.targets")));
  lines.push_back(tensor_metadata_row(
      "tr.logits",
      tensor_store.temp_tr_logits(static_cast<int64_t>(cfg.training.batch_size),
                                    static_cast<int64_t>(cfg.training.train_seq_len)),
      infer_temp_purpose("tr.logits")));
  lines.push_back(tensor_metadata_row("tr.loss", tensor_store.temp_tr_loss(), infer_temp_purpose("tr.loss")));
  lines.push_back(tensor_metadata_row(
      "tr.X",
      tensor_store.temp_tr_X(static_cast<int64_t>(cfg.training.batch_size),
                               static_cast<int64_t>(cfg.training.train_seq_len)),
      infer_temp_purpose("tr.X")));
  lines.push_back(tensor_metadata_row(
      "tr.Y",
      tensor_store.temp_tr_Y(static_cast<int64_t>(cfg.training.batch_size),
                               static_cast<int64_t>(cfg.training.train_seq_len)),
      infer_temp_purpose("tr.Y")));
  lines.push_back(tensor_metadata_row(
      "tr.Xn",
      tensor_store.temp_tr_Xn(static_cast<int64_t>(cfg.training.batch_size),
                                static_cast<int64_t>(cfg.training.train_seq_len)),
      infer_temp_purpose("tr.Xn")));
  lines.push_back(tensor_metadata_row("bw.XnT", tensor_store.temp_bw_XnT(T), infer_temp_purpose("bw.XnT")));
  lines.push_back(tensor_metadata_row("bw.lm_wT", tensor_store.temp_bw_lm_wT(), infer_temp_purpose("bw.lm_wT")));
  lines.push_back(tensor_metadata_row("bw.d_xn", tensor_store.temp_bw_d_xn(T), infer_temp_purpose("bw.d_xn")));
  lines.push_back(tensor_metadata_row("bw.d_xlast", tensor_store.temp_bw_d_xlast(T), infer_temp_purpose("bw.d_xlast")));

  for (uint32_t l = 0; l < cfg.model.n_layers; ++l) {
    const int li = static_cast<int>(l);
    const std::string p = "layer" + std::to_string(l) + ".";
    lines.push_back(tensor_metadata_row(
        p + "ln1",
        tensor_store.temp_layer_ln1(li, static_cast<int64_t>(cfg.training.batch_size),
                                      static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose(p + "ln1")));
    lines.push_back(tensor_metadata_row(
        p + "attn_out",
        tensor_store.temp_layer_attn_out(li, static_cast<int64_t>(cfg.training.batch_size),
                                           static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose(p + "attn_out")));
    lines.push_back(tensor_metadata_row(
        p + "resid1",
        tensor_store.temp_layer_resid1(li, static_cast<int64_t>(cfg.training.batch_size),
                                         static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose(p + "resid1")));
    lines.push_back(tensor_metadata_row(
        p + "ln2",
        tensor_store.temp_layer_ln2(li, static_cast<int64_t>(cfg.training.batch_size),
                                      static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose(p + "ln2")));
    lines.push_back(tensor_metadata_row(
        p + "ffn_out",
        tensor_store.temp_layer_ffn_out(li, static_cast<int64_t>(cfg.training.batch_size),
                                          static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose(p + "ffn_out")));
    lines.push_back(tensor_metadata_row(p + "bw.d_prev", tensor_store.temp_layer_bw_d_prev(li, T), infer_temp_purpose(p + "bw.d_prev")));
    lines.push_back(tensor_metadata_row(p + "dln2", tensor_store.temp_layer_dln2(li, T), infer_temp_purpose(p + "dln2")));
    lines.push_back(tensor_metadata_row(p + "dy_ln2", tensor_store.temp_layer_dy_ln2(li, T), infer_temp_purpose(p + "dy_ln2")));
    lines.push_back(tensor_metadata_row(p + "dy_total", tensor_store.temp_layer_dy_total(li, T), infer_temp_purpose(p + "dy_total")));
    lines.push_back(tensor_metadata_row(p + "dln1", tensor_store.temp_layer_dln1(li, T), infer_temp_purpose(p + "dln1")));
    lines.push_back(tensor_metadata_row(p + "dx_ln1", tensor_store.temp_layer_dx_ln1(li, T), infer_temp_purpose(p + "dx_ln1")));

    lines.push_back(tensor_metadata_row(
        p + "attn.qkv",
        tensor_store.temp_attn_qkv(li, static_cast<int64_t>(cfg.training.batch_size),
                                     static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose("attn.qkv")));
    lines.push_back(tensor_metadata_row(
        p + "attn.context",
        tensor_store.temp_attn_context(li, static_cast<int64_t>(cfg.training.batch_size),
                                         static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose("attn.context")));
    lines.push_back(tensor_metadata_row(p + "attn.scores", tensor_store.temp_attn_scores(li, T), infer_temp_purpose("attn.scores")));
    if (tensor_store.algo_factory().attention_needs_weights_buffer()) {
      lines.push_back(tensor_metadata_row(
          p + "attn.weights", tensor_store.temp_attn_weights(li, T),
          infer_temp_purpose("attn.weights")));
    }
    lines.push_back(tensor_metadata_row(
        p + "attn.head",
        tensor_store.temp_attn_head(
            li, static_cast<int64_t>(cfg.training.batch_size),
            static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose("attn.head")));
    lines.push_back(tensor_metadata_row(p + "attn.contextT", tensor_store.temp_attn_contextT(li, T), infer_temp_purpose("attn.contextT")));
    lines.push_back(tensor_metadata_row(p + "attn.WoT", tensor_store.temp_attn_WoT(li), infer_temp_purpose("attn.WoT")));
    lines.push_back(tensor_metadata_row(p + "attn.dcontext", tensor_store.temp_attn_dcontext(li, T), infer_temp_purpose("attn.dcontext")));
    lines.push_back(tensor_metadata_row(p + "attn.dqkv", tensor_store.temp_attn_dqkv(li, T), infer_temp_purpose("attn.dqkv")));
    lines.push_back(tensor_metadata_row(p + "attn.KhT", tensor_store.temp_attn_KhT(li, T), infer_temp_purpose("attn.KhT")));
    lines.push_back(tensor_metadata_row(p + "attn.VhT", tensor_store.temp_attn_VhT(li, T), infer_temp_purpose("attn.VhT")));
    lines.push_back(tensor_metadata_row(p + "attn.dweights", tensor_store.temp_attn_dweights(li, T), infer_temp_purpose("attn.dweights")));
    lines.push_back(tensor_metadata_row(p + "attn.weightsT", tensor_store.temp_attn_weightsT(li, T), infer_temp_purpose("attn.weightsT")));
    lines.push_back(tensor_metadata_row(p + "attn.dscores", tensor_store.temp_attn_dscores(li, T), infer_temp_purpose("attn.dscores")));
    lines.push_back(tensor_metadata_row(p + "attn.dscoresT", tensor_store.temp_attn_dscoresT(li, T), infer_temp_purpose("attn.dscoresT")));
    lines.push_back(tensor_metadata_row(p + "attn.WqkvT", tensor_store.temp_attn_WqkvT(li), infer_temp_purpose("attn.WqkvT")));
    lines.push_back(tensor_metadata_row(p + "attn.xT", tensor_store.temp_attn_xT(li, T), infer_temp_purpose("attn.xT")));

    lines.push_back(tensor_metadata_row(
        p + "ffn.h",
        tensor_store.temp_ffn_h(li, static_cast<int64_t>(cfg.training.batch_size),
                                  static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose("ffn.h")));
    if (tensor_store.algo_factory().ffn_needs_activation_buffers()) {
      lines.push_back(tensor_metadata_row(
          p + "ffn.a",
          tensor_store.temp_ffn_a(
              li, static_cast<int64_t>(cfg.training.batch_size),
              static_cast<int64_t>(cfg.training.train_seq_len)),
          infer_temp_purpose("ffn.a")));
    }
    lines.push_back(tensor_metadata_row(p + "ffn.aT", tensor_store.temp_ffn_aT(li, T), infer_temp_purpose("ffn.aT")));
    lines.push_back(tensor_metadata_row(p + "ffn.W2T", tensor_store.temp_ffn_W2T(li), infer_temp_purpose("ffn.W2T")));
    lines.push_back(tensor_metadata_row(
        p + "ffn.da",
        tensor_store.temp_ffn_da(li, static_cast<int64_t>(cfg.training.batch_size),
                                   static_cast<int64_t>(cfg.training.train_seq_len)),
        infer_temp_purpose("ffn.da")));
    if (tensor_store.algo_factory().ffn_needs_activation_buffers()) {
      lines.push_back(tensor_metadata_row(
          p + "ffn.dh",
          tensor_store.temp_ffn_dh(
              li, static_cast<int64_t>(cfg.training.batch_size),
              static_cast<int64_t>(cfg.training.train_seq_len)),
          infer_temp_purpose("ffn.dh")));
    }
    lines.push_back(tensor_metadata_row(p + "ffn.xT", tensor_store.temp_ffn_xT(li, T), infer_temp_purpose("ffn.xT")));
    lines.push_back(tensor_metadata_row(p + "ffn.W1T", tensor_store.temp_ffn_W1T(li), infer_temp_purpose("ffn.W1T")));
  }
  lines.push_back("+--------------------+---------+-----------------------+--------+----------------+---------+------------------------------+");

  for (const auto &line : lines) {
    print(line);
  }
  print("");
}
