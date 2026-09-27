#include "config_yaml.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#define LITNICE_DIAGNOSTICS_FIELDS(X) \
  X(fw_after_forward_logits) \
  X(fw_after_loss_scalar) \
  X(fw_after_logits_targets_backward) \
  X(fw_after_cross_entropy_backward) \
  X(on_nonfinite_grad_norm_original_dlogits) \
  X(on_nonfinite_grad_norm_replay) \
  X(bk_transformer_dlogits) \
  X(bk_transformer_d_lm_w) \
  X(bk_transformer_d_xn) \
  X(bk_transformer_d_xlast) \
  X(bk_transformer_d_lnf_g) \
  X(bk_transformer_d_lnf_b) \
  X(bk_transformer_layer_d_prev) \
  X(bk_transformer_d_cur_before_embeddings) \
  X(bk_transformer_d_tok) \
  X(bk_transformer_d_pos) \
  X(bk_layer_dln2_after_ffn) \
  X(bk_layer_dy_ln2) \
  X(bk_layer_dln2_gamma) \
  X(bk_layer_dln2_beta) \
  X(bk_layer_dy_total) \
  X(bk_layer_dln1_after_attn) \
  X(bk_layer_dx_ln1) \
  X(bk_layer_dln1_gamma) \
  X(bk_layer_dln1_beta) \
  X(bk_layer_dx) \
  X(bk_ffn_dW2) \
  X(bk_ffn_db2) \
  X(bk_ffn_da) \
  X(bk_ffn_dh) \
  X(bk_ffn_dW1) \
  X(bk_ffn_db1) \
  X(bk_ffn_dx) \
  X(bk_attn_dWo) \
  X(bk_attn_dbo) \
  X(bk_attn_dcontext) \
  X(bk_attn_dweights) \
  X(bk_attn_dVh) \
  X(bk_attn_dscores_softmax_backward) \
  X(bk_attn_dscores_masked) \
  X(bk_attn_dQh) \
  X(bk_attn_dKh) \
  X(bk_attn_dx) \
  X(bk_attn_dWqkv) \
  X(bk_attn_dbqkv) \

namespace {

// Shortest decimal text that parses back to the same float.
std::string float_text(float v) {
  char buf[64];
  for (int precision = 6; precision <= 9; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, static_cast<double>(v));
    if (std::strtof(buf, nullptr) == v) {
      break;
    }
  }
  return buf;
}

// The loader strips one pair of surrounding quotes and keeps the rest
// verbatim. Newlines cannot appear in a single-line value, so they are
// written as \n (only inter_file_boundary decodes them back).
std::string quoted(const std::string &v) {
  std::string out = "\"";
  for (char c : v) {
    if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else {
      out.push_back(c);
    }
  }
  return out + "\"";
}

// inter_file_boundary is decoded with parse_escaped_string, so backslashes
// must be escaped as well.
std::string quoted_escaped(const std::string &v) {
  std::string escaped;
  for (char c : v) {
    if (c == '\\') {
      escaped += "\\\\";
    } else {
      escaped.push_back(c);
    }
  }
  return quoted(escaped);
}

const char *boolean(bool v) { return v ? "true" : "false"; }

} // namespace

std::string config_to_yaml(const Config &cfg) {
  std::ostringstream y;
  y << "conf:\n"
    << "  version: " << quoted(cfg.conf_version) << "\n\n"
    << "transformer_layers: " << cfg.transformer_layers << "\n"
    << "parameter_bytes: " << cfg.parameter_bytes << "\n"
    << "optimizer_bytes: " << cfg.optimizer_bytes << "\n"
    << "arena_alignment: " << cfg.arena_alignment << "\n"
    << "max_steps: " << cfg.max_steps << "\n\n";

  y << "backend:\n"
    << "  library: " << quoted(cfg.backend.library) << "\n\n";

  y << "tokenizer:\n"
    << "  type: " << quoted(cfg.tokenizer.type) << "\n"
    << "  target_vocab_size: " << cfg.tokenizer.target_vocab_size << "\n"
    << "  training_corpus: " << quoted(cfg.tokenizer.bpe_corpus_file) << "\n"
    << "  inter_file_boundary: "
    << quoted_escaped(cfg.tokenizer.inter_file_boundary) << "\n"
    << "  artifacts_dir: " << quoted(cfg.tokenizer.bpe_artifacts_dir) << "\n"
    << "  bpe_vocab_file: " << quoted(cfg.tokenizer.bpe_vocab_file) << "\n"
    << "  bpe_merges_file: " << quoted(cfg.tokenizer.bpe_merges_file) << "\n"
    << "  bpe_validation_num_threads: "
    << cfg.tokenizer.bpe_validation_num_threads << "\n"
    << "  bpe_validation_sample_rate: "
    << cfg.tokenizer.bpe_validation_sample_rate << "\n"
    << "  run_validation: " << boolean(cfg.tokenizer.run_validation) << "\n\n";

  y << "tokenization:\n"
    << "  input_corpus: " << quoted(cfg.tokenization.input_corpus) << "\n"
    << "  output_binary: " << quoted(cfg.tokenization.output_binary) << "\n"
    << "  chunk_size_mb: " << cfg.tokenization.chunk_size_mb << "\n\n";

  y << "memory:\n"
    << "  alignment_bytes: " << cfg.memory.alignment_bytes << "\n\n";

  y << "model:\n"
    << "  max_seq_len: " << cfg.model.max_seq_len << "\n"
    << "  n_layers: " << cfg.model.n_layers << "\n"
    << "  n_heads: " << cfg.model.n_heads << "\n"
    << "  d_model: " << cfg.model.d_model << "\n"
    << "  d_ff: " << cfg.model.d_ff << "\n\n";

  y << "model_algo:\n"
    << "  attention: " << quoted(cfg.model_algo.attention) << "\n"
    << "  ffn: " << quoted(cfg.model_algo.ffn) << "\n"
    << "  position_encoding: " << quoted(cfg.model_algo.position_encoding)
    << "\n\n";

  const TrainingConfig &t = cfg.training;
  y << "training:\n"
    << "  learning_rate: " << float_text(t.learning_rate) << "\n"
    << "  beta1: " << float_text(t.beta1) << "\n"
    << "  beta2: " << float_text(t.beta2) << "\n"
    << "  eps: " << float_text(t.eps) << "\n"
    << "  weight_decay: " << float_text(t.weight_decay) << "\n"
    << "  incremental: " << boolean(t.incremental) << "\n"
    << "  dry_run: " << boolean(t.dry_run) << "\n"
    << "  num_epochs_train: " << t.num_epochs_train << "\n"
    << "  num_epochs_dry_run: " << t.num_epochs_dry_run << "\n"
    << "  save_interval_epochs: " << t.save_interval_epochs << "\n"
    << "  grad_clip: " << float_text(t.grad_clip) << "\n"
    << "  train_seq_len: " << t.train_seq_len << "\n"
    << "  window_stride: " << t.window_stride << "\n"
    << "  batch_size: " << t.batch_size << "\n"
    << "  target_loss: " << float_text(t.target_loss) << "\n"
    << "  min_delta: " << float_text(t.min_delta) << "\n"
    << "  patience_epochs: " << t.patience_epochs << "\n"
    << "  min_epochs: " << t.min_epochs << "\n"
    << "  stop_on_nonfinite_loss: " << boolean(t.stop_on_nonfinite_loss) << "\n"
    << "  diagnostics:\n";
#define LITNICE_DUMP_DIAGNOSTIC(name) \
  y << "    " #name ": " << boolean(t.diagnostics.name) << "\n";
  LITNICE_DIAGNOSTICS_FIELDS(LITNICE_DUMP_DIAGNOSTIC)
#undef LITNICE_DUMP_DIAGNOSTIC
  y << "\n";

  y << "paths:\n"
    << "  model_file_latest: " << quoted(cfg.paths.model_file_latest) << "\n"
    << "  model_file_best: " << quoted(cfg.paths.model_file_best) << "\n"
    << "  journal_file: " << quoted(cfg.paths.journal_file) << "\n\n";

  y << "inference:\n"
    << "  prompt: " << quoted(cfg.inference.prompt) << "\n"
    << "  max_new: " << cfg.inference.max_new << "\n"
    << "  temp: " << float_text(cfg.inference.temp) << "\n"
    << "  top_k: " << cfg.inference.top_k << "\n"
    << "  top_p: " << float_text(cfg.inference.top_p) << "\n"
    << "  seed: " << cfg.inference.seed << "\n\n";

  y << "logging:\n"
    << "  show_bpe: " << boolean(cfg.logging.show_bpe) << "\n"
    << "  show_train: " << boolean(cfg.logging.show_train) << "\n"
    << "  show_inference: " << boolean(cfg.logging.show_inference) << "\n"
    << "  report_every_n_steps: " << cfg.logging.report_every_n_steps << "\n"
    << "  epoch_report_every: " << cfg.logging.epoch_report_every << "\n\n";

  y << "reporting:\n  verbose_epoch_index: [";
  for (size_t i = 0; i < cfg.reporting.verbose_epoch_index.size(); ++i) {
    y << (i == 0 ? "" : ", ") << cfg.reporting.verbose_epoch_index[i];
  }
  y << "]\n"
    << "  verbose_init: " << boolean(cfg.reporting.verbose_init) << "\n";

  const TrackingConfig &tr = cfg.tracking;
  if (!tr.sinks.empty() || !tr.options.empty() ||
      tr.metrics_every_n_steps != 0) {
    y << "\ntracking:\n  sinks: [";
    for (size_t i = 0; i < tr.sinks.size(); ++i) {
      y << (i == 0 ? "" : ", ") << quoted(tr.sinks[i]);
    }
    y << "]\n"
      << "  metrics_every_n_steps: " << tr.metrics_every_n_steps << "\n";
    // Dotted keys flatten to the same tracking.<sink>.<option> names.
    for (const auto &[key, value] : tr.options) {
      y << "  " << key << ": " << quoted(value) << "\n";
    }
  }
  return y.str();
}
