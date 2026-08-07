#include "config_mapper.hpp"

#include "string_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
bool parse_bool_or_throw(const std::string &v, const std::string &k) {
  const std::string s = string_utils::trim_copy(v);
  if (s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes" || s == "on") {
    return true;
  }
  if (s == "0" || s == "false" || s == "False" || s == "FALSE" || s == "no" || s == "off") {
    return false;
  }
  throw std::runtime_error("Config::load_from_file: invalid bool for key " + k + ": " + v);
}

uint64_t parse_u64_or_throw(const std::string &v, const std::string &k) {
  try {
    return static_cast<uint64_t>(std::stoull(string_utils::trim_copy(v)));
  } catch (...) {
    throw std::runtime_error("Config::load_from_file: invalid uint for key " + k + ": " + v);
  }
}

uint32_t parse_u32_or_throw(const std::string &v, const std::string &k) {
  const uint64_t x = parse_u64_or_throw(v, k);
  if (x > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Config::load_from_file: u32 overflow for key " + k);
  }
  return static_cast<uint32_t>(x);
}

int32_t parse_i32_or_throw(const std::string &v, const std::string &k) {
  try {
    const long x = std::stol(string_utils::trim_copy(v));
    if (x < static_cast<long>(std::numeric_limits<int32_t>::min()) ||
        x > static_cast<long>(std::numeric_limits<int32_t>::max())) {
      throw std::runtime_error("Config::load_from_file: i32 overflow for key " + k);
    }
    return static_cast<int32_t>(x);
  } catch (const std::runtime_error &) {
    throw;
  } catch (...) {
    throw std::runtime_error("Config::load_from_file: invalid int for key " + k + ": " + v);
  }
}

float parse_f32_or_throw(const std::string &v, const std::string &k) {
  try {
    return std::stof(string_utils::trim_copy(v));
  } catch (...) {
    throw std::runtime_error("Config::load_from_file: invalid float for key " + k + ": " + v);
  }
}

std::pair<int32_t, int32_t> parse_i32_pair_or_throw(const std::string &v,
                                                     const std::string &k) {
  std::string s = string_utils::trim_copy(v);
  if (s.size() < 5 || s.front() != '[' || s.back() != ']') {
    throw std::runtime_error("Config::load_from_file: invalid [i32,i32] for key " + k +
                             ": " + v);
  }
  s = string_utils::trim_copy(s.substr(1, s.size() - 2));
  const size_t comma = s.find(',');
  if (comma == std::string::npos) {
    throw std::runtime_error("Config::load_from_file: invalid [i32,i32] for key " + k +
                             ": " + v);
  }
  const std::string a = string_utils::trim_copy(s.substr(0, comma));
  const std::string b = string_utils::trim_copy(s.substr(comma + 1));
  return {parse_i32_or_throw(a, k), parse_i32_or_throw(b, k)};
}

std::string canonical_config_key(const std::string &key) {
  if (key == "model.window_capacity") {
    return "model.max_seq_len";
  }
  if (key == "training.window_training") {
    return "training.train_seq_len";
  }
  return key;
}

const std::vector<std::string> &required_keys() {
  static const std::vector<std::string> keys = {
      "conf.version",
      "transformer_layers",
      "parameter_bytes",
      "optimizer_bytes",
      "arena_alignment",
      "max_steps",
      "backend.library",
      "model.n_layers",
      "model.n_heads",
      "model.d_model",
      "model.d_ff",
      "model.max_seq_len",
      "memory.alignment_bytes",
      "paths.model_file_latest",
      "paths.model_file_best",
      "paths.journal_file",
      "tokenizer.type",
      "tokenizer.target_vocab_size",
      "tokenizer.training_corpus",
      "tokenizer.artifacts_dir",
      "tokenizer.bpe_vocab_file",
      "tokenizer.bpe_merges_file",
      "tokenizer.run_validation",
      "tokenizer.bpe_validation_num_threads",
      "tokenizer.bpe_validation_sample_rate",
      "tokenization.input_corpus",
      "tokenization.output_binary",
      "tokenization.chunk_size_mb",
      "training.learning_rate",
      "training.beta1",
      "training.beta2",
      "training.eps",
      "training.weight_decay",
      "training.incremental",
      "training.dry_run",
      "training.num_epochs_train",
      "training.num_epochs_dry_run",
      "training.save_interval_epochs",
      "training.grad_clip",
      "training.train_seq_len",
      "training.window_stride",
      "training.batch_size",
      "training.target_loss",
      "training.min_delta",
      "training.patience_epochs",
      "training.min_epochs",
      "training.stop_on_nonfinite_loss",
      "inference.prompt",
      "inference.max_new",
      "inference.temp",
      "inference.top_k",
      "inference.top_p",
      "inference.seed",
      "logging.show_bpe",
      "logging.show_train",
      "logging.show_inference",
      "logging.report_every_n_steps",
      "logging.epoch_report_every",
      "reporting.verbose_epoch_index",
      "reporting.verbose_init",
  };
  return keys;
}

bool map_root_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "conf.version") {
    cfg.conf_version = value;
    return true;
  }
  if (key == "transformer_layers") {
    cfg.transformer_layers = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "parameter_bytes") {
    cfg.parameter_bytes = parse_u64_or_throw(value, key);
    return true;
  }
  if (key == "optimizer_bytes") {
    cfg.optimizer_bytes = parse_u64_or_throw(value, key);
    return true;
  }
  if (key == "arena_alignment") {
    cfg.arena_alignment = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "max_steps") {
    cfg.max_steps = parse_u64_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_backend_fields(const std::string &key, const std::string &value,
                        Config &cfg) {
  if (key == "backend.library") {
    cfg.backend.library = value;
    return true;
  }
  return false;
}

bool map_model_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "model.n_layers") {
    cfg.model.n_layers = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "model.n_heads") {
    cfg.model.n_heads = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "model.d_model") {
    cfg.model.d_model = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "model.d_ff") {
    cfg.model.d_ff = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "model.max_seq_len" || key == "model.window_capacity") {
    cfg.model.max_seq_len = parse_u32_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_memory_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "memory.alignment_bytes") {
    cfg.memory.alignment_bytes = parse_u64_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_paths_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "paths.model_file_latest") {
    cfg.paths.model_file_latest = value;
    return true;
  }
  if (key == "paths.model_file_best") {
    cfg.paths.model_file_best = value;
    return true;
  }
  if (key == "paths.journal_file") {
    cfg.paths.journal_file = value;
    return true;
  }
  return false;
}

bool map_tokenizer_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "tokenizer.type") {
    cfg.tokenizer.type = value;
    return true;
  }
  if (key == "tokenizer.target_vocab_size") {
    cfg.tokenizer.target_vocab_size = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "tokenizer.training_corpus") {
    cfg.tokenizer.bpe_corpus_file = value;
    return true;
  }
  if (key == "tokenizer.artifacts_dir") {
    cfg.tokenizer.bpe_artifacts_dir = value;
    return true;
  }
  if (key == "tokenizer.bpe_corpus_file") {
    cfg.tokenizer.bpe_corpus_file = value;
    return true;
  }
  if (key == "tokenizer.bpe_artifacts_dir") {
    cfg.tokenizer.bpe_artifacts_dir = value;
    return true;
  }
  if (key == "tokenizer.bpe_vocab_file") {
    cfg.tokenizer.bpe_vocab_file = value;
    return true;
  }
  if (key == "tokenizer.bpe_merges_file") {
    cfg.tokenizer.bpe_merges_file = value;
    return true;
  }
  if (key == "tokenizer.run_validation") {
    cfg.tokenizer.run_validation = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "tokenizer.bpe_validation_num_threads") {
    cfg.tokenizer.bpe_validation_num_threads = parse_i32_or_throw(value, key);
    return true;
  }
  if (key == "tokenizer.bpe_validation_sample_rate") {
    cfg.tokenizer.bpe_validation_sample_rate = parse_u32_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_tokenization_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "tokenization.input_corpus") {
    cfg.tokenization.input_corpus = value;
    return true;
  }
  if (key == "tokenization.output_binary") {
    cfg.tokenization.output_binary = value;
    return true;
  }
  if (key == "tokenization.chunk_size_mb") {
    cfg.tokenization.chunk_size_mb = parse_u32_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_training_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "training.learning_rate") {
    cfg.training.learning_rate = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.beta1") {
    cfg.training.beta1 = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.beta2") {
    cfg.training.beta2 = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.eps") {
    cfg.training.eps = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.weight_decay") {
    cfg.training.weight_decay = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.incremental") {
    cfg.training.incremental = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.dry_run") {
    cfg.training.dry_run = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.num_epochs_train") {
    cfg.training.num_epochs_train = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.num_epochs_dry_run") {
    cfg.training.num_epochs_dry_run = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.save_interval_epochs") {
    cfg.training.save_interval_epochs = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.grad_clip") {
    cfg.training.grad_clip = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.train_seq_len" || key == "training.window_training") {
    cfg.training.train_seq_len = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.window_stride") {
    cfg.training.window_stride = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.batch_size") {
    cfg.training.batch_size = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.target_loss") {
    cfg.training.target_loss = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.min_delta") {
    cfg.training.min_delta = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "training.patience_epochs") {
    cfg.training.patience_epochs = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.min_epochs") {
    cfg.training.min_epochs = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.stop_on_nonfinite_loss") {
    cfg.training.stop_on_nonfinite_loss = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.fw_after_forward_logits") {
    cfg.training.diagnostics.fw_after_forward_logits =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.fw_after_loss_scalar") {
    cfg.training.diagnostics.fw_after_loss_scalar = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.fw_after_logits_targets_backward") {
    cfg.training.diagnostics.fw_after_logits_targets_backward =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.fw_after_cross_entropy_backward") {
    cfg.training.diagnostics.fw_after_cross_entropy_backward =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.on_nonfinite_grad_norm_original_dlogits") {
    cfg.training.diagnostics.on_nonfinite_grad_norm_original_dlogits =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.on_nonfinite_grad_norm_replay") {
    cfg.training.diagnostics.on_nonfinite_grad_norm_replay =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_dlogits") {
    cfg.training.diagnostics.bk_transformer_dlogits =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_lm_w") {
    cfg.training.diagnostics.bk_transformer_d_lm_w =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_xn") {
    cfg.training.diagnostics.bk_transformer_d_xn = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_xlast") {
    cfg.training.diagnostics.bk_transformer_d_xlast =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_lnf_g") {
    cfg.training.diagnostics.bk_transformer_d_lnf_g =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_lnf_b") {
    cfg.training.diagnostics.bk_transformer_d_lnf_b =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_layer_d_prev") {
    cfg.training.diagnostics.bk_transformer_layer_d_prev =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_cur_before_embeddings") {
    cfg.training.diagnostics.bk_transformer_d_cur_before_embeddings =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_tok") {
    cfg.training.diagnostics.bk_transformer_d_tok = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_transformer_d_pos") {
    cfg.training.diagnostics.bk_transformer_d_pos = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dln2_after_ffn") {
    cfg.training.diagnostics.bk_layer_dln2_after_ffn =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dy_ln2") {
    cfg.training.diagnostics.bk_layer_dy_ln2 = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dln2_gamma") {
    cfg.training.diagnostics.bk_layer_dln2_gamma =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dln2_beta") {
    cfg.training.diagnostics.bk_layer_dln2_beta = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dy_total") {
    cfg.training.diagnostics.bk_layer_dy_total = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dln1_after_attn") {
    cfg.training.diagnostics.bk_layer_dln1_after_attn =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dx_ln1") {
    cfg.training.diagnostics.bk_layer_dx_ln1 = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dln1_gamma") {
    cfg.training.diagnostics.bk_layer_dln1_gamma =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dln1_beta") {
    cfg.training.diagnostics.bk_layer_dln1_beta = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_layer_dx") {
    cfg.training.diagnostics.bk_layer_dx = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_dW2") {
    cfg.training.diagnostics.bk_ffn_dW2 = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_db2") {
    cfg.training.diagnostics.bk_ffn_db2 = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_da") {
    cfg.training.diagnostics.bk_ffn_da = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_dh") {
    cfg.training.diagnostics.bk_ffn_dh = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_dW1") {
    cfg.training.diagnostics.bk_ffn_dW1 = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_db1") {
    cfg.training.diagnostics.bk_ffn_db1 = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_ffn_dx") {
    cfg.training.diagnostics.bk_ffn_dx = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dWo") {
    cfg.training.diagnostics.bk_attn_dWo = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dbo") {
    cfg.training.diagnostics.bk_attn_dbo = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dcontext") {
    cfg.training.diagnostics.bk_attn_dcontext = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dweights") {
    cfg.training.diagnostics.bk_attn_dweights = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dVh") {
    cfg.training.diagnostics.bk_attn_dVh = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dscores_softmax_backward") {
    cfg.training.diagnostics.bk_attn_dscores_softmax_backward =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dscores_masked") {
    cfg.training.diagnostics.bk_attn_dscores_masked =
        parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dQh") {
    cfg.training.diagnostics.bk_attn_dQh = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dKh") {
    cfg.training.diagnostics.bk_attn_dKh = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dx") {
    cfg.training.diagnostics.bk_attn_dx = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dWqkv") {
    cfg.training.diagnostics.bk_attn_dWqkv = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "training.diagnostics.bk_attn_dbqkv") {
    cfg.training.diagnostics.bk_attn_dbqkv = parse_bool_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_inference_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "inference.prompt") {
    cfg.inference.prompt = value;
    return true;
  }
  if (key == "inference.max_new") {
    cfg.inference.max_new = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "inference.temp") {
    cfg.inference.temp = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "inference.top_k") {
    cfg.inference.top_k = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "inference.top_p") {
    cfg.inference.top_p = parse_f32_or_throw(value, key);
    return true;
  }
  if (key == "inference.seed") {
    cfg.inference.seed = parse_u32_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_logging_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "logging.show_bpe") {
    cfg.logging.show_bpe = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "logging.show_train") {
    cfg.logging.show_train = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "logging.show_inference") {
    cfg.logging.show_inference = parse_bool_or_throw(value, key);
    return true;
  }
  if (key == "logging.report_every_n_steps") {
    cfg.logging.report_every_n_steps = parse_i32_or_throw(value, key);
    return true;
  }
  if (key == "logging.epoch_report_every") {
    cfg.logging.epoch_report_every = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "reporting.verbose_epoch_index") {
    const auto p = parse_i32_pair_or_throw(value, key);
    cfg.reporting.verbose_epoch_index = {p.first, p.second};
    return true;
  }
  if (key == "reporting.verbose_init") {
    cfg.reporting.verbose_init = parse_bool_or_throw(value, key);
    return true;
  }
  return false;
}
} // namespace

void map_config_entries(const std::vector<YamlEntry> &entries, Config &cfg) {
  std::unordered_set<std::string> seen;
  for (const auto &entry : entries) {
    const bool mapped =
        map_root_fields(entry.key, entry.value, cfg) ||
        map_backend_fields(entry.key, entry.value, cfg) ||
        map_model_fields(entry.key, entry.value, cfg) ||
        map_memory_fields(entry.key, entry.value, cfg) ||
        map_paths_fields(entry.key, entry.value, cfg) ||
        map_tokenizer_fields(entry.key, entry.value, cfg) ||
        map_tokenization_fields(entry.key, entry.value, cfg) ||
        map_training_fields(entry.key, entry.value, cfg) ||
        map_inference_fields(entry.key, entry.value, cfg) ||
        map_logging_fields(entry.key, entry.value, cfg);

    if (!mapped) {
      throw std::runtime_error("Config::load_from_file: unknown key at line " +
                               std::to_string(entry.lineno) + ": " + entry.key);
    }
    seen.insert(canonical_config_key(entry.key));
  }

  for (const auto &key : required_keys()) {
    if (seen.find(key) == seen.end()) {
      throw std::runtime_error("Config::load_from_file: required key missing: " + key);
    }
  }
}

void validate_config(const Config &cfg) {
  //placeholder for future use
}

void Config::apply_env_overrides(const std::string &prefix) {
  auto lookup_u32 = [&](const std::string &suffix, uint32_t &field) {
    const std::string full_key = prefix + suffix;
    if (const char *val = std::getenv(full_key.c_str())) {
      field = parse_u32_or_throw(val, full_key);
      std::cout << "  -> ENV Override: " << full_key << " set to " << field << "\n";
    }
  };

  auto lookup_f32 = [&](const std::string &suffix, float &field) {
    const std::string full_key = prefix + suffix;
    if (const char *val = std::getenv(full_key.c_str())) {
      field = parse_f32_or_throw(val, full_key);
      std::cout << "  -> ENV Override: " << full_key << " set to " << field << "\n";
    }
  };

  lookup_f32("LR", this->training.learning_rate);
  lookup_u32("BATCH_SIZE", this->training.batch_size);
}

std::string Config::apply_command_overrides(const Command &cmd) {
  std::vector<std::string> replaced;

  if (cmd.train_seq_len_override > 0) {
    this->training.train_seq_len = cmd.train_seq_len_override;
    replaced.push_back("training.train_seq_len=" +
                       std::to_string(cmd.train_seq_len_override));
  }
  if (cmd.batch_size_override > 0) {
    this->training.batch_size = cmd.batch_size_override;
    replaced.push_back("training.batch_size=" +
                       std::to_string(cmd.batch_size_override));
  }
  if (cmd.has_incremental_override) {
    this->training.incremental = cmd.incremental_override;
    replaced.push_back(std::string("training.incremental=") +
                       (cmd.incremental_override ? "true" : "false"));
  }
  if (cmd.num_epochs_override > 0) {
    if (cmd.target == Command::Target::DRY_RUN) {
      this->training.num_epochs_dry_run = cmd.num_epochs_override;
      replaced.push_back("training.num_epochs_dry_run=" +
                         std::to_string(cmd.num_epochs_override));
    } else {
      this->training.num_epochs_train = cmd.num_epochs_override;
      replaced.push_back("training.num_epochs_train=" +
                         std::to_string(cmd.num_epochs_override));
    }
  }
  if (cmd.runtime_flags.epoch_report_every > 0) {
    this->logging.epoch_report_every = cmd.runtime_flags.epoch_report_every;
    replaced.push_back("logging.epoch_report_every=" +
                       std::to_string(cmd.runtime_flags.epoch_report_every));
  }
  if (cmd.has_prompt_override) {
    this->inference.prompt = cmd.prompt;
    replaced.push_back("inference.prompt=" + cmd.prompt);
  }
  if (cmd.has_max_new_override) {
    this->inference.max_new = cmd.max_new_override;
    replaced.push_back("inference.max_new=" +
                       std::to_string(cmd.max_new_override));
  }
  if (cmd.has_temp_override) {
    this->inference.temp = cmd.temp_override;
    replaced.push_back("inference.temp=" + std::to_string(cmd.temp_override));
  }
  if (cmd.has_top_k_override) {
    this->inference.top_k = cmd.top_k_override;
    replaced.push_back("inference.top_k=" +
                       std::to_string(cmd.top_k_override));
  }
  if (cmd.has_top_p_override) {
    this->inference.top_p = cmd.top_p_override;
    replaced.push_back("inference.top_p=" + std::to_string(cmd.top_p_override));
  }
  if (cmd.has_seed_override) {
    this->inference.seed = cmd.seed_override;
    replaced.push_back("inference.seed=" + std::to_string(cmd.seed_override));
  }

  std::ostringstream out;
  out << "[Step 4] CLI overrides applied (" << replaced.size() << " replaced)";
  if (!replaced.empty()) {
    out << ": ";
    for (size_t i = 0; i < replaced.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << replaced[i];
    }
  }
  return out.str();
}

void Config::validate_base() const { /* plaseholder for future use*/}
