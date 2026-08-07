#include "config_mapper.hpp"

#include "string_utils.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

bool map_root_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "conf.version") {
    cfg.conf_version = value;
    return true;
  }
  if (key == "device") {
    if (value == "cpu" || value == "CPU") {
      cfg.device = Device::CPU;
      return true;
    }
    if (value == "gpu" || value == "GPU") {
      cfg.device = Device::GPU;
      return true;
    }
    throw std::runtime_error("Config::load_from_file: invalid device: " + value);
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
  if (key == "model.window_capacity") {
    cfg.model.window_capacity = parse_u32_or_throw(value, key);
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
  if (key == "paths.model_file") {
    cfg.paths.model_file = value;
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
  if (key == "training.window_training") {
    cfg.training.window_training = parse_u32_or_throw(value, key);
    return true;
  }
  if (key == "training.batch_size") {
    cfg.training.batch_size = parse_u32_or_throw(value, key);
    return true;
  }
  return false;
}

bool map_inference_fields(const std::string &key, const std::string &value, Config &cfg) {
  if (key == "inference.window_inference") {
    cfg.inference.window_inference = parse_u32_or_throw(value, key);
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
  for (const auto &entry : entries) {
    const bool mapped =
        map_root_fields(entry.key, entry.value, cfg) ||
        map_model_fields(entry.key, entry.value, cfg) ||
        map_memory_fields(entry.key, entry.value, cfg) ||
        map_paths_fields(entry.key, entry.value, cfg) ||
        map_tokenizer_fields(entry.key, entry.value, cfg) ||
        map_tokenization_fields(entry.key, entry.value, cfg) ||
        map_training_fields(entry.key, entry.value, cfg) ||
        map_inference_fields(entry.key, entry.value, cfg) ||
        map_logging_fields(entry.key, entry.value, cfg);

    if (!mapped) {
      std::cerr << "Warning: Config::load_from_file: unknown key at line "
                << entry.lineno << ": " << entry.key << "\n";
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

  if (cmd.window_training_override > 0) {
    this->training.window_training = cmd.window_training_override;
    replaced.push_back("training.window_training=" +
                       std::to_string(cmd.window_training_override));
  }
  if (cmd.batch_size_override > 0) {
    this->training.batch_size = cmd.batch_size_override;
    replaced.push_back("training.batch_size=" +
                       std::to_string(cmd.batch_size_override));
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
