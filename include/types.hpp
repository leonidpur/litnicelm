#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class Device : uint8_t {
  CPU = 0,
  GPU = 1,
};

struct TrainingConfig {
  float learning_rate = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  bool incremental = false;
  bool dry_run = false;
  uint32_t num_epochs_train = 1;
  uint32_t num_epochs_dry_run = 1;
  uint32_t save_interval_epochs = 1;
  float grad_clip = 1.0f;
  uint32_t window_training = 128;
  uint32_t batch_size = 1;
};

struct PathsConfig {
  std::string model_file = "checkpoints/model.ckpt";
  std::string journal_file = "journal/journal.txt";
};

struct TokenizerConfig {
  std::string type = "character";
  uint32_t target_vocab_size = 256;
  std::string bpe_corpus_file;
  std::string bpe_artifacts_dir;
  std::string bpe_vocab_file;
  std::string bpe_merges_file;
  bool run_validation = true;
  int32_t bpe_validation_num_threads = 0;
  uint32_t bpe_validation_sample_rate = 100;
};

struct TokenizationConfig {
  std::string input_corpus;
  std::string output_binary;
  uint32_t chunk_size_mb = 64;
};

struct InferenceConfig {
  uint32_t window_inference = 64;
};

struct LoggingConfig {
  bool show_bpe = true;
  bool show_train = true;
  bool show_inference = true;
  int32_t report_every_n_steps = 1;
};

struct ReportingConfig {
  std::vector<int32_t> verbose_epoch_index{-1, 0};
  bool verbose_init = false;
};

struct ModelConfig {
  uint32_t n_layers = 2;
  uint32_t n_heads = 4;
  uint32_t d_model = 32;
  uint32_t d_ff = 64;
  uint32_t target_vocab_size = 256;
  uint32_t window_capacity = 128;
};

struct MemoryConfig {
  uint64_t alignment_bytes = 64;
};

struct Command;

struct Config {
  std::string conf_version = "1";
  Device device = Device::CPU;
  uint32_t transformer_layers = 0;
  uint64_t parameter_bytes = 0;
  uint64_t optimizer_bytes = 0;
  uint32_t arena_alignment = 64;
  uint64_t max_steps = 1;
  ModelConfig model;
  MemoryConfig memory;
  PathsConfig paths;
  TokenizerConfig tokenizer;
  TokenizationConfig tokenization;
  TrainingConfig training;
  InferenceConfig inference;
  LoggingConfig logging;
  ReportingConfig reporting;

  static Config load_from_file(const std::string &path);
  void apply_env_overrides(const std::string &prefix);
  std::string apply_command_overrides(const Command &cmd);
  void validate_base() const;
  void validate() const;
};

struct RuntimeFlags {
  bool do_probe = false;
  bool logit = false;
  uint32_t print_mod = 100;
};

struct Command {
  enum class Target {
    TRAIN,
    DRY_RUN,
    TOKENIZER_TRAINING,
    TOKENIZE,
    INFER,
    INSPECT,
    INFERLOOP,
    BACKUP,
  };

  Target target = Target::TRAIN;
  std::string config_path = "./config.yaml";

  uint32_t window_training_override = 0;
  uint32_t batch_size_override = 0;
  RuntimeFlags runtime_flags{};
  uint32_t num_epochs_override = 0;

  std::string prompt;
  std::string backup_input;
  std::string backup_root;
};
