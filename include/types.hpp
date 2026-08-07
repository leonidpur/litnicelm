#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class Device : uint8_t {
  CPU = 0,
  GPU = 1,
};

struct TrainingConfig {
  float learning_rate;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  bool incremental;
  bool dry_run;
  uint32_t num_epochs_train;
  uint32_t num_epochs_dry_run;
  uint32_t save_interval_epochs;
  float grad_clip;
  uint32_t window_training;
  uint32_t batch_size;
};

struct PathsConfig {
  std::string model_file;
  std::string journal_file;
};

struct TokenizerConfig {
  std::string type;
  uint32_t target_vocab_size;
  std::string bpe_corpus_file;
  std::string bpe_artifacts_dir;
  std::string bpe_vocab_file;
  std::string bpe_merges_file;
  bool run_validation;
  int32_t bpe_validation_num_threads;
  uint32_t bpe_validation_sample_rate;
};

struct TokenizationConfig {
  std::string input_corpus;
  std::string output_binary;
  uint32_t chunk_size_mb;
};

struct InferenceConfig {
  uint32_t window_inference;
};

struct LoggingConfig {
  bool show_bpe;
  bool show_train;
  bool show_inference;
  int32_t report_every_n_steps;
  uint32_t epoch_report_every;
};

struct ReportingConfig {
  std::vector<int32_t> verbose_epoch_index;
  bool verbose_init;
};

struct ModelConfig {
  uint32_t n_layers;
  uint32_t n_heads;
  uint32_t d_model;
  uint32_t d_ff;
  uint32_t target_vocab_size;
  uint32_t window_capacity;
};

struct MemoryConfig {
  uint64_t alignment_bytes;
};

struct Command;

struct Config {
  std::string conf_version;
  Device device;
  uint32_t transformer_layers;
  uint64_t parameter_bytes;
  uint64_t optimizer_bytes;
  uint32_t arena_alignment;
  uint64_t max_steps;
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
  struct ProbeFlags {
    bool embeddings;
    bool output_head;
    bool loss;
    bool backward;
    bool attention;
    bool ffn;
    bool layernorm;
  } probe;
  bool logit;
  uint32_t epoch_report_every;
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

  Target target;
  std::string config_path;

  uint32_t window_training_override;
  uint32_t batch_size_override;
  bool has_incremental_override;
  bool incremental_override;
  RuntimeFlags runtime_flags;
  uint32_t num_epochs_override;

  std::string prompt;
  std::string backup_input;
  std::string backup_root;
};
