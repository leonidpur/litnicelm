#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class Device : uint8_t {
  CPU = 0,
  GPU = 1,
};

struct TrainingConfig {
  struct DiagnosticsConfig {
    bool fw_after_forward_logits;
    bool fw_after_loss_scalar;
    bool fw_after_logits_targets_backward;
    bool fw_after_cross_entropy_backward;
    bool on_nonfinite_grad_norm_original_dlogits;
    bool on_nonfinite_grad_norm_replay;
    bool bk_transformer_dlogits;
    bool bk_transformer_d_lm_w;
    bool bk_transformer_d_xn;
    bool bk_transformer_d_xlast;
    bool bk_transformer_d_lnf_g;
    bool bk_transformer_d_lnf_b;
    bool bk_transformer_layer_d_prev;
    bool bk_transformer_d_cur_before_embeddings;
    bool bk_transformer_d_tok;
    bool bk_transformer_d_pos;
    bool bk_layer_dln2_after_ffn;
    bool bk_layer_dy_ln2;
    bool bk_layer_dln2_gamma;
    bool bk_layer_dln2_beta;
    bool bk_layer_dy_total;
    bool bk_layer_dln1_after_attn;
    bool bk_layer_dx_ln1;
    bool bk_layer_dln1_gamma;
    bool bk_layer_dln1_beta;
    bool bk_layer_dx;
    bool bk_ffn_dW2;
    bool bk_ffn_db2;
    bool bk_ffn_da;
    bool bk_ffn_dh;
    bool bk_ffn_dW1;
    bool bk_ffn_db1;
    bool bk_ffn_dx;
    bool bk_attn_dWo;
    bool bk_attn_dbo;
    bool bk_attn_dcontext;
    bool bk_attn_dweights;
    bool bk_attn_dVh;
    bool bk_attn_dscores_softmax_backward;
    bool bk_attn_dscores_masked;
    bool bk_attn_dQh;
    bool bk_attn_dKh;
    bool bk_attn_dx;
    bool bk_attn_dWqkv;
    bool bk_attn_dbqkv;
  };

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
  uint32_t train_seq_len;
  uint32_t window_stride;
  uint32_t batch_size;
  float target_loss;
  float min_delta;
  uint32_t patience_epochs;
  uint32_t min_epochs;
  bool stop_on_nonfinite_loss;
  DiagnosticsConfig diagnostics;
};

struct PathsConfig {
  std::string model_file_latest;
  std::string model_file_best;
  std::string journal_file;
};

struct BackendConfig {
  std::string library;
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
  std::string prompt;
  uint32_t max_new;
  float temp;
  uint32_t top_k;
  float top_p;
  uint32_t seed;
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
  uint32_t max_seq_len;
};

struct MemoryConfig {
  uint64_t alignment_bytes;
};

struct Command;

struct Config {
  std::string conf_version;
  uint32_t transformer_layers;
  uint64_t parameter_bytes;
  uint64_t optimizer_bytes;
  uint32_t arena_alignment;
  uint64_t max_steps;
  ModelConfig model;
  MemoryConfig memory;
  PathsConfig paths;
  BackendConfig backend;
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
  };

  Target target;
  std::string config_path;

  uint32_t train_seq_len_override;
  uint32_t batch_size_override;
  bool has_incremental_override;
  bool incremental_override;
  RuntimeFlags runtime_flags;
  uint32_t num_epochs_override;

  bool has_prompt_override;
  std::string prompt;
  bool has_max_new_override;
  uint32_t max_new_override;
  bool has_temp_override;
  float temp_override;
  bool has_top_k_override;
  uint32_t top_k_override;
  bool has_top_p_override;
  float top_p_override;
  bool has_seed_override;
  uint32_t seed_override;
};
