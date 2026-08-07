#include "tokenizer_plugin.hpp"
#include "tokenizer_utils.hpp"

#include <tokenizer_factory.hpp>

#include <config.hpp>
#include "operation_journal.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {
std::string resolve_tokenizer_artifacts_dir(const Config &cfg) {
  if (!cfg.tokenizer.bpe_artifacts_dir.empty()) {
    return cfg.tokenizer.bpe_artifacts_dir;
  }
  if (!cfg.tokenizer.bpe_vocab_file.empty()) {
    return fs::path(cfg.tokenizer.bpe_vocab_file).parent_path().string();
  }
  if (!cfg.tokenizer.bpe_merges_file.empty()) {
    return fs::path(cfg.tokenizer.bpe_merges_file).parent_path().string();
  }
  return "";
}

std::string resolve_tokenizer_training_corpus(const Config &cfg) {
  return cfg.tokenizer.bpe_corpus_file;
}

void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TOKENIZER, event, step, value,
                          message);
}
} // namespace

int run_tokenizer_training_mode(const std::string &config_path, ReportSink *sink) {
  const Config cfg = Config::load_from_file(config_path);
  const std::string training_corpus = resolve_tokenizer_training_corpus(cfg);
  const std::string artifacts_dir = resolve_tokenizer_artifacts_dir(cfg);
  if (artifacts_dir.empty()) {
    throw std::runtime_error(
        "run_tokenizer_training_mode: tokenizer.artifacts_dir is required");
  }
  if (training_corpus.empty()) {
    throw std::runtime_error(
        "run_tokenizer_training_mode: tokenizer.training_corpus is required");
  }

  auto plugin = TokenizerFactory::create_plugin(cfg, sink);
  const std::string create_message =
      "Tokenizer training plugin loaded: plugin=" +
      std::string(plugin->name()) + ", training_corpus=" + training_corpus +
      ", artifacts_dir=" + artifacts_dir +
      ", target_vocab_size=" + std::to_string(cfg.model.target_vocab_size) +
      ", outputs=[" + (fs::path(artifacts_dir) / "spm.model").string() + ", " +
      (fs::path(artifacts_dir) / "spm.vocab").string() + "]";
  std::cout << "[TOKENIZER_TRAINIG] " << create_message << "\n";
  report_if(sink, ReportEvent::START, 0, 0.0f, create_message);
  plugin->train(training_corpus, artifacts_dir, cfg.model.target_vocab_size,
                sink);

  const std::string details =
      "Status: SUCCESS\n\n"
      "Training Corpus: " + training_corpus + "\n\n"
      "Artifacts Dir: " + artifacts_dir + "\n\n"
      "Tokenizer: " + plugin->name() + "\n\n"
      "Vocab Size: " + std::to_string(cfg.model.target_vocab_size);
  log_operation(cfg.paths.journal_file, "BPE_TOKENIZER_GEN", details);
  return 0;
}

int run_tokenization_mode(const std::string &config_path, ReportSink *sink) {
  const Config cfg = Config::load_from_file(config_path);
  const std::string artifacts_dir = resolve_tokenizer_artifacts_dir(cfg);
  if (artifacts_dir.empty()) {
    throw std::runtime_error(
        "run_tokenization_mode: tokenizer.artifacts_dir is required");
  }

  auto plugin = TokenizerFactory::create_plugin(cfg, sink);
  if (!plugin->load(artifacts_dir)) {
    throw std::runtime_error(
        "run_tokenization_mode: failed to load tokenizer artifacts from " +
        artifacts_dir);
  }

  const std::string tokenize_message =
      "Tokenization plugin loaded: plugin=" + std::string(plugin->name()) +
      ", input=" + cfg.tokenization.input_corpus +
      ", output=" + cfg.tokenization.output_binary +
      ", artifacts_dir=" + artifacts_dir +
      ", chunk_size_mb=" + std::to_string(cfg.tokenization.chunk_size_mb);
  std::cout << "[TOKENIZER] " << tokenize_message << "\n";
  report_if(sink, ReportEvent::START, 0, 0.0f, tokenize_message);

  TokenizerUtils::stream_tokenize(plugin.get(), cfg.tokenization.input_corpus,
                                  cfg.tokenization.output_binary,
                                  cfg.tokenization.chunk_size_mb, sink);

  const std::string details =
      "Status: SUCCESS\n\n"
      "Input Corpus: " + cfg.tokenization.input_corpus + "\n\n"
      "Output Dataset: " + cfg.tokenization.output_binary + "\n\n"
      "Artifacts Dir: " + artifacts_dir + "\n\n"
      "Tokenizer: " + plugin->name();
  log_operation(cfg.paths.journal_file, "TOKENIZATION_RUN", details);
  return 0;
}
