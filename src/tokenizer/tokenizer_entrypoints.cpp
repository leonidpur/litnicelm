#include "tokenizer_plugin.hpp"
#include "tokenizer_utils.hpp"
#include "corpus_input.hpp"

#include <tokenizer_factory.hpp>

#include <config.hpp>

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::string now_timestamp_local() {
  const auto now = std::time(nullptr);
  std::tm tmv{};
#if defined(_WIN32)
  localtime_s(&tmv, &now);
#else
  localtime_r(&now, &tmv);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

void write_tokenizer_journal_entry(const std::string &journal_path,
                                   const std::string &op_name,
                                   const std::string &details) {
  if (journal_path.empty()) {
    return;
  }

  const fs::path path(journal_path);
  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }

  std::ofstream journal(path, std::ios::app);
  if (!journal) {
    throw std::runtime_error(
        "tokenizer journal: failed to open journal: " + path.string());
  }

  journal << "[" << now_timestamp_local() << "] OPERATION: " << op_name
          << "\n\n";
  journal << details << "\n\n";
  if (!journal) {
    throw std::runtime_error(
        "tokenizer journal: failed to write journal: " + path.string());
  }
}

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
  const std::string artifacts_dir = resolve_tokenizer_artifacts_dir(cfg);
  if (artifacts_dir.empty()) {
    throw std::runtime_error(
        "run_tokenizer_training_mode: tokenizer.artifacts_dir is required");
  }
  auto plugin = TokenizerFactory::create_plugin(cfg, sink);

  const std::string training_corpus = resolve_tokenizer_training_corpus(cfg);
  std::vector<std::string> training_files;
  if (training_corpus.empty()) {
    if (std::string(plugin->name()).find("CharacterTokenizer") ==
        std::string::npos) {
      throw std::runtime_error(
          "run_tokenizer_training_mode: tokenizer.training_corpus is required");
    }
  } else {
    training_files = CorpusInput::resolve_files(training_corpus);
  }

  std::string plugin_training_corpus = training_corpus;
  std::string temp_training_corpus;
  if (CorpusInput::needs_merge_file(training_files)) {
    temp_training_corpus = CorpusInput::materialize_merged_temp_file(
        training_files, cfg.tokenizer.inter_file_boundary, "tokenizer_training");
    plugin_training_corpus = temp_training_corpus;
  }

  const std::string create_message =
      "Tokenizer training plugin loaded: plugin=" +
      std::string(plugin->name()) + ", training_corpus=" + training_corpus +
      ", artifacts_dir=" + artifacts_dir +
      ", target_vocab_size=" + std::to_string(cfg.tokenizer.target_vocab_size) +
      ", artifacts_dir_out=" + artifacts_dir +
      (training_files.empty() ? "" : ", " + CorpusInput::describe_files(training_files));
  std::cout << "[TOKENIZER_TRAINING] " << create_message << "\n";
  report_if(sink, ReportEvent::START, 0, 0.0f, create_message);
  try {
    plugin->train(plugin_training_corpus, artifacts_dir,
                  cfg.tokenizer.target_vocab_size, sink);
  } catch (...) {
    if (!temp_training_corpus.empty()) {
      fs::remove(temp_training_corpus);
    }
    throw;
  }
  if (!temp_training_corpus.empty()) {
    fs::remove(temp_training_corpus);
  }

  const std::string details =
      "Status: SUCCESS\n\n"
      "Training Corpus: " + training_corpus + "\n\n"
      "Resolved Corpus: " +
      (training_files.empty() ? "files=0" : CorpusInput::describe_files(training_files)) +
      "\n\n"
      "Artifacts Dir: " + artifacts_dir + "\n\n"
      "Tokenizer: " + plugin->name() + "\n\n"
      "Vocab Size: " + std::to_string(cfg.tokenizer.target_vocab_size);
  write_tokenizer_journal_entry(cfg.paths.journal_file, "TOKENIZER_GEN",
                                details);
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
                                  cfg.tokenization.chunk_size_mb,
                                  cfg.tokenizer.inter_file_boundary,
                                  cfg.tokenizer.run_validation, sink);

  const std::string details =
      "Status: SUCCESS\n\n"
      "Input Corpus: " + cfg.tokenization.input_corpus + "\n\n"
      "Output Dataset: " + cfg.tokenization.output_binary + "\n\n"
      "Artifacts Dir: " + artifacts_dir + "\n\n"
      "Tokenizer: " + plugin->name();
  write_tokenizer_journal_entry(cfg.paths.journal_file, "TOKENIZATION_RUN",
                                details);
  return 0;
}
