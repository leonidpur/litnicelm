#include "tokenization_utility.hpp"

#include "bpe_tokenizer.hpp"
#include <config.hpp>
#include <dataset.hpp>
#include "operation_journal.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

std::string read_file_or_throw(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("TokenizationUtility: failed to open input corpus: " +
                             path.string());
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void write_tokens_or_throw(const fs::path &path, const std::vector<int32_t> &tokens,
                           uint32_t vocab_size) {
  static constexpr uint32_t kDatasetMagic = 0x4C4E4750u; // "LNGP"
  static constexpr uint32_t kDatasetVersion = 1u;

  if (!path.parent_path().empty()) {
    fs::create_directories(path.parent_path());
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("TokenizationUtility: failed to open output dataset: " +
                             path.string());
  }

  DatasetHeader header{};
  header.magic = kDatasetMagic;
  header.version = kDatasetVersion;
  header.vocab_size = vocab_size;
  header.num_tokens = static_cast<uint64_t>(tokens.size());

  out.write(reinterpret_cast<const char *>(&header),
            static_cast<std::streamsize>(sizeof(header)));
  for (int32_t token : tokens) {
    if (token < 0) {
      throw std::runtime_error("TokenizationUtility: token id < 0");
    }
    const uint32_t u = static_cast<uint32_t>(token);
    if (u >= header.vocab_size) {
      throw std::runtime_error(
          "TokenizationUtility: token id out of header vocab range");
    }
    out.write(reinterpret_cast<const char *>(&u), sizeof(uint32_t));
  }

  if (!out) {
    throw std::runtime_error("TokenizationUtility: failed to write dataset: " +
                             path.string());
  }
}

std::string resolve_bpe_vocab_path(const Config &cfg) {
  if (!cfg.tokenizer.bpe_vocab_file.empty()) {
    return cfg.tokenizer.bpe_vocab_file;
  }
  if (!cfg.tokenizer.bpe_artifacts_dir.empty()) {
    return (fs::path(cfg.tokenizer.bpe_artifacts_dir) / "vocab.txt").string();
  }
  return "";
}

std::string resolve_bpe_merges_path(const Config &cfg) {
  if (!cfg.tokenizer.bpe_merges_file.empty()) {
    return cfg.tokenizer.bpe_merges_file;
  }
  if (!cfg.tokenizer.bpe_artifacts_dir.empty()) {
    return (fs::path(cfg.tokenizer.bpe_artifacts_dir) / "merges.txt").string();
  }
  return "";
}
} // namespace

TokenizationUtility::TokenizationUtility(std::string input_corpus_path,
                                         std::string output_binary_path,
                                         std::string vocab_path,
                                         std::string merges_path,
                                         uint32_t chunk_size_mb,
                                         uint32_t target_vocab_size,
                                         ReportSink *sink)
    : input_corpus_path_(std::move(input_corpus_path)),
      output_binary_path_(std::move(output_binary_path)),
      vocab_path_(std::move(vocab_path)),
      merges_path_(std::move(merges_path)),
      chunk_size_mb_(chunk_size_mb),
      target_vocab_size_(target_vocab_size),
      sink_(sink) {}

void TokenizationUtility::run() const {
  if (input_corpus_path_.empty()) {
    throw std::runtime_error(
        "TokenizationUtility: tokenization.input_corpus is required");
  }
  if (output_binary_path_.empty()) {
    throw std::runtime_error(
        "TokenizationUtility: tokenization.output_binary is required");
  }
  if (vocab_path_.empty() || merges_path_.empty()) {
    throw std::runtime_error(
        "TokenizationUtility: bpe vocab/merges paths are required");
  }

  BPETokenizer tokenizer;
  tokenizer.set_report_sink(sink_);
  if (!tokenizer.load_from_files(vocab_path_, merges_path_, target_vocab_size_)) {
    throw std::runtime_error(
        "TokenizationUtility: failed to load BPE tokenizer artifacts");
  }
  if (target_vocab_size_ == 0) {
    throw std::runtime_error(
        "TokenizationUtility: model.target_vocab_size must be > 0");
  }
  if (static_cast<uint32_t>(tokenizer.vocab_size()) != target_vocab_size_) {
    throw std::runtime_error("TokenizationUtility: tokenizer vocab mismatch. "
                             "target=" +
                             std::to_string(target_vocab_size_) + ", file=" +
                             std::to_string(tokenizer.vocab_size()));
  }

  const std::string corpus = read_file_or_throw(input_corpus_path_);

  report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::START, 0, 0.0f,
            "Tokenization params: input corpus=" + input_corpus_path_ +
                ", output binary=" + output_binary_path_ +
                ", vocab=" + vocab_path_ + ", merges=" + merges_path_ +
                ", chunk size MB=" + std::to_string(chunk_size_mb_) +
                ", input size=" + std::to_string(corpus.size()) + " bytes");

  const size_t chunk_bytes =
      std::max<size_t>(1, static_cast<size_t>(chunk_size_mb_) * 1024u * 1024u);
  const size_t total_chunks =
      std::max<size_t>(1, (corpus.size() + chunk_bytes - 1) / chunk_bytes);

  std::vector<int32_t> tokens;
  tokens.reserve(corpus.size());

  report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::PROGRESS, 0, 0.0f,
            "Tokenization progress: 0/" + std::to_string(total_chunks));
  for (size_t chunk = 0; chunk < total_chunks; ++chunk) {
    const size_t start = chunk * chunk_bytes;
    const size_t count = std::min(chunk_bytes, corpus.size() - start);
    const std::string piece = corpus.substr(start, count);
    const std::vector<int32_t> piece_tokens = tokenizer.encode(piece);
    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());

    const size_t completed = chunk + 1;
    const size_t pct = (completed * 100) / total_chunks;
    report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::PROGRESS, static_cast<uint32_t>(completed),
              static_cast<float>(pct),
              "Tokenization progress: " + std::to_string(completed) + "/" +
                  std::to_string(total_chunks));
  }

  write_tokens_or_throw(output_binary_path_, tokens, target_vocab_size_);

  report_utils::report_if(sink_, ReportPhase::TOKENIZER, ReportEvent::END, static_cast<uint32_t>(total_chunks), 100.0f,
            "Tokenization complete. tokens written=" +
                std::to_string(tokens.size()) +
                ", dataset=" + output_binary_path_);
}

int run_tokenization_mode(const std::string &config_path, ReportSink *sink) {
  const Config cfg = Config::load_from_file(config_path);
  const std::string vocab_path = resolve_bpe_vocab_path(cfg);
  const std::string merges_path = resolve_bpe_merges_path(cfg);
  TokenizationUtility utility(cfg.tokenization.input_corpus,
                              cfg.tokenization.output_binary, vocab_path,
                              merges_path, cfg.tokenization.chunk_size_mb,
                              cfg.model.target_vocab_size, sink);
  utility.run();

  const std::string details =
      "Status: SUCCESS\n\n"
      "Input Corpus: " + cfg.tokenization.input_corpus + "\n\n"
      "Output Dataset: " + cfg.tokenization.output_binary + "\n\n"
      "Tokenizer: BPETokenizer (Config: " + vocab_path + ")";
  log_operation(cfg.paths.journal_file, "TOKENIZATION_RUN", details);
  return 0;
}
