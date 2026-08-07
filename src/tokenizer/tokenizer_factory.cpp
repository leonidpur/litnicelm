#include <tokenizer_factory.hpp>

#include "sentencepiece_plugin.hpp"
#include "tokenizer_plugin.hpp"
#include <config.hpp>
#include "character_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
std::string resolve_bpe_artifacts_dir(const Config &cfg) {
  if (!cfg.tokenizer.bpe_artifacts_dir.empty()) {
    return cfg.tokenizer.bpe_artifacts_dir;
  }
  if (!cfg.tokenizer.bpe_vocab_file.empty()) {
    return std::filesystem::path(cfg.tokenizer.bpe_vocab_file)
        .parent_path()
        .string();
  }
  if (!cfg.tokenizer.bpe_merges_file.empty()) {
    return std::filesystem::path(cfg.tokenizer.bpe_merges_file)
        .parent_path()
        .string();
  }
  return "";
}
} // namespace

std::unique_ptr<TokenizerPlugin> TokenizerFactory::create_plugin(
    const Config &cfg, ReportSink *sink) {
  (void)sink;
  std::string type = cfg.tokenizer.type;
  std::transform(type.begin(), type.end(), type.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (type == "bpe" || type == "sentencepiece" || type == "spm") {
    std::cout << "[TOKENIZER_FACTORY] plugin selection: tokenizer.type="
              << cfg.tokenizer.type
              << " resolved_type=" << type
              << " plugin=SentencePiecePlugin\n";
    return std::make_unique<SentencePiecePlugin>();
  }

  throw std::runtime_error("TokenizerFactory: tokenizer.type does not support plugin mode: " +
                           cfg.tokenizer.type);
}

std::unique_ptr<Tokenizer> TokenizerFactory::create(const Config &cfg,
                                                    ReportSink *sink) {
  std::string type = cfg.tokenizer.type;
  std::transform(type.begin(), type.end(), type.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (type == "character" || type == "char" || type == "bytes") {
    std::cout << "[TOKENIZER_FACTORY] runtime selection: tokenizer.type="
              << cfg.tokenizer.type
              << " resolved_type=" << type
              << " tokenizer=CharacterTokenizer\n";
    return std::make_unique<CharacterTokenizer>();
  }

  if (type == "bpe" || type == "sentencepiece" || type == "spm") {
    const std::string artifacts_dir = resolve_bpe_artifacts_dir(cfg);
    std::cout << "[TOKENIZER_FACTORY] runtime selection: tokenizer.type="
              << cfg.tokenizer.type
              << " resolved_type=" << type
              << " artifacts_dir=" << artifacts_dir << "\n";
    if (artifacts_dir.empty()) {
      throw std::runtime_error(
          "TokenizerFactory: tokenizer.bpe_artifacts_dir is required for tokenizer.type=bpe");
    }

    auto tok = create_plugin(cfg, sink);
    std::cout << "[TOKENIZER_FACTORY] runtime load: plugin=" << tok->name()
              << " from " << artifacts_dir << "\n";
    if (!tok->load(artifacts_dir)) {
      throw std::runtime_error(
          "TokenizerFactory: failed to load tokenizer plugin artifacts from: " +
          artifacts_dir);
    }
    return tok;
  }

  throw std::runtime_error("TokenizerFactory: unknown tokenizer.type: " + cfg.tokenizer.type);
}
