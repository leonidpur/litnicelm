#include <tokenizer_factory.hpp>

#include "character_tokenizer_plugin.hpp"
#include "only_seen_chars_tokenizer_plugin.hpp"
#include "sentencepiece_plugin.hpp"
#include "tokenizer_plugin.hpp"
#include "logger.hpp"
#include <config.hpp>
#include "character_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
namespace fs = std::filesystem;

std::string resolve_bpe_artifacts_dir(const Config &cfg) {
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

std::string tokenizer_load_failure_message(const std::string &type,
                                           const std::string &artifacts_dir) {
  std::ostringstream oss;
  oss << "TokenizerFactory: failed to load tokenizer plugin artifacts from: "
      << artifacts_dir;

  if (artifacts_dir.empty()) {
    return oss.str();
  }

  std::error_code ec;
  const fs::path dir(artifacts_dir);
  if (!fs::exists(dir, ec)) {
    oss << " (artifacts_dir does not exist)";
    return oss.str();
  }
  if (!fs::is_directory(dir, ec)) {
    oss << " (artifacts_dir is not a directory)";
    return oss.str();
  }

  if (type == "only_seen_chars" || type == "char_seen" || type == "seen_chars") {
    const fs::path artifact = dir / "char_seen.json";
    if (!fs::exists(artifact, ec)) {
      oss << " (missing artifact file: " << artifact.string() << ")";
    } else {
      oss << " (artifact file exists but could not be loaded: "
          << artifact.string() << ")";
    }
    return oss.str();
  }

  if (type == "bpe" || type == "sentencepiece" || type == "spm") {
#if LITNICELM_HAVE_SENTENCEPIECE
    const fs::path model = dir / "spm.model";
    if (!fs::exists(model, ec)) {
      oss << " (missing artifact file: " << model.string() << ")";
    } else {
      oss << " (artifact file exists but could not be loaded: "
          << model.string() << ")";
    }
#else
    oss << " (SentencePiece support is not available in this build)";
#endif
    return oss.str();
  }

  return oss.str();
}
} // namespace

std::unique_ptr<TokenizerPlugin> TokenizerFactory::create_plugin(
    const Config &cfg) {
  std::string type = cfg.tokenizer.type;
  std::transform(type.begin(), type.end(), type.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (type == "bpe" || type == "sentencepiece" || type == "spm") {
    logger().log(EventId::TokenizerPluginSelected,
                 "tokenizer.type=" + cfg.tokenizer.type + " resolved_type=" +
                     type + " plugin=SentencePiecePlugin");
    return std::make_unique<SentencePiecePlugin>();
  }

  if (type == "character" || type == "char" || type == "bytes") {
    logger().log(EventId::TokenizerPluginSelected,
                 "tokenizer.type=" + cfg.tokenizer.type + " resolved_type=" +
                     type + " plugin=CharacterTokenizerPlugin vocab_size=" +
                     std::to_string(cfg.tokenizer.target_vocab_size));
    return std::make_unique<CharacterTokenizerPlugin>(
        cfg.tokenizer.target_vocab_size);
  }

  if (type == "only_seen_chars" || type == "char_seen" ||
      type == "seen_chars") {
    logger().log(EventId::TokenizerPluginSelected,
                 "tokenizer.type=" + cfg.tokenizer.type + " resolved_type=" +
                     type + " plugin=OnlySeenCharsTokenizer vocab_size=" +
                     std::to_string(cfg.tokenizer.target_vocab_size));
    return std::make_unique<OnlySeenCharsTokenizerPlugin>(
        cfg.tokenizer.target_vocab_size);
  }

  throw std::runtime_error("TokenizerFactory: tokenizer.type does not support plugin mode: " +
                           cfg.tokenizer.type);
}

std::unique_ptr<Tokenizer> TokenizerFactory::create(const Config &cfg) {
  std::string type = cfg.tokenizer.type;
  std::transform(type.begin(), type.end(), type.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (type == "character" || type == "char" || type == "bytes") {
    logger().log(EventId::TokenizerRuntimeSelected,
                 "tokenizer.type=" + cfg.tokenizer.type + " resolved_type=" +
                     type + " tokenizer=CharacterTokenizer vocab_size=" +
                     std::to_string(cfg.tokenizer.target_vocab_size));
    return std::make_unique<CharacterTokenizer>(cfg.tokenizer.target_vocab_size);
  }

  if (type == "bpe" || type == "sentencepiece" || type == "spm") {
    const std::string artifacts_dir = resolve_bpe_artifacts_dir(cfg);
    logger().log(EventId::TokenizerRuntimeSelected,
                 "tokenizer.type=" + cfg.tokenizer.type + " resolved_type=" +
                     type + " artifacts_dir=" + artifacts_dir);
    if (artifacts_dir.empty()) {
      throw std::runtime_error(
          "TokenizerFactory: tokenizer.bpe_artifacts_dir is required for tokenizer.type=bpe");
    }

    auto tok = create_plugin(cfg);
    logger().log(EventId::TokenizerLoaded, "plugin=" + std::string(tok->name()) +
                                               " from " + artifacts_dir);
    if (!tok->load(artifacts_dir)) {
      throw std::runtime_error(
          tokenizer_load_failure_message(type, artifacts_dir));
    }
    return tok;
  }

  if (type == "only_seen_chars" || type == "char_seen" ||
      type == "seen_chars") {
    const std::string artifacts_dir = resolve_bpe_artifacts_dir(cfg);
    logger().log(EventId::TokenizerRuntimeSelected,
                 "tokenizer.type=" + cfg.tokenizer.type + " resolved_type=" +
                     type + " artifacts_dir=" + artifacts_dir);
    if (artifacts_dir.empty()) {
      throw std::runtime_error(
          "TokenizerFactory: tokenizer.artifacts_dir is required for tokenizer.type=only_seen_chars");
    }
    auto tok = std::make_unique<OnlySeenCharsTokenizerPlugin>(
        cfg.tokenizer.target_vocab_size);
    logger().log(EventId::TokenizerLoaded, "plugin=" + std::string(tok->name()) +
                                               " from " + artifacts_dir);
    if (!tok->load(artifacts_dir)) {
      throw std::runtime_error(
          tokenizer_load_failure_message(type, artifacts_dir));
    }
    return tok;
  }

  throw std::runtime_error("TokenizerFactory: unknown tokenizer.type: " + cfg.tokenizer.type);
}
