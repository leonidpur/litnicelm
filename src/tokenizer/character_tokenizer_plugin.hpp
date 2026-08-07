#pragma once

#include "character_tokenizer.hpp"
#include "tokenizer_plugin.hpp"

#include <cstdint>
#include <string>

class CharacterTokenizerPlugin final : public TokenizerPlugin {
public:
  explicit CharacterTokenizerPlugin(uint32_t vocab_size);

  void train(const std::string &corpus_path, const std::string &artifacts_dir,
             uint32_t target_vocab_size, ReportSink *sink) override;
  bool load(const std::string &artifacts_dir) override;

  std::vector<int32_t> encode(const std::string &text) const override;
  std::string decode(const std::vector<int32_t> &ids) const override;
  int32_t vocab_size() const override;
  const char *name() const override;

private:
  CharacterTokenizer tokenizer_;
};
