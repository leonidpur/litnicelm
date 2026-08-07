#pragma once

#include "tokenizer_plugin.hpp"

#include <memory>
#include <string>
#include <vector>

class SentencePiecePlugin final : public TokenizerPlugin {
public:
  SentencePiecePlugin();
  ~SentencePiecePlugin() override;

  void train(const std::string &corpus_path, const std::string &artifacts_dir,
             uint32_t target_vocab_size, ReportSink *sink) override;
  bool load(const std::string &artifacts_dir) override;

  std::vector<int32_t> encode(const std::string &text) const override;
  std::string decode(const std::vector<int32_t> &ids) const override;
  int32_t vocab_size() const override;
  const char *name() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
