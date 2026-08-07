#pragma once

#include "tokenizer_plugin.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class OnlySeenCharsTokenizerPlugin final : public TokenizerPlugin {
public:
  explicit OnlySeenCharsTokenizerPlugin(uint32_t vocab_size_limit);

  void train(const std::string &corpus_path, const std::string &artifacts_dir,
             uint32_t target_vocab_size, ReportSink *sink) override;
  bool load(const std::string &artifacts_dir) override;

  std::vector<int32_t> encode(const std::string &text) const override;
  std::string decode(const std::vector<int32_t> &ids) const override;
  int32_t vocab_size() const override;
  const char *name() const override;

private:
  uint32_t vocab_size_limit_ = 0;
  std::vector<std::string> vocab_;
  std::unordered_map<std::string, int32_t> char_to_id_;

  void rebuild_index_();
};
