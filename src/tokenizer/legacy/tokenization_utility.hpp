#pragma once

#include <report_interface.hpp>

#include <cstdint>
#include <string>

class TokenizationUtility final {
public:
  TokenizationUtility(std::string input_corpus_path, std::string output_binary_path,
                      std::string vocab_path, std::string merges_path,
                      uint32_t chunk_size_mb, uint32_t target_vocab_size,
                      ReportSink *sink = nullptr);

  void run() const;

private:
  std::string input_corpus_path_;
  std::string output_binary_path_;
  std::string vocab_path_;
  std::string merges_path_;
  uint32_t chunk_size_mb_ = 64;
  uint32_t target_vocab_size_ = 0;
  ReportSink *sink_ = nullptr;
};

int run_tokenization_mode(const std::string &config_path, ReportSink *sink = nullptr);
