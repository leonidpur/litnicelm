#pragma once

#include <report_interface.hpp>

#include <cstdint>
#include <string>

class BPE_TrainingUtility final {
public:
  BPE_TrainingUtility(std::string corpus_path, std::string artifacts_dir,
                      uint32_t target_vocab_size, bool run_validation,
                      int32_t bpe_validation_num_threads,
                      uint32_t bpe_validation_sample_rate,
                      ReportSink *sink = nullptr);

  void run() const;

private:
  std::string corpus_path_;
  std::string artifacts_dir_;
  uint32_t target_vocab_size_ = 256;
  bool run_validation_ = true;
  int32_t bpe_validation_num_threads_ = 0;
  uint32_t bpe_validation_sample_rate_ = 100;
  ReportSink *sink_ = nullptr;
};

int run_create_bpe_mode(const std::string &config_path, ReportSink *sink = nullptr);
