// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <tokenizer.hpp>

#include <cstdint>
#include <string>

class TokenizerPlugin : public Tokenizer {
public:
  ~TokenizerPlugin() override = default;

  virtual void train(const std::string &corpus_path,
                     const std::string &artifacts_dir,
                     uint32_t target_vocab_size) = 0;

  virtual bool load(const std::string &artifacts_dir) = 0;
};
