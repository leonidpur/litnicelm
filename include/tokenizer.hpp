#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Tokenizer {
public:
  virtual ~Tokenizer() = default;

  // Convert UTF-8 text -> token ids.
  virtual std::vector<int32_t> encode(const std::string &text) const = 0;

  // Convert token ids -> UTF-8 text.
  virtual std::string decode(const std::vector<int32_t> &ids) const = 0;

  // Vocabulary size (ids are in [0, vocab_size)).
  virtual int32_t vocab_size() const = 0;

  // Optional: name for logging.
  virtual const char *name() const = 0;
};
