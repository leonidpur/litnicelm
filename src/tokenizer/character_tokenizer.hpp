#pragma once

#include <tokenizer.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

// Simple byte-level tokenizer (NOT unicode codepoints):
// - Treats input string as bytes (UTF-8 bytes).
// - vocab_size caps the accepted byte/id range
// - encode: each byte -> id
// - decode: each id -> byte
class CharacterTokenizer final : public Tokenizer {
public:
  explicit CharacterTokenizer(uint32_t vocab_size = 256)
      : vocab_size_(vocab_size) {}

  std::vector<int32_t> encode(const std::string &text) const override {
    std::vector<int32_t> out;
    out.reserve(text.size());
    for (unsigned char b : text) {
      if (static_cast<uint32_t>(b) >= vocab_size_) {
        throw std::runtime_error(
            "CharacterTokenizer::encode: byte " +
            std::to_string(static_cast<uint32_t>(b)) +
            " is outside vocab_size=" + std::to_string(vocab_size_));
      }
      out.push_back(static_cast<int32_t>(b));
    }
    return out;
  }

  std::string decode(const std::vector<int32_t> &ids) const override {
    std::string out;
    out.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      const int32_t id = ids[i];
      if (id < 0 || static_cast<uint32_t>(id) >= vocab_size_) {
        throw std::runtime_error("CharacterTokenizer::decode: id out of range");
      }
      out[i] = static_cast<char>(static_cast<uint8_t>(id));
    }
    return out;
  }

  int32_t vocab_size() const override { return static_cast<int32_t>(vocab_size_); }

  const char *name() const override { return "CharacterTokenizer(bytes)"; }

private:
  uint32_t vocab_size_ = 256;
};
