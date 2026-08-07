#pragma once

#include <tokenizer.hpp>

#include <stdexcept>

// Simple byte-level tokenizer (NOT unicode codepoints):
// - Treats input string as bytes (UTF-8 bytes).
// - vocab_size = 256
// - encode: each byte -> id
// - decode: each id -> byte
class CharacterTokenizer final : public Tokenizer {
public:
  CharacterTokenizer() = default;

  std::vector<int32_t> encode(const std::string &text) const override {
    std::vector<int32_t> out;
    out.reserve(text.size());
    for (unsigned char b : text) {
      out.push_back(static_cast<int32_t>(b));
    }
    return out;
  }

  std::string decode(const std::vector<int32_t> &ids) const override {
    std::string out;
    out.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      const int32_t id = ids[i];
      if (id < 0 || id > 255) {
        throw std::runtime_error("CharacterTokenizer::decode: id out of range");
      }
      out[i] = static_cast<char>(static_cast<uint8_t>(id));
    }
    return out;
  }

  int32_t vocab_size() const override { return 256; }

  const char *name() const override { return "CharacterTokenizer(bytes)"; }
};
