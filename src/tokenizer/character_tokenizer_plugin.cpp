#include "character_tokenizer_plugin.hpp"

#include <filesystem>

CharacterTokenizerPlugin::CharacterTokenizerPlugin(uint32_t vocab_size)
    : tokenizer_(vocab_size) {}

void CharacterTokenizerPlugin::train(const std::string &corpus_path,
                                     const std::string &artifacts_dir,
                                     uint32_t target_vocab_size,
                                     ReportSink *sink) {
  (void)corpus_path;
  (void)target_vocab_size;
  (void)sink;
  if (!artifacts_dir.empty()) {
    std::filesystem::create_directories(artifacts_dir);
  }
}

bool CharacterTokenizerPlugin::load(const std::string &artifacts_dir) {
  (void)artifacts_dir;
  return true;
}

std::vector<int32_t>
CharacterTokenizerPlugin::encode(const std::string &text) const {
  return tokenizer_.encode(text);
}

std::string
CharacterTokenizerPlugin::decode(const std::vector<int32_t> &ids) const {
  return tokenizer_.decode(ids);
}

int32_t CharacterTokenizerPlugin::vocab_size() const {
  return tokenizer_.vocab_size();
}

const char *CharacterTokenizerPlugin::name() const {
  return "CharacterTokenizerPlugin(bytes)";
}
