#pragma once

#include <tokenizer.hpp>

#include <memory>
#include <string>

struct Config;
class TokenizerPlugin;

class TokenizerFactory final {
public:
  static std::unique_ptr<Tokenizer> create(const Config &cfg);
  static std::unique_ptr<TokenizerPlugin> create_plugin(const Config &cfg);
};

int run_tokenizer_training_mode(const std::string &config_path);
int run_tokenization_mode(const std::string &config_path);
