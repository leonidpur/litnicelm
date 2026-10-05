#include "runtime_config.hpp"

#include "logger.hpp"

#include <tokenizer.hpp>

#include <string>

Config with_tokenizer_vocab_size(const Config &cfg, const Tokenizer &tokenizer) {
  Config result = cfg;
  result.model.vocab_size = static_cast<uint32_t>(tokenizer.vocab_size());
  logger().log(EventId::VocabSizeImported,
               std::to_string(result.model.vocab_size) + " from tokenizer '" +
                   tokenizer.name() + "' (tokenizer.target_vocab_size: " +
                   std::to_string(cfg.tokenizer.target_vocab_size) + ")");
  return result;
}
