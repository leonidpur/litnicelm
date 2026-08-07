#include <config.hpp>

#include "config_mapper.hpp"
#include "yaml_parser.hpp"

#include <stdexcept>

Config Config::load_from_file(const std::string &path) {
  Config cfg{};
  const auto entries = parse_yaml_file_flat(path);
  map_config_entries(entries, cfg);
  cfg.model.target_vocab_size = cfg.tokenizer.target_vocab_size;
  validate_config(cfg);
  return cfg;
}
