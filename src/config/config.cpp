#include <config.hpp>

#include "config_mapper.hpp"
#include "yaml_parser.hpp"

#include <stdexcept>

Config Config::load_from_file(const std::string &path) {
  Config cfg;
  const auto entries = parse_yaml_file_flat(path);
  bool has_target_vocab_size = false;
  for (const auto &entry : entries) {
    if (entry.key == "model.target_vocab_size") {
      has_target_vocab_size = true;
      break;
    }
  }
  if (!has_target_vocab_size) {
    throw std::runtime_error(
        "Config::load_from_file: required key missing: model.target_vocab_size");
  }
  map_config_entries(entries, cfg);
  validate_config(cfg);
  return cfg;
}
