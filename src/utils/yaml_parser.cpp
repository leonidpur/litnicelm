#include "yaml_parser.hpp"
#include "string_utils.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<YamlEntry> parse_yaml_file_flat(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Config::load_from_file: failed to open " + path);
  }

  struct YamlLevel {
    int indent = 0;
    std::string key;
  };

  std::vector<YamlEntry> entries;
  std::vector<YamlLevel> stack;

  std::string line;
  uint64_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const size_t hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }

    size_t indent = 0;
    while (indent < line.size() && (line[indent] == ' ' || line[indent] == '\t')) {
      ++indent;
    }

    const std::string stripped = string_utils::trim_copy(line);
    if (stripped.empty()) {
      continue;
    }

    const size_t sep = stripped.find(':');
    if (sep == std::string::npos) {
      throw std::runtime_error("Config::load_from_file: invalid yaml line " +
                               std::to_string(lineno));
    }

    const std::string key = string_utils::trim_copy(stripped.substr(0, sep));
    const std::string raw = string_utils::trim_copy(stripped.substr(sep + 1));

    while (!stack.empty() && static_cast<int>(indent) <= stack.back().indent) {
      stack.pop_back();
    }

    std::string full_key;
    for (const auto &level : stack) {
      if (!full_key.empty()) {
        full_key.push_back('.');
      }
      full_key += level.key;
    }
    if (!full_key.empty()) {
      full_key.push_back('.');
    }
    full_key += key;

    if (raw.empty()) {
      stack.push_back({static_cast<int>(indent), key});
      continue;
    }

    entries.push_back(YamlEntry{full_key, string_utils::unquote_copy(raw), lineno});
  }

  return entries;
}
