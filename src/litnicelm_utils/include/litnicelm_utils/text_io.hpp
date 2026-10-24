#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// Whole text files in one call; errors throw std::runtime_error naming the
// file.
namespace text_io {

// Writes text to path, replacing the file.
inline void write_text_file(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::trunc);
  out << text;
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
}

// The file's whole contents, byte for byte.
inline std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  if (!in) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return text.str();
}

} // namespace text_io
