#include "corpus_input.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool has_wildcard(const std::string &s) {
  return s.find('*') != std::string::npos || s.find('?') != std::string::npos;
}

bool wildcard_match(const std::string &pattern, const std::string &text) {
  size_t p = 0;
  size_t t = 0;
  size_t star = std::string::npos;
  size_t match = 0;
  while (t < text.size()) {
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
      ++p;
      ++t;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star = p++;
      match = t;
    } else if (star != std::string::npos) {
      p = star + 1;
      t = ++match;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') {
    ++p;
  }
  return p == pattern.size();
}

void sort_files(std::vector<std::string> &files) {
  std::sort(files.begin(), files.end());
}

std::vector<std::string> resolve_directory(const fs::path &dir) {
  std::vector<std::string> files;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path().string());
    }
  }
  sort_files(files);
  return files;
}

std::string unique_temp_path(const std::string &name_hint) {
  const auto now = std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count();
  const std::string safe_hint = name_hint.empty() ? "corpus" : name_hint;
  return (fs::temp_directory_path() /
          ("litnicelm_" + safe_hint + "_" + std::to_string(now) + ".txt"))
      .string();
}

} // namespace

namespace CorpusInput {

std::vector<std::string> resolve_files(const std::string &corpus_spec) {
  if (corpus_spec.empty()) {
    throw std::runtime_error("CorpusInput: corpus path is required");
  }

  const fs::path spec(corpus_spec);
  std::vector<std::string> files;

  if (!has_wildcard(corpus_spec)) {
    if (fs::is_regular_file(spec)) {
      files.push_back(spec.string());
    } else if (fs::is_directory(spec)) {
      files = resolve_directory(spec);
    } else {
      throw std::runtime_error("CorpusInput: corpus path does not exist: " +
                               corpus_spec);
    }
  } else {
    const fs::path parent =
        spec.parent_path().empty() ? fs::path(".") : spec.parent_path();
    const std::string pattern = spec.filename().string();
    if (has_wildcard(parent.string())) {
      throw std::runtime_error(
          "CorpusInput: wildcards are supported only in the filename part: " +
          corpus_spec);
    }
    if (!fs::is_directory(parent)) {
      throw std::runtime_error("CorpusInput: wildcard parent is not a directory: " +
                               parent.string());
    }
    for (const auto &entry : fs::directory_iterator(parent)) {
      if (entry.is_regular_file() &&
          wildcard_match(pattern, entry.path().filename().string())) {
        files.push_back(entry.path().string());
      }
    }
    sort_files(files);
  }

  if (files.empty()) {
    throw std::runtime_error("CorpusInput: corpus resolved to no files: " +
                             corpus_spec);
  }
  return files;
}

std::string describe_files(const std::vector<std::string> &files) {
  if (files.empty()) {
    return "files=0";
  }
  std::ostringstream oss;
  oss << "files=" << files.size() << ", first=" << files.front();
  if (files.size() > 1) {
    oss << ", last=" << files.back();
  }
  return oss.str();
}

void for_each_text_chunk(const std::vector<std::string> &files,
                         const std::string &inter_file_boundary,
                         size_t chunk_bytes,
                         const std::function<void(const std::string &)> &fn) {
  if (chunk_bytes == 0) {
    throw std::runtime_error("CorpusInput: chunk_bytes must be > 0");
  }
  std::vector<char> buffer(chunk_bytes);
  for (size_t i = 0; i < files.size(); ++i) {
    if (i != 0 && !inter_file_boundary.empty()) {
      fn(inter_file_boundary);
    }

    std::ifstream in(files[i], std::ios::binary);
    if (!in) {
      throw std::runtime_error("CorpusInput: failed to open corpus file: " +
                               files[i]);
    }
    while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) ||
           in.gcount() > 0) {
      const size_t bytes_read = static_cast<size_t>(in.gcount());
      if (bytes_read == 0) {
        break;
      }
      fn(std::string(buffer.data(), bytes_read));
    }
  }
}

std::string materialize_merged_temp_file(const std::vector<std::string> &files,
                                         const std::string &inter_file_boundary,
                                         const std::string &name_hint) {
  const std::string path = unique_temp_path(name_hint);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("CorpusInput: failed to create temp corpus: " +
                             path);
  }
  for_each_text_chunk(files, inter_file_boundary, 4u * 1024u * 1024u,
                      [&](const std::string &chunk) { out.write(chunk.data(), static_cast<std::streamsize>(chunk.size())); });
  if (!out) {
    throw std::runtime_error("CorpusInput: failed to write temp corpus: " +
                             path);
  }
  return path;
}

bool needs_merge_file(const std::vector<std::string> &files) {
  return files.size() > 1;
}

} // namespace CorpusInput
