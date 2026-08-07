#include "tokenizer_utils.hpp"

#include <report_interface.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
constexpr uint32_t kDatasetMagic = 0x4C4E4750u; // "LNGP"
constexpr uint32_t kDatasetVersion = 1u;
constexpr uint32_t kDefaultChunkSizeMb = 64u;

void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TOKENIZER, event, step, value,
                          message);
}
} // namespace

namespace TokenizerUtils {

void stream_tokenize(TokenizerPlugin *plugin, const std::string &input_path,
                     const std::string &output_path, uint32_t chunk_size_mb,
                     ReportSink *sink) {
  if (plugin == nullptr) {
    throw std::runtime_error("TokenizerUtils::stream_tokenize: plugin is null");
  }
  if (input_path.empty()) {
    throw std::runtime_error(
        "TokenizerUtils::stream_tokenize: input_path is required");
  }
  if (output_path.empty()) {
    throw std::runtime_error(
        "TokenizerUtils::stream_tokenize: output_path is required");
  }

  const uint32_t vocab_size = static_cast<uint32_t>(plugin->vocab_size());
  if (vocab_size == 0) {
    throw std::runtime_error(
        "TokenizerUtils::stream_tokenize: plugin vocab_size must be > 0");
  }

  std::ifstream in(input_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(
        "TokenizerUtils::stream_tokenize: failed to open input corpus: " +
        input_path);
  }

  const fs::path out_path(output_path);
  if (!out_path.parent_path().empty()) {
    fs::create_directories(out_path.parent_path());
  }

  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error(
        "TokenizerUtils::stream_tokenize: failed to open output dataset: " +
        output_path);
  }

  const uint32_t header[3] = {kDatasetMagic, kDatasetVersion, vocab_size};
  out.write(reinterpret_cast<const char *>(header),
            static_cast<std::streamsize>(sizeof(header)));

  const uint32_t effective_chunk_mb =
      std::max<uint32_t>(chunk_size_mb, kDefaultChunkSizeMb);
  const size_t chunk_bytes =
      static_cast<size_t>(effective_chunk_mb) * 1024u * 1024u;
  std::vector<char> buffer(chunk_bytes);

  uint64_t total_tokens = 0;
  uint32_t chunk_index = 0;
  report_if(sink, ReportEvent::START, 0, 0.0f,
            "Streaming tokenization started: input=" + input_path +
                ", output=" + output_path + ", chunk_size_mb=" +
                std::to_string(effective_chunk_mb));

  while (in.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) ||
         in.gcount() > 0) {
    const size_t bytes_read = static_cast<size_t>(in.gcount());
    if (bytes_read == 0) {
      break;
    }

    const std::string chunk(buffer.data(), bytes_read);
    std::vector<int32_t> tokens = plugin->encode(chunk);
    for (int32_t token : tokens) {
      if (token < 0 || static_cast<uint32_t>(token) >= vocab_size) {
        throw std::runtime_error(
            "TokenizerUtils::stream_tokenize: token id out of range");
      }
    }

    if (!tokens.empty()) {
      out.write(reinterpret_cast<const char *>(tokens.data()),
                static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
    }

    total_tokens += static_cast<uint64_t>(tokens.size());
    ++chunk_index;
    report_if(sink, ReportEvent::PROGRESS, chunk_index, 0.0f,
              "Tokenized chunk " + std::to_string(chunk_index) +
                  ": bytes=" + std::to_string(bytes_read) +
                  ", tokens=" + std::to_string(tokens.size()));
  }

  if (!out) {
    throw std::runtime_error(
        "TokenizerUtils::stream_tokenize: failed while writing dataset: " +
        output_path);
  }

  report_if(sink, ReportEvent::END, chunk_index, 100.0f,
            "Streaming tokenization complete: chunks=" +
                std::to_string(chunk_index) + ", tokens=" +
                std::to_string(total_tokens));
}

} // namespace TokenizerUtils
