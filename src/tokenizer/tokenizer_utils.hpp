#pragma once

#include "tokenizer_plugin.hpp"

#include <cstdint>
#include <string>

class ReportSink;

namespace TokenizerUtils {

void stream_tokenize(TokenizerPlugin *plugin, const std::string &input_path,
                     const std::string &output_path, uint32_t chunk_size_mb,
                     const std::string &inter_file_boundary,
                     bool validate_decode_identity, ReportSink *sink);

} // namespace TokenizerUtils
