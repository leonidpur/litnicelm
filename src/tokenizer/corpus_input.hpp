// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace CorpusInput {

std::vector<std::string> resolve_files(const std::string &corpus_spec);

std::string describe_files(const std::vector<std::string> &files);

void for_each_text_chunk(const std::vector<std::string> &files,
                         const std::string &inter_file_boundary,
                         size_t chunk_bytes,
                         const std::function<void(const std::string &)> &fn);

std::string materialize_merged_temp_file(const std::vector<std::string> &files,
                                         const std::string &inter_file_boundary,
                                         const std::string &name_hint);

bool needs_merge_file(const std::vector<std::string> &files);

} // namespace CorpusInput
