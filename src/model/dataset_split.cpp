#include "dataset_split.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

DatasetSplit DatasetSplit::from_config(const Config &cfg, uint64_t num_tokens) {
  DatasetSplit split;
  split.numTokens_ = num_tokens;

  const float fraction = cfg.training.validation_fraction;
  if (fraction == 0.0f) {
    return split;
  }
  if (!(fraction > 0.0f && fraction <= 0.5f)) {
    throw std::runtime_error(
        "DatasetSplit: training.validation_fraction must be in [0, 0.5], got " +
        std::to_string(fraction));
  }
  const uint64_t chunk = cfg.training.validation_chunk_tokens;
  const uint64_t window_tokens =
      static_cast<uint64_t>(cfg.training.train_seq_len) + 1;
  if (chunk < window_tokens) {
    throw std::runtime_error(
        "DatasetSplit: training.validation_chunk_tokens (" +
        std::to_string(chunk) + ") must hold one window of train_seq_len + 1 (" +
        std::to_string(window_tokens) + ") tokens");
  }

  const uint64_t group =
      static_cast<uint64_t>(std::lround(1.0 / static_cast<double>(fraction)));
  const uint64_t num_chunks = (num_tokens + chunk - 1) / chunk;
  for (uint64_t i = group - 1; i < num_chunks; i += group) {
    const uint64_t begin = i * chunk;
    const uint64_t end = std::min(begin + chunk, num_tokens);
    split.validationRanges_.push_back(TokenRange{begin, end});
  }
  if (split.validationRanges_.empty()) {
    throw std::runtime_error(
        "DatasetSplit: dataset of " + std::to_string(num_tokens) +
        " tokens is too small to hold out 1 of every " + std::to_string(group) +
        " chunks of " + std::to_string(chunk) +
        " tokens; lower training.validation_chunk_tokens");
  }
  return split;
}

bool DatasetSplit::contains(Side side, uint64_t begin, uint64_t end) const {
  if (end > numTokens_ || begin >= end) {
    return false;
  }
  for (const TokenRange &r : validationRanges_) {
    const bool overlaps = begin < r.end && r.begin < end;
    if (side == Side::Validation) {
      if (begin >= r.begin && end <= r.end) {
        return true;
      }
    } else if (overlaps) {
      return false;
    }
  }
  return side == Side::Train;
}
