#pragma once

#include <config.hpp>

#include <cstdint>
#include <vector>

// Token range [begin, end) of a dataset.
struct TokenRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

// Which tokens of a dataset belong to training and which to validation.
//
// The token axis is cut into chunks of training.validation_chunk_tokens; one
// chunk of every round(1 / training.validation_fraction) is held out for
// validation (the last chunk of each group), spreading validation text over
// the whole file. Depends only on the config and the dataset size, so every
// run and resume, and every dataset built from the same split, agree.
//
// A window belongs to a side only if all tokens it reads are on that side.
class DatasetSplit {
public:
  enum class Side {
    Train,
    Validation,
  };

  // validation_fraction 0 holds out nothing.
  static DatasetSplit from_config(const Config &cfg, uint64_t num_tokens);

  uint64_t num_tokens() const { return numTokens_; }
  const std::vector<TokenRange> &validation_ranges() const {
    return validationRanges_;
  }

  // True when every token of [begin, end) is on `side`.
  bool contains(Side side, uint64_t begin, uint64_t end) const;

private:
  uint64_t numTokens_ = 0;
  std::vector<TokenRange> validationRanges_;
};
