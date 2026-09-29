#pragma once

#include <cstdint>

// Running mean of per-batch losses.
class MeanLoss {
public:
  void add(double loss) {
    sum_ += loss;
    count_ += 1;
  }
  // 0 when nothing was added.
  double mean() const {
    return count_ == 0 ? 0.0 : sum_ / static_cast<double>(count_);
  }
  uint64_t count() const { return count_; }

private:
  double sum_ = 0.0;
  uint64_t count_ = 0;
};
