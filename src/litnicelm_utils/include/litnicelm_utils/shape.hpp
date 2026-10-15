#pragma once

#include "litnicelm_utils/checks.hpp"
#include "litnicelm_utils/dtype.hpp"
#include "litnicelm_utils/tensor_view_struct.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

constexpr size_t kMaxTensorRank = LITNICELM_MAX_TENSOR_RANK;

class Shape {
public:
  Shape() = default;

  Shape(std::initializer_list<int64_t> dims) {
    init_(dims.begin(), dims.size());
  }

  explicit Shape(const std::vector<int64_t> &dims) {
    init_(dims.begin(), dims.size());
  }

  // The first `rank` entries of dims.
  Shape(size_t rank, const int64_t *dims) { init_(dims, rank); }

private:
  template <typename Iter>
  void init_(Iter begin, size_t size) {
    checker().check(CheckKind::Layout, size <= kMaxTensorRank, "Shape",
                    "rank exceeds kMaxTensorRank");
    rank_ = static_cast<uint8_t>(size);
    size_t i = 0;
    for (Iter it = begin; i < size; ++it) {
      dims_[i++] = *it;
    }
  }

public:
  static Shape scalar() { return Shape{}; }

  size_t rank() const { return rank_; }

  int64_t dim(size_t axis) const {
    checker().check_index(static_cast<int64_t>(axis), rank_, "Shape", "axis");
    return dims_[axis];
  }

  uint64_t numel() const {
    uint64_t out = 1;
    for (size_t i = 0; i < rank_; ++i) {
      checker().check(CheckKind::Contract, dims_[i] >= 0, "Shape",
                      "negative dimension");
      out *= static_cast<uint64_t>(dims_[i]);
    }
    return out;
  }

  const std::array<int64_t, kMaxTensorRank> &dims() const { return dims_; }

  uint8_t rank_ = 0;
  std::array<int64_t, kMaxTensorRank> dims_{};
};

inline uint64_t numel(const Shape &s) { return s.numel(); }

inline uint64_t nbytes(const Shape &s, DType dt) {
  const uint64_t elems = numel(s);
  const uint64_t esz = static_cast<uint64_t>(dtype_size(dt));
  checker().check(CheckKind::Contract, esz != 0, "nbytes", "unknown dtype");
  checker().check(CheckKind::Layout, elems == 0 || esz <= UINT64_MAX / elems,
                  "nbytes", "size overflows 64 bits");
  return elems * esz;
}
