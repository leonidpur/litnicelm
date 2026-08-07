#pragma once

#include "tensor.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Config;

struct LayoutSlice {
  std::string name;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  DType dtype = DType::F32;
};

class NamedLayout {
public:
  const std::vector<LayoutSlice> &slices() const { return slices_; }
  uint64_t total_bytes() const { return total_bytes_; }

  const LayoutSlice *find(const std::string &name) const {
    for (const auto &s : slices_) {
      if (s.name == name) {
        return &s;
      }
    }
    return nullptr;
  }

  static uint64_t align_up(uint64_t v, uint64_t a) {
    if (a == 0) {
      throw std::runtime_error("NamedLayout: alignment must be > 0");
    }
    const uint64_t r = v % a;
    return (r == 0) ? v : (v + (a - r));
  }

  static uint64_t checked_mul(uint64_t a, uint64_t b, const char *what) {
    if (a == 0 || b == 0) {
      return 0;
    }
    if (a > (std::numeric_limits<uint64_t>::max() / b)) {
      throw std::overflow_error(std::string("NamedLayout overflow in ") + what);
    }
    return a * b;
  }

  static uint64_t tensor_bytes(uint64_t rows, uint64_t cols, DType dtype,
                               const char *what) {
    const uint64_t elems = checked_mul(rows, cols, what);
    return checked_mul(elems, static_cast<uint64_t>(dtype_size(dtype)), what);
  }

protected:
  std::vector<LayoutSlice> slices_;
  uint64_t total_bytes_ = 0;

  friend NamedLayout build_param_layout(const Config &cfg);
  friend NamedLayout build_temp_layout(const Config &cfg);
};

NamedLayout build_param_layout(const Config &cfg);
NamedLayout build_temp_layout(const Config &cfg);
