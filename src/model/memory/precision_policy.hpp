#pragma once

#include "tensor.hpp"

#include <config.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

// What a tensor holds; each role has one dtype.
enum class TensorRole : uint8_t {
  Parameter,       // model weights (master copy)
  Gradient,        // gradients of the parameters
  OptimizerState,  // AdamW m and v
  Activation,      // forward/backward temporaries
  Count
};

// The single place that decides which dtype each tensor role uses; layouts
// and stores take their dtypes from here. Currently FP32 for every role.
class PrecisionPolicy {
public:
  static PrecisionPolicy from_config(const Config &cfg) {
    (void)cfg;
    PrecisionPolicy policy;
    policy.by_role_.fill(DType::F32);
    return policy;
  }

  DType dtype(TensorRole role) const {
    return by_role_[static_cast<size_t>(role)];
  }

private:
  std::array<DType, static_cast<size_t>(TensorRole::Count)> by_role_{};
};
