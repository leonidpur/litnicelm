#pragma once

#include "backend/backend_plugin_api.hpp"
#include "tensor.hpp"

#include <config.hpp>
#include <litnicelm_utils/checks.hpp>

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
// and stores take their dtypes from here. Currently FP32 for every role,
// which runs the backend's _f32 kernels.
class PrecisionPolicy {
public:
  // backend_variants: the backend's kPrecisionVariant* bits.
  static PrecisionPolicy from_config(const Config &cfg,
                                     uint32_t backend_variants) {
    (void)cfg;
    checker().check(CheckKind::Capability,
                    (backend_variants & kPrecisionVariantF32) != 0,
                    "PrecisionPolicy", "backend has no FP32 (_f32) kernels");
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
