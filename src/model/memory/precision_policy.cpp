#include "precision_policy.hpp"

#include "backend/backend_plugin_api.hpp"

#include <litnicelm_utils/checks.hpp>

namespace {

// The presets, in ItemType order: Parameter, ComputeWeight, Gradient,
// OptimizerState, Activation, TokenIds, Loss.
constexpr std::array<DType, static_cast<size_t>(ItemType::Count)> kFp32 = {
    DType::F32, DType::F32, DType::F32, DType::F32,
    DType::F32, DType::I32, DType::F32};
constexpr std::array<DType, static_cast<size_t>(ItemType::Count)> kBf16Mixed = {
    DType::F32, DType::BF16, DType::F32, DType::F32,
    DType::BF16, DType::I32, DType::F32};

} // namespace

PrecisionPolicy PrecisionPolicy::create(const Config &cfg,
                                        uint32_t backend_variants) {
  if (cfg.precision == Precision::Bf16Mixed) {
    // Gate until E3 stores the BF16 compute weights; removing it is E3's
    // first step.
    checker().check(CheckKind::Capability, false, "PrecisionPolicy",
                    "bf16_mixed: ComputeWeight storage not implemented (E3)");
    checker().check(CheckKind::Capability,
                    (backend_variants & kPrecisionVariantBf16Act) != 0,
                    "PrecisionPolicy",
                    "bf16_mixed: backend has no BF16-activation (_bf16act) kernels");
    return PrecisionPolicy(kBf16Mixed);
  }
  checker().check(CheckKind::Capability,
                  (backend_variants & kPrecisionVariantF32) != 0,
                  "PrecisionPolicy", "fp32: backend has no FP32 (_f32) kernels");
  return PrecisionPolicy(kFp32);
}
