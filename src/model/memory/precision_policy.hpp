#pragma once

#include "tensor.hpp"

#include <config.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

// What a tensor holds; each item type has one dtype.
enum class ItemType : uint8_t {
  Parameter,       // model weights (master copy)
  ComputeWeight,   // the weights the forward/backward GEMMs read
  Gradient,        // gradients of the parameters
  OptimizerState,  // AdamW m and v
  Activation,      // forward/backward temporaries and their gradients
  TokenIds,        // input and target token ids
  Loss,            // the loss scalar
  Count
};

// The dtype of each item type under the config's precision preset. Created
// once at setup and passed by const reference to the memory managers, which
// take every layout and arena dtype from it. Immutable; the backend picks
// its kernels from tensor dtypes, not from here.
class PrecisionPolicy {
public:
  // Checks the preset against backend_variants (the backend's
  // kPrecisionVariant* bits) and fails with a Capability error if the
  // backend cannot run it.
  static PrecisionPolicy create(const Config &cfg, uint32_t backend_variants);

  DType dtype(ItemType item) const {
    return by_item_[static_cast<size_t>(item)];
  }
  // True when the GEMMs can read the parameters themselves (no separate
  // compute copy).
  bool compute_weight_aliases_parameter() const {
    return dtype(ItemType::ComputeWeight) == dtype(ItemType::Parameter);
  }

private:
  using Table = std::array<DType, static_cast<size_t>(ItemType::Count)>;
  explicit PrecisionPolicy(const Table &by_item) : by_item_(by_item) {}

  Table by_item_;
};
