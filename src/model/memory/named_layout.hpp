#pragma once

#include <litnicelm_utils/layout_slice.hpp>
#include "tensor.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct Config;
class ModelAlgoFactory;
class PrecisionPolicy;


class NamedLayout;

// The layouts of a training run, one per tensor role.
struct TrainingLayouts {
  const NamedLayout &param;
  const NamedLayout &grad;
  const NamedLayout &optimizer;  // one AdamW state (m or v)
  const NamedLayout &temp;
};

// The two contiguous parts of a parameter-shaped layout: slices that get
// weight decay first, then those that do not.
enum class DecayZone : uint8_t { Decay, NoDecay };

class NamedLayout {
public:
  // The parameter shapes with elements of `dtype`; also used for tensors
  // that mirror the parameters (gradients, optimizer state).
  static NamedLayout build_param_layout(const Config &cfg,
                                        const ModelAlgoFactory &algo, DType dtype);
  // The weights the GEMMs read (attention QKV and output, FFN W1 and W2 per
  // layer, the LM head) with elements of `dtype`; allocated only when that
  // differs from the parameter dtype.
  static NamedLayout build_compute_weight_layout(const Config &cfg, DType dtype);
  // Temporaries: activations, token ids and the loss scalar, each in its
  // item type's dtype under `precision`.
  static NamedLayout build_training_temp_layout(const Config &cfg,
                                                const ModelAlgoFactory &algo,
                                                const PrecisionPolicy &precision);
  static NamedLayout build_inference_temp_layout(const Config &cfg,
                                                 const ModelAlgoFactory &algo,
                                                 const PrecisionPolicy &precision);
  static NamedLayout build_temp_layout(const Config &cfg,
                                       const ModelAlgoFactory &algo,
                                       const PrecisionPolicy &precision);

  const std::vector<LayoutSlice> &slices() const { return slices_; }
  uint64_t total_bytes() const { return total_bytes_; }
  uint64_t decay_bytes() const { return decay_bytes_; }
  // The dtype every slice shares; throws if the layout is empty or mixed.
  // For views spanning the whole layout (flat gradient and optimizer zones).
  DType uniform_dtype() const;
  // A flat view of one decay zone of an arena laid out by this layout;
  // empty when the zone holds no slices.
  std::optional<TensorView> zone_view(void *base, Device device,
                                      DecayZone zone) const;

  static uint64_t align_up(uint64_t v, uint64_t a) {
    if (a == 0) {
      throw std::runtime_error("NamedLayout: alignment must be > 0");
    }
    const uint64_t r = v % a;
    return (r == 0) ? v : (v + (a - r));
  }

  static uint64_t checked_mul(uint64_t multiplicand, uint64_t multiplier,
                              const char *what) {
    if (multiplicand == 0 || multiplier == 0) {
      return 0;
    }
    if (multiplicand > (std::numeric_limits<uint64_t>::max() / multiplier)) {
      throw std::overflow_error(std::string("NamedLayout overflow in ") + what);
    }
    return multiplicand * multiplier;
  }

  static uint64_t tensor_bytes(uint64_t rows, uint64_t cols, DType dtype,
                               const char *what) {
    const uint64_t elems = checked_mul(rows, cols, what);
    return checked_mul(elems, static_cast<uint64_t>(dtype_size(dtype)), what);
  }

protected:
  std::vector<LayoutSlice> slices_;
  uint64_t total_bytes_ = 0;
  uint64_t decay_bytes_ = 0;
};
