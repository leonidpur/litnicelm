#pragma once

#include "checkpoint.hpp"
#include "host_tensor_stage.hpp"
#include "named_layout.hpp"
#include <litnicelm_utils/tensor.hpp>

#include <config.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class GradientStore {
public:
  // grad_layout: the parameter shapes in the gradient dtype; the grad arena
  // is laid out by it.
  GradientStore(const Config &cfg, const ModelAlgoFactory &algo,
                const NamedLayout &param_layout, const NamedLayout &grad_layout,
                void *params_base, uint64_t params_bytes,
                const ArenaView &grad_arena);

  TensorView grad_for_param(const TensorView &param) const;
  TensorView full_gradient_view() const;
  // Gradients of one decay zone as a flat view; empty if the zone is.
  std::optional<TensorView> zone_view(DecayZone zone) const;
  std::string first_nonfinite_diagnostic(DeviceBackend &backend) const;

private:
  struct NamedGradientSlot {
    std::string name;
    TensorView grad;
  };

  const Config &cfg_;
  uint8_t *params_base_ = nullptr;
  uint64_t params_bytes_ = 0;
  uint8_t *grad_base_ = nullptr;
  uint64_t grad_bytes_ = 0;
  Device device_ = Device::CPU;
  NamedLayout grad_layout_;
  std::unordered_map<std::string, LayoutSlice> grad_slice_by_name_;
  TensorView full_gradient_view_{};
  std::unordered_map<const void *, TensorView> grad_by_param_data_;
  std::vector<NamedGradientSlot> named_gradient_slots_;

  void build_gradient_views(const NamedLayout &param_layout,
                            const ModelAlgoFactory &algo);
  void register_gradient_slot(const std::string &name, const TensorView &param,
                              const TensorView &grad);
  TensorView make_param_view(const LayoutSlice &s, Shape shape) const;
  // The gradient of the parameter in slice s (looked up by name).
  TensorView make_grad_view(const LayoutSlice &s, Shape shape) const;
  void check_layer(int layer) const;
  std::string lname(int layer, const char *suffix) const;
};
