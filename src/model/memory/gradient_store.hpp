#pragma once

#include "checkpoint.hpp"
#include "host_tensor_stage.hpp"
#include "named_layout.hpp"
#include "tensor.hpp"

#include <config.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class GradientStore {
public:
  GradientStore(const Config &cfg, const NamedLayout &param_layout,
                  void *params_base, uint64_t params_bytes,
                  const ArenaView &grad_arena);

  TensorView grad_for_param(const TensorView &param) const;
  TensorView full_gradient_view() const;
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
  TensorView full_gradient_view_{};
  std::unordered_map<const void *, TensorView> grad_by_param_data_;
  std::vector<NamedGradientSlot> named_gradient_slots_;

  void build_gradient_views(const NamedLayout &param_layout);
  void register_gradient_slot(const std::string &name, const TensorView &param,
                              const TensorView &grad);
  TensorView make_param_view_f32(const LayoutSlice &s, Shape shape) const;
  TensorView make_grad_view_f32(const LayoutSlice &s, Shape shape) const;
  void check_layer(int layer) const;
  std::string lname(int layer, const char *suffix) const;
};
