#pragma once

#include "adam_state_factory.hpp"
#include "arena.hpp"
#include "checkpoint.hpp"
#include "gradient_factory.hpp"
#include "memory_resource_info.hpp"
#include "named_layout.hpp"
#include "tensor_factory.hpp"

#include <config.hpp>

#include <cstdint>
#include <memory>

class DeviceBackend;
class TrainingSessionController;

class TrainingMemoryManager {
public:
  TrainingMemoryManager(const Config &cfg, DeviceBackend &backend,
                        TrainingSessionController &session_controller);

  const NamedLayout &param_layout() const;
  const NamedLayout &temp_layout() const;

  const ArenaView &data_arena() const;
  const ArenaView &grad_arena() const;
  const AdamStateView &adam_state() const;
  const TrainingMemoryUsage &memory_usage() const;

  void *temp_base() const;
  uint64_t temp_bytes() const;

  TensorFactory &tensor_factory() const;
  GradientFactory &gradient_factory() const;
  AdamStateFactory &adam_state_factory() const;

private:
  const Config &cfg_;
  DeviceBackend &backend_;
  NamedLayout param_layout_;
  NamedLayout temp_layout_;
  std::unique_ptr<Arena> param_arena_;
  std::unique_ptr<Arena> grad_arena_;
  std::unique_ptr<Arena> adam_arena_;
  std::unique_ptr<Arena> temp_arena_;
  ArenaView data_view_{};
  ArenaView grad_view_{};
  AdamStateView adam_view_{};
  TrainingMemoryUsage memory_usage_{};
  std::unique_ptr<TensorFactory> tensor_factory_;
  std::unique_ptr<GradientFactory> gradient_factory_;
  std::unique_ptr<AdamStateFactory> adam_state_factory_;
};
