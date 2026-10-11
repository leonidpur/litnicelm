#pragma once

#include "adam_state_store.hpp"
#include "arena.hpp"
#include "checkpoint.hpp"
#include "gradient_store.hpp"
#include "precision_policy.hpp"
#include "memory_resource_info.hpp"
#include "named_layout.hpp"
#include "tensor_store.hpp"

#include <config.hpp>

#include <cstdint>
#include <memory>

class DeviceBackend;
class TrainingSessionController;

class TrainingMemoryManager {
public:
  // algo must outlive the manager (it lives next to the backend).
  TrainingMemoryManager(const Config &cfg,
                        const ModelAlgoFactory &algo,
                        DeviceBackend &backend,
                        TrainingSessionController &session_controller);

  const PrecisionPolicy &precision() const { return precision_; }
  const NamedLayout &param_layout() const;
  const NamedLayout &temp_layout() const;

  const ArenaView &data_arena() const;
  const ArenaView &grad_arena() const;
  // Gradient sum over a step's batches; empty (bytes 0) at one batch per step.
  const ArenaView &grad_sum_arena() const;
  const AdamStateView &adam_state() const;
  const TrainingMemoryUsage &memory_usage() const;

  void *temp_base() const;
  uint64_t temp_bytes() const;

  TensorStore &tensor_store() const;
  GradientStore &gradient_store() const;
  AdamStateStore &adam_state_store() const;

private:
  const Config &cfg_;
  DeviceBackend &backend_;
  const PrecisionPolicy precision_;
  NamedLayout param_layout_;
  // Parameter-shaped layouts of the gradients and of each AdamW state (m, v),
  // in their own dtypes.
  NamedLayout grad_layout_;
  NamedLayout optimizer_layout_;
  NamedLayout temp_layout_;
  std::unique_ptr<Arena> param_arena_;
  std::unique_ptr<Arena> grad_arena_;
  std::unique_ptr<Arena> grad_sum_arena_;
  std::unique_ptr<Arena> adam_arena_;
  std::unique_ptr<Arena> temp_arena_;
  ArenaView data_view_{};
  ArenaView grad_view_{};
  ArenaView grad_sum_view_{};
  AdamStateView adam_view_{};
  TrainingMemoryUsage memory_usage_{};
  std::unique_ptr<TensorStore> tensor_store_;
  std::unique_ptr<GradientStore> gradient_store_;
  std::unique_ptr<AdamStateStore> adam_state_store_;
};
