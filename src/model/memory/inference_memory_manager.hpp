#pragma once

#include "arena.hpp"
#include "checkpoint.hpp"
#include "named_layout.hpp"
#include "precision_policy.hpp"
#include "tensor_store.hpp"

#include <config.hpp>

#include <cstdint>
#include <memory>
#include <optional>

class DeviceBackend;

class InferenceMemoryManager {
public:
  // algo and precision must outlive the manager (they live next to the
  // backend).
  InferenceMemoryManager(const Config &cfg,
                         const ModelAlgoFactory &algo,
                         DeviceBackend &backend,
                         const PrecisionPolicy &precision);

  const NamedLayout &param_layout() const;
  const NamedLayout &temp_layout() const;

  const ArenaView &data_arena() const;
  const AdamStateView &adam_state() const;

  void *temp_base() const;
  uint64_t temp_bytes() const;

  TensorStore &tensor_store() const;

private:
  const Config &cfg_;
  DeviceBackend &backend_;
  const PrecisionPolicy &precision_;
  NamedLayout param_layout_;
  NamedLayout temp_layout_;
  // The BF16 GEMM weights; absent when the GEMMs read the parameters.
  std::optional<NamedLayout> compute_layout_;
  std::unique_ptr<Arena> param_arena_;
  std::unique_ptr<Arena> adam_arena_;
  std::unique_ptr<Arena> temp_arena_;
  std::unique_ptr<Arena> compute_arena_;
  ArenaView data_view_{};
  AdamStateView adam_view_{};
  std::unique_ptr<TensorStore> tensor_store_;
};
