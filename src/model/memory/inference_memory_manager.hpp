#pragma once

#include "arena.hpp"
#include "checkpoint.hpp"
#include "named_layout.hpp"
#include "tensor_factory.hpp"

#include <config.hpp>

#include <cstdint>
#include <memory>

class DeviceBackend;

class InferenceMemoryManager {
public:
  InferenceMemoryManager(const Config &cfg, DeviceBackend &backend);

  const NamedLayout &param_layout() const;
  const NamedLayout &temp_layout() const;

  const ArenaView &data_arena() const;
  const AdamStateView &adam_state() const;

  void *temp_base() const;
  uint64_t temp_bytes() const;

  TensorFactory &tensor_factory() const;

private:
  const Config &cfg_;
  DeviceBackend &backend_;
  NamedLayout param_layout_;
  NamedLayout temp_layout_;
  std::unique_ptr<Arena> param_arena_;
  std::unique_ptr<Arena> adam_arena_;
  std::unique_ptr<Arena> temp_arena_;
  ArenaView data_view_{};
  AdamStateView adam_view_{};
  std::unique_ptr<TensorFactory> tensor_factory_;
};
