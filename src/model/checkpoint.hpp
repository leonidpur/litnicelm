#pragma once

#include "backend/device_backend.hpp"
#include <config.hpp>
#include "tensor.hpp"

#include <cstdint>
#include <string>

struct ArenaView {
  void *base = nullptr;
  uint64_t bytes = 0;
  Device device = Device::CPU;
};

struct AdamStateView {
  void *base = nullptr;
  uint64_t bytes = 0;
  Device device = Device::CPU;
};

struct CheckpointConvergenceState {
  static constexpr uint32_t kHasBest = 1u << 0;
  static constexpr uint32_t kStopRequested = 1u << 1;

  float best_loss = 0.0f;
  float last_epoch_loss = -1.0f;
  uint32_t best_epoch = 0;
  uint32_t epochs_without_improvement = 0;
  uint32_t flags = 0;
  uint32_t stop_reason = 0;
};

bool save_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state, uint64_t global_step,
                     uint32_t epoch,
                     const CheckpointConvergenceState *convergence_state = nullptr);

bool load_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state,
                     uint64_t &restored_step, uint32_t &restored_epoch,
                     CheckpointConvergenceState *restored_convergence_state = nullptr,
                     std::string *error_detail = nullptr);
