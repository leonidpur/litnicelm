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

bool save_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state, uint64_t global_step,
                     uint32_t epoch);

bool load_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes, DeviceBackend &backend,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state,
                     uint64_t &restored_step, uint32_t &restored_epoch,
                     std::string *error_detail = nullptr);
