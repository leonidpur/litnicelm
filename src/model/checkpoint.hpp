#pragma once

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
                     uint64_t alignment_bytes,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state, uint64_t global_step);

bool load_checkpoint(const std::string &path, const ModelConfig &model,
                     const std::string &conf_version,
                     uint64_t alignment_bytes,
                     const ArenaView &data_arena,
                     const AdamStateView &adam_state,
                     uint64_t &restored_step,
                     std::string *error_detail = nullptr);
