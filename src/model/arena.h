#pragma once

#include "backend/device_backend.hpp"

#include <cstdint>

#include "tensor.hpp"

class Arena {
public:
  Arena(DeviceBackend &backend, Device dev, uint64_t bytes, uint32_t alignment);
  ~Arena();

  void *ptr();
  uint64_t size_bytes() const;
  Device device() const;

private:
  Device dev_;
  uint64_t bytes_;
  void *ptr_;
};
