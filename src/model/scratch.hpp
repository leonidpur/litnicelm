#pragma once

#include "tensor.hpp"
#include "arena.hpp"

#include <cstdint>
#include <stdexcept>

class Scratch {
public:
  Scratch(Arena &arena, Device dev);

  void reset();

  TensorView alloc_f32(Device dev, Shape shape);

  uint64_t used_bytes() const { return offset_; }
  uint64_t capacity_bytes() const { return arena_.size_bytes(); }

private:
  Arena &arena_;
  Device device_;
  uint64_t offset_ = 0;

  static uint64_t align_up(uint64_t x, uint64_t a);
};
