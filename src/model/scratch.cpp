#include "scratch.hpp"

#include <cstring>
#include <limits>

static uint64_t safe_mul(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  if (a > std::numeric_limits<uint64_t>::max() / b) {
    throw std::overflow_error("Scratch: size overflow");
  }
  return a * b;
}

uint64_t Scratch::align_up(uint64_t x, uint64_t a) {
  uint64_t r = x % a;
  return (r == 0) ? x : (x + (a - r));
}

Scratch::Scratch(Arena &arena, Device dev)
    : arena_(arena), device_(dev), offset_(0) {}

void Scratch::reset() { offset_ = 0; }

TensorView Scratch::alloc_f32(Device dev, Shape shape) {
  if (dev != device_) {
    throw std::runtime_error("Scratch: device mismatch");
  }
  if (dev != Device::CPU) {
    throw std::runtime_error("Scratch: GPU path not implemented");
  }

  const uint64_t bytes = nbytes(shape, DType::F32);
  (void)safe_mul(1, bytes);

  constexpr uint64_t ALIGN = 64;
  uint64_t aligned_offset = align_up(offset_, ALIGN);

  if (aligned_offset + bytes > arena_.size_bytes()) {
    throw std::runtime_error("Scratch: out of memory");
  }

  uint8_t *base = reinterpret_cast<uint8_t *>(arena_.ptr());
  void *data_ptr = base + aligned_offset;

  offset_ = aligned_offset + bytes;

  std::memset(data_ptr, 0, static_cast<size_t>(bytes));

  return TensorView(Device::CPU, DType::F32, data_ptr, shape);
}
