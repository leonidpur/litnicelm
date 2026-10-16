#pragma once

#include <cstddef>
#include <cstdint>

// Element type of a tensor.
enum class DType : uint8_t {
  F32 = 0,
  I32 = 1,
  BF16 = 2,  // bfloat16: the upper half of an F32 (see bf16.hpp)
};

inline constexpr size_t dtype_size(DType dt) {
  switch (dt) {
  case DType::F32:
    return 4;
  case DType::I32:
    return 4;
  case DType::BF16:
    return 2;
  default:
    return 0;
  }
}

inline constexpr const char *dtype_name(DType dt) {
  switch (dt) {
  case DType::F32:
    return "f32";
  case DType::I32:
    return "i32";
  case DType::BF16:
    return "bf16";
  default:
    return "unknown";
  }
}
