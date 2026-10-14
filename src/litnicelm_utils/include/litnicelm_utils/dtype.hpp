#pragma once

#include <cstddef>
#include <cstdint>

// Element type of a tensor.
enum class DType : uint8_t {
  F32 = 0,
  I32 = 1,
};

inline constexpr size_t dtype_size(DType dt) {
  switch (dt) {
  case DType::F32:
    return 4;
  case DType::I32:
    return 4;
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
  default:
    return "unknown";
  }
}
