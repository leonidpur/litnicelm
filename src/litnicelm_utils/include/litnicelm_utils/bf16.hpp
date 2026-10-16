#pragma once

// Host-side bfloat16: the upper 16 bits of an IEEE-754 binary32, so the
// same sign and exponent range as F32 with an 8-bit significand.

#include <cstdint>
#include <cstring>

// F32 -> BF16 bits, rounding to nearest, ties to even. NaN is tested before
// rounding (a NaN whose payload lies only in the low 16 bits would otherwise
// round to Inf) and comes out as a quiet NaN of the same sign. Inf stays Inf;
// finite values beyond the BF16 range round to Inf, as in IEEE.
inline uint16_t f32_to_bf16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// BF16 bits -> F32; exact.
inline float bf16_bits_to_f32(uint16_t bf16) {
  const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// value rounded to the nearest BF16, as an F32.
inline float round_to_bf16(float value) {
  return bf16_bits_to_f32(f32_to_bf16_bits(value));
}
