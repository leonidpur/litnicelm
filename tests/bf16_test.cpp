// Host F32 <-> BF16 conversion: round to nearest even, NaN preserved (also
// when its payload lies only in the low 16 bits), Inf and overflow, zeros,
// subnormals, and exact round trips.

#include <litnicelm_utils/bf16.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

float from_bits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::string hex(uint32_t v) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(4) << std::setfill('0') << v;
  return oss.str();
}

void expect_bits(const std::string &name, float in, uint16_t want) {
  const uint16_t got = f32_to_bf16_bits(in);
  const bool ok = got == want;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
  if (!ok) {
    std::cout << ": got " << hex(got) << ", want " << hex(want);
    ++g_failures;
  }
  std::cout << "\n";
}

void expect(const std::string &name, bool ok) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
  if (!ok) {
    ++g_failures;
  }
}

} // namespace

int main() {
  // Exact values keep their upper half.
  expect_bits("1.0", 1.0f, 0x3F80);
  expect_bits("-2.5", -2.5f, 0xC020);
  expect_bits("+0", 0.0f, 0x0000);
  expect_bits("-0", -0.0f, 0x8000);

  // Round to nearest, ties to even. 1.0 + 2^-8 lies halfway between 1.0
  // (even) and 1.0 + 2^-7 (odd): ties go to 1.0.
  expect_bits("tie to even (down)", from_bits(0x3F808000), 0x3F80);
  // 1.0 + 3*2^-8 is halfway between odd 0x3F81 and even 0x3F82: up.
  expect_bits("tie to even (up)", from_bits(0x3F818000), 0x3F82);
  expect_bits("just above half rounds up", from_bits(0x3F808001), 0x3F81);
  expect_bits("just below half rounds down", from_bits(0x3F807FFF), 0x3F80);

  // Inf and overflow.
  expect_bits("+Inf", std::numeric_limits<float>::infinity(), 0x7F80);
  expect_bits("-Inf", -std::numeric_limits<float>::infinity(), 0xFF80);
  expect_bits("max finite F32 overflows to +Inf",
              std::numeric_limits<float>::max(), 0x7F80);
  expect_bits("largest BF16 stays finite", from_bits(0x7F7F0000), 0x7F7F);

  // NaN stays NaN with its sign, including payloads only in the low bits,
  // which plain rounding would turn into Inf.
  expect_bits("quiet NaN", from_bits(0x7FC00000), 0x7FC0);
  expect_bits("-NaN", from_bits(0xFFC00000), 0xFFC0);
  expect_bits("signaling NaN, low payload only", from_bits(0x7F800001), 0x7FC0);
  expect_bits("NaN with all low bits set", from_bits(0x7F80FFFF), 0x7FC0);
  expect("round_to_bf16(NaN) is NaN",
         std::isnan(round_to_bf16(std::numeric_limits<float>::quiet_NaN())));

  // Subnormals round like normals.
  expect_bits("smallest F32 subnormal rounds to +0", from_bits(0x00000001), 0x0000);
  expect_bits("BF16 subnormal kept", from_bits(0x00010000), 0x0001);

  // Every BF16 value survives BF16 -> F32 -> BF16; NaNs stay NaN.
  bool round_trips = true;
  for (uint32_t b = 0; b <= 0xFFFF; ++b) {
    const float f = bf16_bits_to_f32(static_cast<uint16_t>(b));
    const uint16_t back = f32_to_bf16_bits(f);
    round_trips = round_trips && (std::isnan(f) ? std::isnan(bf16_bits_to_f32(back))
                                                : back == b);
  }
  expect("all 65536 BF16 values round-trip", round_trips);

  std::cout << (g_failures == 0 ? "All BF16 conversion tests passed.\n"
                                : "BF16 conversion tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
