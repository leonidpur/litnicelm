// Exhaustive cast check on a backend plugin: every one of the 2^32 F32 bit
// patterns through cast F32 -> BF16, and every one of the 2^16 BF16 patterns
// through cast BF16 -> F32, compared bit for bit with the host conversion
// (f32_to_bf16_bits / bf16_bits_to_f32): NaN sign and payload, Inf,
// subnormals and every rounding tie included. Slow (seconds to minutes).
//
// Usage: litnice_cast_exhaustive_test <path to backend plugin .so>

#include "backend/device_backend.hpp"
#include "tensor.hpp"

#include <config.hpp>
#include <litnicelm_utils/bf16.hpp>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kChunk = uint64_t{1} << 24;  // F32 patterns per launch
constexpr int64_t kCols = 4096;
constexpr int64_t kRows = static_cast<int64_t>(kChunk) / kCols;

std::string hex(uint64_t v, int width) {
  std::ostringstream oss;
  oss << "0x" << std::hex << std::setw(width) << std::setfill('0') << v;
  return oss.str();
}

// Device buffer of one chunk; freed on scope exit.
struct DeviceBuffer {
  DeviceBackend &backend;
  void *data;
  DeviceBuffer(DeviceBackend &b, uint64_t bytes)
      : backend(b), data(b.alloc(bytes, 256)) {}
  ~DeviceBuffer() { backend.free(data); }
};

bool check_f32_to_bf16(DeviceBackend &backend) {
  DeviceBuffer src(backend, kChunk * sizeof(float));
  DeviceBuffer dst(backend, kChunk * sizeof(uint16_t));
  TensorView src_view(backend.device(), DType::F32, src.data, Shape{kRows, kCols});
  TensorView dst_view(backend.device(), DType::BF16, dst.data, Shape{kRows, kCols});
  std::vector<uint32_t> in(kChunk);
  std::vector<uint16_t> out(kChunk);
  uint64_t mismatches = 0;
  for (uint64_t base = 0; base < (uint64_t{1} << 32); base += kChunk) {
    for (uint64_t i = 0; i < kChunk; ++i) {
      in[i] = static_cast<uint32_t>(base + i);
    }
    backend.copy_host2device(src.data, in.data(), kChunk * sizeof(float));
    backend.cast(src_view, dst_view);
    backend.copy_device2host(out.data(), dst.data, kChunk * sizeof(uint16_t));
    for (uint64_t i = 0; i < kChunk; ++i) {
      float value;
      std::memcpy(&value, &in[i], sizeof(value));
      const uint16_t want = f32_to_bf16_bits(value);
      if (out[i] != want) {
        if (mismatches < 10) {
          std::cout << "  F32 " << hex(in[i], 8) << ": device " << hex(out[i], 4)
                    << ", host " << hex(want, 4) << "\n";
        }
        ++mismatches;
      }
    }
  }
  std::cout << (mismatches == 0 ? "[PASS] " : "[FAIL] ")
            << "all 2^32 F32 patterns, cast F32 -> BF16 matches host bit for bit"
            << " (" << mismatches << " mismatches)\n";
  return mismatches == 0;
}

bool check_bf16_to_f32(DeviceBackend &backend) {
  constexpr int64_t kCount = 1 << 16;
  DeviceBuffer src(backend, kCount * sizeof(uint16_t));
  DeviceBuffer dst(backend, kCount * sizeof(float));
  TensorView src_view(backend.device(), DType::BF16, src.data, Shape{256, 256});
  TensorView dst_view(backend.device(), DType::F32, dst.data, Shape{256, 256});
  std::vector<uint16_t> in(kCount);
  std::vector<uint32_t> out(kCount);
  for (int64_t i = 0; i < kCount; ++i) {
    in[i] = static_cast<uint16_t>(i);
  }
  backend.copy_host2device(src.data, in.data(), kCount * sizeof(uint16_t));
  backend.cast(src_view, dst_view);
  backend.copy_device2host(out.data(), dst.data, kCount * sizeof(float));
  uint64_t mismatches = 0;
  for (int64_t i = 0; i < kCount; ++i) {
    const float want_value = bf16_bits_to_f32(in[i]);
    uint32_t want;
    std::memcpy(&want, &want_value, sizeof(want));
    if (out[i] != want) {
      if (mismatches < 10) {
        std::cout << "  BF16 " << hex(in[i], 4) << ": device " << hex(out[i], 8)
                  << ", host " << hex(want, 8) << "\n";
      }
      ++mismatches;
    }
  }
  std::cout << (mismatches == 0 ? "[PASS] " : "[FAIL] ")
            << "all 2^16 BF16 patterns, cast BF16 -> F32 matches host bit for bit"
            << " (" << mismatches << " mismatches)\n";
  return mismatches == 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <backend plugin .so>\n";
    return 2;
  }
  Config cfg{};
  cfg.backend.library = argv[1];
  std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
  std::cout << "Exhaustive cast test on " << argv[1] << "\n";
  const bool ok = check_bf16_to_f32(*backend) & check_f32_to_bf16(*backend);
  std::cout << (ok ? "All exhaustive cast tests passed.\n"
                   : "Exhaustive cast tests FAILED.\n");
  return ok ? 0 : 1;
}
