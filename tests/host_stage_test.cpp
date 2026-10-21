// host_tensor_stage::stage_to_cpu_f32: every one of the 2^16 BF16 patterns
// in a backend tensor arrives on the host as its exact F32 value (bit for
// bit, NaN payloads included); F32 tensors arrive unchanged.
//
// Usage: litnice_host_stage_test <backend plugin .so>

#include "backend/device_backend.hpp"
#include "host_tensor_stage.hpp"
#include "tensor.hpp"

#include <config.hpp>
#include <litnicelm_utils/bf16.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <backend plugin .so>\n";
    return 2;
  }
  Config cfg{};
  cfg.backend.library = argv[1];
  std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
  int failures = 0;

  constexpr int64_t kCount = 1 << 16;
  std::vector<uint16_t> bits(kCount);
  for (int64_t i = 0; i < kCount; ++i) {
    bits[static_cast<size_t>(i)] = static_cast<uint16_t>(i);
  }
  void *bf16 = backend->alloc(kCount * sizeof(uint16_t), 256);
  backend->copy_host2device(bf16, bits.data(), kCount * sizeof(uint16_t));
  const TensorView bf16_view(backend->device(), DType::BF16, bf16, Shape{256, 256});
  const Tensor staged = host_tensor_stage::stage_to_cpu_f32(*backend, bf16_view, "test");
  const auto *got = static_cast<const uint32_t *>(staged.view().data());
  uint64_t mismatches = 0;
  for (int64_t i = 0; i < kCount; ++i) {
    const float want_value = bf16_bits_to_f32(bits[static_cast<size_t>(i)]);
    uint32_t want;
    std::memcpy(&want, &want_value, sizeof(want));
    mismatches += got[i] != want;
  }
  const bool bf16_ok = staged.view().dtype() == DType::F32 && mismatches == 0;
  std::cout << (bf16_ok ? "[PASS] " : "[FAIL] ")
            << "all 2^16 BF16 patterns staged as their exact F32 values ("
            << mismatches << " mismatches)\n";
  failures += !bf16_ok;
  backend->free(bf16);

  std::vector<float> values = {1.5f, -0.0f, 3.0e-39f, 65504.0f};
  void *f32 = backend->alloc(values.size() * sizeof(float), 256);
  backend->copy_host2device(f32, values.data(), values.size() * sizeof(float));
  const TensorView f32_view(backend->device(), DType::F32, f32,
                            Shape{1, static_cast<int64_t>(values.size())});
  const Tensor f32_staged = host_tensor_stage::stage_to_cpu_f32(*backend, f32_view, "test");
  const bool f32_ok =
      std::memcmp(f32_staged.view().data(), values.data(), values.size() * sizeof(float)) == 0;
  std::cout << (f32_ok ? "[PASS] " : "[FAIL] ") << "F32 tensors staged unchanged\n";
  failures += !f32_ok;
  backend->free(f32);

  std::cout << (failures == 0 ? "All host stage tests passed.\n"
                              : "Host stage tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
