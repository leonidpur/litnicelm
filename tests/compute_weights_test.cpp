// Compute weights: under fp32 every compute_* view of TensorStore is the
// parameter itself (no allocation); the BF16 compute-weight layout holds
// exactly the GEMM weights (QKV, output, W1, W2 per layer, LM head) at two
// bytes per element; the weight sync writes each compute weight as the
// BF16 rounding of its parameter, bit for bit.
//
// Usage: litnice_compute_weights_test <config.yaml> <backend plugin .so>

#include "backend/device_backend.hpp"
#include "memory/inference_memory_manager.hpp"
#include "memory/compute_weight_sync.hpp"
#include "memory/named_layout.hpp"
#include "memory/precision_policy.hpp"
#include "memory/tensor_store.hpp"
#include "model_algo_config.hpp"
#include "model_algo_factory.hpp"

#include <config.hpp>
#include <litnicelm_utils/bf16.hpp>

#include <cstring>
#include <random>
#include <vector>

#include <iostream>
#include <memory>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++failures;
  }
}

bool same(const TensorView &a, const TensorView &b) {
  if (a.data() != b.data() || a.dtype() != b.dtype() || a.rank() != b.rank()) {
    return false;
  }
  for (size_t i = 0; i < a.rank(); ++i) {
    if (a.dim(i) != b.dim(i)) {
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <config.yaml> <backend plugin .so>\n";
    return 2;
  }
  Config cfg = Config::load_from_file(argv[1]);
  cfg.backend.library = argv[2];
  const std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
  const ModelAlgoFactory algo(ModelAlgoConfig::from_config(cfg));
  const PrecisionPolicy precision =
      PrecisionPolicy::create(cfg, backend->precision_variants());
  InferenceMemoryManager memory(cfg, algo, *backend, precision);
  const TensorStore &store = memory.tensor_store();

  bool aliased = true;
  for (int l = 0; l < static_cast<int>(cfg.model.n_layers); ++l) {
    aliased = aliased && same(store.compute_attn_qkv_w(l), store.param_attn_qkv_w(l)) &&
              same(store.compute_attn_out_w(l), store.param_attn_out_w(l)) &&
              same(store.compute_ffn_w1(l), store.param_ffn_w1(l)) &&
              same(store.compute_ffn_w2(l), store.param_ffn_w2(l));
  }
  aliased = aliased && same(store.compute_lm_head_w(), store.param_lm_head_w());
  expect(aliased, "fp32: every compute weight is its parameter");

  const NamedLayout layout = NamedLayout::build_compute_weight_layout(cfg, DType::BF16);
  const uint64_t D = cfg.model.d_model;
  const uint64_t F = cfg.model.d_ff;
  const uint64_t V = cfg.model.vocab_size;
  const uint64_t L = cfg.model.n_layers;
  uint64_t data_bytes = 0;
  bool all_bf16 = true;
  for (const LayoutSlice &s : layout.slices()) {
    data_bytes += s.bytes;
    all_bf16 = all_bf16 && s.dtype == DType::BF16;
  }
  expect(layout.slices().size() == 4 * L + 1,
         "BF16 compute layout: 4 GEMM weights per layer + LM head (" +
             std::to_string(layout.slices().size()) + " slices)");
  expect(all_bf16, "BF16 compute layout: every slice is BF16");
  expect(data_bytes == 2 * (L * (4 * D * D + 2 * D * F) + D * V),
         "BF16 compute layout: 2 bytes per GEMM weight element (" +
             std::to_string(data_bytes) + " bytes)");

  // A store with a BF16 compute arena next to the F32 parameters, built by
  // hand (bf16_mixed setup is gated until E3 ends).
  const NamedLayout param_layout =
      NamedLayout::build_param_layout(cfg, algo, DType::F32);
  const NamedLayout temp_layout =
      NamedLayout::build_inference_temp_layout(cfg, algo, precision);
  const uint64_t compute_bytes = layout.total_bytes();
  void *params = backend->alloc(param_layout.total_bytes(), 256);
  void *temps = backend->alloc(temp_layout.total_bytes(), 256);
  void *compute = backend->alloc(compute_bytes, 256);
  std::vector<float> host_params(param_layout.total_bytes() / sizeof(float));
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (float &x : host_params) {
    x = dist(rng);
  }
  backend->copy_host2device(params, host_params.data(), param_layout.total_bytes());
  {
    TensorStore bf16_store(cfg, algo, param_layout, params, param_layout.total_bytes(),
                           backend->device(), temp_layout, temps,
                           temp_layout.total_bytes(),
                           TensorStore::TempLayoutKind::Inference, &layout, compute,
                           compute_bytes);
    cast_parameters_to_compute_weights(cfg, bf16_store, *backend);
    std::vector<uint16_t> host_compute(compute_bytes / sizeof(uint16_t));
    backend->copy_device2host(host_compute.data(), compute, compute_bytes);
    auto rounded = [&](const TensorView &param, const TensorView &weight) {
      // Byte offsets of the two views in their arenas.
      const uint64_t p_off = static_cast<uint64_t>(
          static_cast<const uint8_t *>(param.data()) - static_cast<const uint8_t *>(params));
      const uint64_t w_off = static_cast<uint64_t>(
          static_cast<const uint8_t *>(weight.data()) - static_cast<const uint8_t *>(compute));
      for (uint64_t i = 0; i < param.numel(); ++i) {
        if (host_compute[w_off / 2 + i] !=
            f32_to_bf16_bits(host_params[p_off / 4 + i])) {
          return false;
        }
      }
      return weight.dtype() == DType::BF16;
    };
    bool synced = true;
    for (int l = 0; l < static_cast<int>(cfg.model.n_layers); ++l) {
      synced = synced &&
               rounded(bf16_store.param_attn_qkv_w(l), bf16_store.compute_attn_qkv_w(l)) &&
               rounded(bf16_store.param_attn_out_w(l), bf16_store.compute_attn_out_w(l)) &&
               rounded(bf16_store.param_ffn_w1(l), bf16_store.compute_ffn_w1(l)) &&
               rounded(bf16_store.param_ffn_w2(l), bf16_store.compute_ffn_w2(l));
    }
    synced = synced &&
             rounded(bf16_store.param_lm_head_w(), bf16_store.compute_lm_head_w());
    expect(synced, "sync: every BF16 compute weight is its parameter rounded to BF16");
  }
  backend->free(compute);
  backend->free(temps);
  backend->free(params);

  std::cout << (failures == 0 ? "All compute weight tests passed.\n"
                              : "Compute weight tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
