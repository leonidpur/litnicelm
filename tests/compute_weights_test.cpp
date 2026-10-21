// Compute weights: under fp32 every compute_* view of TensorStore is the
// parameter itself (no allocation); the BF16 compute-weight layout holds
// exactly the GEMM weights (QKV, output, W1, W2 per layer, LM head) at two
// bytes per element.
//
// Usage: litnice_compute_weights_test <config.yaml> <backend plugin .so>

#include "backend/device_backend.hpp"
#include "memory/inference_memory_manager.hpp"
#include "memory/named_layout.hpp"
#include "memory/precision_policy.hpp"
#include "model_algo_config.hpp"
#include "model_algo_factory.hpp"

#include <config.hpp>

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

  std::cout << (failures == 0 ? "All compute weight tests passed.\n"
                              : "Compute weight tests FAILED.\n");
  return failures == 0 ? 0 : 1;
}
