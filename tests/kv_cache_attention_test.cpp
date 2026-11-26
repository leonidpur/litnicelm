// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

// Contract test of the KV-cached attention core (KvCachedAttentionCore over
// an UnfusedAttentionCore) on one backend plugin: the prompt runs through
// the inner core at position 0 (prefill), then every later token alone at
// its position, attending over the cache (decode). Every context row, prompt
// and decoded, must match FP64 causal attention over the whole sequence, and
// the UnfusedAttentionCore's row on the same inputs; in FP32 and, where the
// backend has _bf16act, BF16 activations. Inputs are BF16-representable in
// both. Positions beyond the cache throw, and so does backward.
//
// Usage: litnice_kv_cache_attention_test <backend plugin .so>

#include "attention_dims.hpp"
#include "backend/backend_plugin_api.hpp"
#include "backend/device_backend.hpp"
#include "kv_cache_attention.hpp"
#include "kv_cached_attention_core.hpp"
#include "ops.hpp"
#include "slice_request.hpp"
#include "tensor.hpp"
#include "unfused_attention_core.hpp"
#include <config.hpp>
#include <litnicelm_utils/bf16.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

std::string sci(double v) {
  std::ostringstream oss;
  oss << std::scientific << v;
  return oss.str();
}

class DeviceBuffer {
public:
  DeviceBuffer(DeviceBackend &backend, DType dtype, Shape shape)
      : backend_(backend), dtype_(dtype), shape_(shape),
        bytes_(shape.numel() * dtype_size(dtype)) {
    data_ = backend_.alloc(bytes_, 64);
    std::vector<uint8_t> zeros(static_cast<size_t>(bytes_), 0);
    backend_.copy_host2device(data_, zeros.data(), bytes_);
  }
  ~DeviceBuffer() { backend_.free(data_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  TensorView view() const {
    return TensorView(backend_.device(), dtype_, data_, shape_);
  }
  void upload(const std::vector<float> &values) {
    if (dtype_ == DType::BF16) {
      std::vector<uint16_t> bits(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        bits[i] = f32_to_bf16_bits(values[i]);
      }
      backend_.copy_host2device(data_, bits.data(), bytes_);
    } else {
      backend_.copy_host2device(data_, values.data(), bytes_);
    }
  }
  std::vector<float> download() const {
    std::vector<float> out(shape_.numel());
    if (dtype_ == DType::BF16) {
      std::vector<uint16_t> bits(shape_.numel());
      backend_.copy_device2host(bits.data(), data_, bytes_);
      for (size_t i = 0; i < bits.size(); ++i) {
        out[i] = bf16_bits_to_f32(bits[i]);
      }
    } else {
      backend_.copy_device2host(out.data(), data_, bytes_);
    }
    return out;
  }

private:
  DeviceBackend &backend_;
  DType dtype_;
  Shape shape_;
  uint64_t bytes_;
  void *data_ = nullptr;
};

struct Case {
  int64_t seq, model_dim, heads, prompt, cache;
  std::string text() const {
    return "S=" + std::to_string(seq) + " D=" + std::to_string(model_dim) +
           " H=" + std::to_string(heads) + " prompt=" + std::to_string(prompt) +
           " cache=" + std::to_string(cache);
  }
};

// qkv [1, S, 3D] of BF16-representable values in [-1, 1].
std::vector<float> random_qkv(const Case &c, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(static_cast<size_t>(c.seq * 3 * c.model_dim));
  for (float &x : v) {
    x = round_to_bf16(dist(rng));
  }
  return v;
}

// FP64 causal attention: qkv [1, S, 3D] -> context [1, S, D].
std::vector<double> attention_f64(const std::vector<float> &qkv, const Case &c) {
  const int64_t D = c.model_dim;
  const int64_t dh = D / c.heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dh));
  std::vector<double> context(static_cast<size_t>(c.seq * D), 0.0);
  auto at = [&](int64_t t, int64_t col) {
    return static_cast<double>(qkv[static_cast<size_t>(t * 3 * D + col)]);
  };
  for (int64_t h = 0; h < c.heads; ++h) {
    for (int64_t i = 0; i < c.seq; ++i) {
      std::vector<double> p(static_cast<size_t>(i + 1));
      double max_score = -1e300;
      for (int64_t j = 0; j <= i; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < dh; ++d) {
          dot += at(i, h * dh + d) * at(j, D + h * dh + d);
        }
        p[static_cast<size_t>(j)] = dot * scale;
        max_score = std::max(max_score, p[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (double &x : p) {
        x = std::exp(x - max_score);
        sum += x;
      }
      for (int64_t d = 0; d < dh; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j <= i; ++j) {
          acc += p[static_cast<size_t>(j)] / sum * at(j, 2 * D + h * dh + d);
        }
        context[static_cast<size_t>(i * D + h * dh + d)] = acc;
      }
    }
  }
  return context;
}

Config attention_config(const Case &c) {
  Config cfg{};
  cfg.model.d_model = static_cast<uint32_t>(c.model_dim);
  cfg.model.n_heads = static_cast<uint32_t>(c.heads);
  cfg.model.max_seq_len = static_cast<uint32_t>(c.cache);
  return cfg;
}

NodeSlices allocate(DeviceBackend &backend, const std::vector<SliceRequest> &requests,
                    DType act, std::vector<std::unique_ptr<DeviceBuffer>> &storage) {
  NodeSlices slices;
  for (const SliceRequest &request : requests) {
    const DType dtype = request.item == ItemType::Statistics ? DType::F32 : act;
    storage.push_back(std::make_unique<DeviceBuffer>(backend, dtype, request.shape));
    slices.add(request.name, storage.back()->view());
  }
  return slices;
}

// Context [1, S, D] from the cached core: the prompt rows in one call at
// position 0 (prefill through the inner core), then each later row alone.
std::vector<float> cached_context(DeviceBackend &backend, const Ops &ops,
                                  const Case &c, DType act,
                                  const std::vector<float> &qkv) {
  const Config cfg = attention_config(c);
  const AttentionDims dims{1, c.cache, c.model_dim, c.heads, false, true};
  std::vector<std::unique_ptr<DeviceBuffer>> storage;
  KvCachedAttentionCore core(
      std::make_unique<UnfusedAttentionCore>(
          0, cfg, ops,
          allocate(backend, UnfusedAttentionCore::slice_requests(dims), act, storage)),
      std::make_unique<KvCacheAttention>(
          0, cfg, ops,
          allocate(backend, KvCacheAttention::slice_requests(dims), act, storage)));
  DeviceBuffer qkv_buf(backend, act, Shape{1, c.seq, 3 * c.model_dim});
  DeviceBuffer context_buf(backend, act, Shape{1, c.seq, c.model_dim});
  qkv_buf.upload(qkv);
  TensorView prompt_context = context_buf.view().slice(1, 0, c.prompt);
  core.forward(qkv_buf.view().slice(1, 0, c.prompt), prompt_context, 0);
  for (int64_t t = c.prompt; t < c.seq; ++t) {
    TensorView row = context_buf.view().slice(1, t, 1);
    core.forward(qkv_buf.view().slice(1, t, 1), row, t);
  }
  return context_buf.download();
}

// The unfused core's context [1, S, D] on the whole sequence.
std::vector<float> unfused_context(DeviceBackend &backend, const Ops &ops,
                                   const Case &c, DType act,
                                   const std::vector<float> &qkv) {
  const Config cfg = attention_config(c);
  const AttentionDims dims{1, c.seq, c.model_dim, c.heads, false};
  std::vector<std::unique_ptr<DeviceBuffer>> storage;
  UnfusedAttentionCore core(0, cfg, ops,
                            allocate(backend, UnfusedAttentionCore::slice_requests(dims),
                                     act, storage));
  DeviceBuffer qkv_buf(backend, act, Shape{1, c.seq, 3 * c.model_dim});
  DeviceBuffer context_buf(backend, act, Shape{1, c.seq, c.model_dim});
  qkv_buf.upload(qkv);
  TensorView context = context_buf.view();
  core.forward(qkv_buf.view(), context, 0);
  return context_buf.download();
}

void test_decode(DeviceBackend &backend, const Ops &ops, const Case &c,
                 DType act, uint32_t seed) {
  const std::string label = std::string(act == DType::BF16 ? "bf16 " : "fp32 ") + c.text();
  const std::vector<float> qkv = random_qkv(c, seed);
  const std::vector<double> truth = attention_f64(qkv, c);
  const std::vector<float> cached = cached_context(backend, ops, c, act, qkv);
  const std::vector<float> unfused = unfused_context(backend, ops, c, act, qkv);
  double vs_truth = 0.0;
  double vs_unfused = 0.0;
  for (size_t i = 0; i < cached.size(); ++i) {
    vs_truth = std::max(vs_truth, std::fabs(cached[i] - truth[i]));
    vs_unfused = std::max(vs_unfused,
                          std::fabs(static_cast<double>(cached[i]) - unfused[i]));
  }
  // FP32: rounding of D-term dot products and S-term sums. BF16: scores,
  // weights and context are each rounded to BF16 (8-bit mantissa) on values
  // of magnitude <= 1, a few units of 2^-8.
  const double tol = act == DType::BF16 ? 2e-2 : 1e-5;
  check(vs_truth <= tol, label + ": prefill + decoded rows vs FP64 causal attention, max abs " +
                             sci(vs_truth) + " <= " + sci(tol));
  check(vs_unfused <= tol, label + ": prefill + decoded rows vs unfused core, max abs " +
                               sci(vs_unfused) + " <= " + sci(tol));
}

void test_bounds(DeviceBackend &backend, const Ops &ops) {
  const Case c{4, 8, 2, 4, 4};
  const Config cfg = attention_config(c);
  const AttentionDims dims{1, c.cache, c.model_dim, c.heads, false, true};
  std::vector<std::unique_ptr<DeviceBuffer>> storage;
  KvCacheAttention cache(0, cfg, ops,
                         allocate(backend, KvCacheAttention::slice_requests(dims),
                                  DType::F32, storage));
  DeviceBuffer qkv_buf(backend, DType::F32, Shape{1, 2, 3 * c.model_dim});
  bool threw = false;
  try {
    cache.append(qkv_buf.view(), 3);  // positions 3, 4 of a 4-position cache
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "append beyond the cache throws");

  std::vector<std::unique_ptr<DeviceBuffer>> core_storage;
  KvCachedAttentionCore core(
      std::make_unique<UnfusedAttentionCore>(
          0, cfg, ops,
          allocate(backend, UnfusedAttentionCore::slice_requests(dims), DType::F32,
                   core_storage)),
      std::make_unique<KvCacheAttention>(
          0, cfg, ops,
          allocate(backend, KvCacheAttention::slice_requests(dims), DType::F32,
                   core_storage)));
  DeviceBuffer d(backend, DType::F32, Shape{1, 1, c.model_dim});
  DeviceBuffer dq(backend, DType::F32, Shape{1, 1, 3 * c.model_dim});
  bool backward_threw = false;
  try {
    TensorView dqkv = dq.view();
    core.backward(d.view(), dqkv);
  } catch (const std::exception &) {
    backward_threw = true;
  }
  check(backward_threw, "KvCachedAttentionCore backward throws (inference only)");
  check(KvCacheAttention::slice_requests(AttentionDims{1, 4, 8, 2, false, false}).empty(),
        "no cache slices without with_kv_cache");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <backend plugin .so>\n";
    return 2;
  }
  try {
    Config cfg{};
    cfg.backend.library = argv[1];
    std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
    Ops ops(*backend);
    std::cout << "KV-cache attention contract tests on " << argv[1] << "\n";
    const bool has_bf16 =
        (backend->precision_variants() & kPrecisionVariantBf16Act) != 0;
    const std::vector<Case> cases{
        {13, 12, 3, 5, 16}, {40, 32, 4, 1, 40}, {33, 64, 8, 32, 64}};
    for (size_t i = 0; i < cases.size(); ++i) {
      const uint32_t seed = 500 + static_cast<uint32_t>(i);
      test_decode(*backend, ops, cases[i], DType::F32, seed);
      if (has_bf16) {
        test_decode(*backend, ops, cases[i], DType::BF16, seed);
      }
    }
    test_bounds(*backend, ops);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (g_failures != 0) {
    std::cout << g_failures << " KV-cache attention test(s) FAILED\n";
    return 1;
  }
  std::cout << "All KV-cache attention tests passed.\n";
  return 0;
}
