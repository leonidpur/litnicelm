// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

// Contract tests of the attention nodes on one backend plugin:
//   - UnfusedAttentionCore against PyTorch goldens (float64 truth);
//   - on the CPU backend, UnfusedAttentionCore's gradient against central
//     finite differences of an FP64 host implementation;
//   - every other core against UnfusedAttentionCore (differential) on
//     seeded inputs with odd shapes, outputs and all gradients, in FP32 and,
//     where the backend has the _bf16act variant, BF16 activations;
//   - the GEMM projections against FP64 host math (output, input gradient,
//     weight and bias gradients);
//   - where the backend has exec contexts, the multistream attention against
//     ComposedSelfAttention with the unfused core, parameters and all;
//   - GuardedSelfAttention: a clean run passes, a write past a slice and a
//     change of saved state between forward and backward are caught.
//
// Usage: litnice_attention_core_test <backend plugin .so> <attention_core.json>

#include "attention_dims.hpp"
#include "backend/backend_plugin_api.hpp"
#include "backend/device_backend.hpp"
#include "composed_self_attention.hpp"
#include "cudnn_flash_attention_core.hpp"
#include "fused_inplace_attention_core.hpp"
#include "fused_linear_output_projection.hpp"
#include "fused_linear_qkv_projection.hpp"
#include "gemm_output_projection.hpp"
#include "gemm_qkv_projection.hpp"
#include "guarded_self_attention.hpp"
#include "linear_params.hpp"
#include "ops.hpp"
#include "unfused_attention_core.hpp"
#include "self_attention_fused_inplace_multistream.hpp"
#include "slice_request.hpp"
#include "tensor.hpp"
#include <config.hpp>
#include <litnicelm_utils/bf16.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;


struct Shape4 {
  int64_t batch, seq, model_dim, heads;
  std::string text() const {
    return "B=" + std::to_string(batch) + " S=" + std::to_string(seq) +
           " D=" + std::to_string(model_dim) + " H=" + std::to_string(heads);
  }
};

// One backend buffer, uploaded from and downloaded to floats.
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

enum class CoreKind { Unfused, FusedInplace, CudnnFlash };

const char *core_name(CoreKind kind) {
  switch (kind) {
  case CoreKind::Unfused:
    return "unfused";
  case CoreKind::FusedInplace:
    return "fused_inplace";
  case CoreKind::CudnnFlash:
    return "cudnn_flash";
  }
  return "?";
}

struct CoreResult {
  std::vector<float> context;
  std::vector<float> dqkv;
};

// Builds one core with its own slices, runs forward and backward once.
CoreResult run_core(DeviceBackend &backend, const Ops &ops, CoreKind kind,
                    const Shape4 &s, DType act, const std::vector<float> &qkv,
                    const std::vector<float> &dcontext) {
  Config cfg{};
  cfg.model.d_model = static_cast<uint32_t>(s.model_dim);
  cfg.model.n_heads = static_cast<uint32_t>(s.heads);
  const AttentionDims dims{s.batch, s.seq, s.model_dim, s.heads, true};
  const std::vector<SliceRequest> requests =
      kind == CoreKind::Unfused      ? UnfusedAttentionCore::slice_requests(dims)
      : kind == CoreKind::FusedInplace ? FusedInplaceAttentionCore::slice_requests(dims)
                                       : CudnnFlashAttentionCore::slice_requests(dims);
  std::vector<std::unique_ptr<DeviceBuffer>> storage;
  NodeSlices slices;
  for (const SliceRequest &request : requests) {
    const DType dtype = request.item == ItemType::Statistics ? DType::F32 : act;
    storage.push_back(std::make_unique<DeviceBuffer>(backend, dtype, request.shape));
    slices.add(request.name, storage.back()->view());
  }
  std::unique_ptr<IAttentionCore> core;
  if (kind == CoreKind::Unfused) {
    core = std::make_unique<UnfusedAttentionCore>(0, cfg, ops, slices);
  } else if (kind == CoreKind::FusedInplace) {
    core = std::make_unique<FusedInplaceAttentionCore>(0, cfg, ops, slices);
  } else {
    core = std::make_unique<CudnnFlashAttentionCore>(0, cfg, ops, slices);
  }

  DeviceBuffer qkv_buf(backend, act, Shape{s.batch, s.seq, 3 * s.model_dim});
  DeviceBuffer context_buf(backend, act, Shape{s.batch, s.seq, s.model_dim});
  DeviceBuffer dcontext_buf(backend, act, Shape{s.batch, s.seq, s.model_dim});
  DeviceBuffer dqkv_buf(backend, act, Shape{s.batch, s.seq, 3 * s.model_dim});
  qkv_buf.upload(qkv);
  dcontext_buf.upload(dcontext);
  TensorView context = context_buf.view();
  TensorView dqkv = dqkv_buf.view();
  core->forward(qkv_buf.view(), context);
  core->backward(dcontext_buf.view(), dqkv);
  return {context_buf.download(), dqkv_buf.download()};
}

// FP64 causal attention: qkv [B, S, 3D] (Q | K | V, heads side by side) ->
// context [B, S, D].
std::vector<double> attention_f64(const std::vector<double> &qkv, const Shape4 &s) {
  const int64_t D = s.model_dim;
  const int64_t dh = D / s.heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dh));
  std::vector<double> context(static_cast<size_t>(s.batch * s.seq * D), 0.0);
  auto at = [&](int64_t b, int64_t t, int64_t col) {
    return qkv[static_cast<size_t>((b * s.seq + t) * 3 * D + col)];
  };
  for (int64_t b = 0; b < s.batch; ++b) {
    for (int64_t h = 0; h < s.heads; ++h) {
      for (int64_t i = 0; i < s.seq; ++i) {
        std::vector<double> p(static_cast<size_t>(i + 1));
        double max_score = -1e300;
        for (int64_t j = 0; j <= i; ++j) {
          double dot = 0.0;
          for (int64_t c = 0; c < dh; ++c) {
            dot += at(b, i, h * dh + c) * at(b, j, D + h * dh + c);
          }
          p[static_cast<size_t>(j)] = dot * scale;
          max_score = std::max(max_score, p[static_cast<size_t>(j)]);
        }
        double sum = 0.0;
        for (double &v : p) {
          v = std::exp(v - max_score);
          sum += v;
        }
        for (int64_t c = 0; c < dh; ++c) {
          double acc = 0.0;
          for (int64_t j = 0; j <= i; ++j) {
            acc += p[static_cast<size_t>(j)] / sum * at(b, j, 2 * D + h * dh + c);
          }
          context[static_cast<size_t>((b * s.seq + i) * D + h * dh + c)] = acc;
        }
      }
    }
  }
  return context;
}

double dot(const std::vector<double> &a, const std::vector<float> &b) {
  double out = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    out += a[i] * static_cast<double>(b[i]);
  }
  return out;
}

std::vector<float> random_values(size_t count, uint32_t seed, bool bf16) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(count);
  for (float &v : out) {
    v = bf16 ? round_to_bf16(dist(rng)) : dist(rng);
  }
  return out;
}

// Passes when |got - want| <= abs + rel * |want| everywhere; reports the
// worst ratio of error to that bound.
template <typename Want>
void expect_close(const std::vector<float> &got, const std::vector<Want> &want,
                  double abs, double rel, const std::string &what) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double w = static_cast<double>(want[i]);
    const double err = std::fabs(static_cast<double>(got[i]) - w);
    worst = std::max(worst, err / (abs + rel * std::fabs(w)));
  }
  const bool ok = worst <= 1.0 && got.size() == want.size();
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what
            << " (worst error / bound " << worst << ")\n";
  g_failures += ok ? 0 : 1;
}

double max_abs(const std::vector<float> &values) {
  double out = 0.0;
  for (float v : values) {
    out = std::max(out, static_cast<double>(std::fabs(v)));
  }
  return out;
}

// --- Finite differences (CPU backend, FP32 node vs FP64 math)

void test_finite_differences(DeviceBackend &backend, const Ops &ops,
                             const Shape4 &s, uint32_t seed) {
  const size_t qkv_count = static_cast<size_t>(s.batch * s.seq * 3 * s.model_dim);
  const size_t context_count = static_cast<size_t>(s.batch * s.seq * s.model_dim);
  const std::vector<float> qkv = random_values(qkv_count, seed, false);
  const std::vector<float> dcontext = random_values(context_count, seed + 1, false);
  const CoreResult got =
      run_core(backend, ops, CoreKind::Unfused, s, DType::F32, qkv, dcontext);

  std::vector<double> qkv64(qkv.begin(), qkv.end());
  const std::vector<double> context64 = attention_f64(qkv64, s);
  expect_close(got.context, context64, 1e-6, 1e-5,
               "reference forward vs FP64, " + s.text());

  // dL/dqkv for L = <context, dcontext>, by central differences.
  constexpr double kStep = 1e-5;
  std::vector<double> numeric(qkv_count);
  for (size_t i = 0; i < qkv_count; ++i) {
    const double saved = qkv64[i];
    qkv64[i] = saved + kStep;
    const double up = dot(attention_f64(qkv64, s), dcontext);
    qkv64[i] = saved - kStep;
    const double down = dot(attention_f64(qkv64, s), dcontext);
    qkv64[i] = saved;
    numeric[i] = (up - down) / (2.0 * kStep);
  }
  expect_close(got.dqkv, numeric, 1e-5, 1e-4,
               "reference backward vs finite differences of FP64, " + s.text());
}

// --- The backend's fused attention against FP64 math

// cudnn_flash keeps scores, softmax statistics and accumulation in FP32; it
// rounds the probabilities to BF16 for the P V product and each output once,
// so on BF16 inputs it must match the FP64 truth to about two BF16
// roundings: tighter than the differential bound, which allows for the
// unfused core's BF16 intermediates.
void test_flash_vs_f64(DeviceBackend &backend, const Ops &ops, const Shape4 &s,
                       uint32_t seed) {
  const size_t qkv_count = static_cast<size_t>(s.batch * s.seq * 3 * s.model_dim);
  const size_t context_count = static_cast<size_t>(s.batch * s.seq * s.model_dim);
  const std::vector<float> qkv = random_values(qkv_count, seed, true);
  const std::vector<float> dcontext = random_values(context_count, seed + 1, true);
  const CoreResult got =
      run_core(backend, ops, CoreKind::CudnnFlash, s, DType::BF16, qkv, dcontext);

  std::vector<double> qkv64(qkv.begin(), qkv.end());
  const std::vector<double> context64 = attention_f64(qkv64, s);
  constexpr double kStep = 1e-4;
  std::vector<double> numeric(qkv_count);
  for (size_t i = 0; i < qkv_count; ++i) {
    const double saved = qkv64[i];
    qkv64[i] = saved + kStep;
    const double up = dot(attention_f64(qkv64, s), dcontext);
    qkv64[i] = saved - kStep;
    const double down = dot(attention_f64(qkv64, s), dcontext);
    qkv64[i] = saved;
    numeric[i] = (up - down) / (2.0 * kStep);
  }
  const auto max_of = [](const std::vector<double> &v) {
    double out = 0.0;
    for (double x : v) out = std::max(out, std::fabs(x));
    return out;
  };
  const double bf16_round = std::ldexp(1.0, -8);
  expect_close(got.context, context64, 0.5 * bf16_round * max_of(context64), bf16_round,
               "cudnn_flash forward vs FP64, " + s.text());
  // dqkv also sees the BF16 rounding of the saved output (D = rowsum(dO*O)).
  expect_close(got.dqkv, numeric, bf16_round * max_of(numeric), 2.0 * bf16_round,
               "cudnn_flash backward vs finite differences of FP64, " + s.text());
}

// --- PyTorch goldens

std::string read_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

// The number or number array after "key": at or after `from`, within a flat
// case object; `from` moves past it.
std::vector<double> json_numbers(const std::string &json, const std::string &key,
                                 size_t &from) {
  const size_t at = json.find("\"" + key + "\":", from);
  if (at == std::string::npos) {
    throw std::runtime_error("golden file has no " + key);
  }
  size_t pos = at + key.size() + 3;
  std::vector<double> out;
  const bool array = json.find('[', pos) < json.find_first_of(",}", pos);
  const size_t end = array ? json.find(']', pos) : json.find_first_of(",}", pos);
  pos = array ? json.find('[', pos) + 1 : pos;
  while (pos < end) {
    const size_t next = json.find_first_of(",]}", pos);
    const std::string token = json.substr(pos, std::min(next, end) - pos);
    if (token.find_first_not_of(" \n\t") != std::string::npos) {
      out.push_back(std::stod(token));
    }
    pos = std::min(next, end) + 1;
  }
  from = end;
  return out;
}

void test_goldens(DeviceBackend &backend, const Ops &ops, const std::string &path) {
  const std::string json = read_file(path);
  size_t from = json.find("\"cases\"");
  while (json.find("\"batch\":", from) != std::string::npos) {
    const auto scalar = [&](const char *key) {
      return static_cast<int64_t>(json_numbers(json, key, from).at(0));
    };
    const Shape4 s{scalar("batch"), scalar("seq"), scalar("model_dim"),
                   scalar("heads")};
    const auto as_floats = [](const std::vector<double> &v) {
      return std::vector<float>(v.begin(), v.end());
    };
    const std::vector<float> qkv = as_floats(json_numbers(json, "qkv", from));
    const std::vector<float> dcontext = as_floats(json_numbers(json, "dcontext", from));
    const std::vector<double> context = json_numbers(json, "context", from);
    const std::vector<double> dqkv = json_numbers(json, "dqkv", from);
    const CoreResult got =
        run_core(backend, ops, CoreKind::Unfused, s, DType::F32, qkv, dcontext);
    expect_close(got.context, context, 1e-6, 1e-5,
                 "reference forward vs PyTorch golden, " + s.text());
    expect_close(got.dqkv, dqkv, 1e-6, 1e-5,
                 "reference backward vs PyTorch golden, " + s.text());
  }
}

// --- Differential: every other core against the unfused core

void test_differential(DeviceBackend &backend, const Ops &ops, CoreKind kind,
                       const Shape4 &s, DType act, uint32_t seed) {
  const bool bf16 = act == DType::BF16;
  const size_t qkv_count = static_cast<size_t>(s.batch * s.seq * 3 * s.model_dim);
  const size_t context_count = static_cast<size_t>(s.batch * s.seq * s.model_dim);
  const std::vector<float> qkv = random_values(qkv_count, seed, bf16);
  const std::vector<float> dcontext = random_values(context_count, seed + 1, bf16);
  const CoreResult want =
      run_core(backend, ops, CoreKind::Unfused, s, act, qkv, dcontext);
  const CoreResult got = run_core(backend, ops, kind, s, act, qkv, dcontext);
  // FP32: the same math in a different op order. BF16: every intermediate
  // (scores, weights, their gradients) is rounded to BF16 at different
  // points, so allow a few BF16 roundings of the tensor's scale.
  const std::string what = std::string(core_name(kind)) + " vs reference, " +
                           (bf16 ? "bf16act" : "f32") + ", " + s.text();
  const double context_abs = bf16 ? 4.0 * std::ldexp(1.0, -8) * max_abs(want.context) : 1e-6;
  const double dqkv_abs = bf16 ? 4.0 * std::ldexp(1.0, -8) * max_abs(want.dqkv) : 1e-6;
  expect_close(got.context, want.context, context_abs, bf16 ? 0.0 : 1e-5,
               what + " context");
  expect_close(got.dqkv, want.dqkv, dqkv_abs, bf16 ? 0.0 : 1e-5,
               what + " dqkv");
}

// --- Projections against FP64 host math

// One affine map's buffers: the master weight and bias F32, the compute
// weight and activations in `act`, the parameter gradients F32.
struct LinearBuffers {
  DeviceBuffer weight, compute_weight, bias, dweight, dbias;
  LinearBuffers(DeviceBackend &backend, DType act, int64_t in, int64_t out)
      : weight(backend, DType::F32, Shape{in, out}),
        compute_weight(backend, act, Shape{in, out}),
        bias(backend, DType::F32, Shape{1, out}),
        dweight(backend, DType::F32, Shape{in, out}),
        dbias(backend, DType::F32, Shape{1, out}) {}
  void upload(const std::vector<float> &w, const std::vector<float> &b) {
    weight.upload(w);
    compute_weight.upload(w);
    bias.upload(b);
  }
  LinearParams params() const {
    return LinearParams{weight.view(), compute_weight.view(), bias.view(),
                        dweight.view(), dbias.view()};
  }
};

// Uniform in [-scale, scale], BF16-representable when bf16.
std::vector<float> scaled_values(size_t count, uint32_t seed, float scale, bool bf16) {
  std::vector<float> out = random_values(count, seed, false);
  for (float &v : out) {
    v = bf16 ? round_to_bf16(v * scale) : v * scale;
  }
  return out;
}

enum class ProjectionSlot { Qkv, Output };

// Each projection variant against FP64 host math; the ReLU flag checks
// Ops::linear_forward's ReLU epilogue directly (no node has a ReLU).
void test_projection(DeviceBackend &backend, const Ops &ops, ProjectionSlot kind,
                     bool fused_linear, const Shape4 &s, DType act, uint32_t seed,
                     bool relu = false) {
  const bool bf16 = act == DType::BF16;
  const int64_t T = s.batch * s.seq;
  const int64_t in = s.model_dim;
  const int64_t out = kind == ProjectionSlot::Qkv ? 3 * s.model_dim : s.model_dim;
  const std::vector<float> x = random_values(static_cast<size_t>(T * in), seed, bf16);
  const std::vector<float> w = scaled_values(static_cast<size_t>(in * out), seed + 1,
                                             1.0f / std::sqrt(static_cast<float>(in)), bf16);
  const std::vector<float> b = random_values(static_cast<size_t>(out), seed + 2, bf16);
  const std::vector<float> dy = random_values(static_cast<size_t>(T * out), seed + 3, bf16);

  LinearBuffers linear(backend, act, in, out);
  linear.upload(w, b);
  DeviceBuffer x_buf(backend, act, Shape{s.batch, s.seq, in});
  DeviceBuffer y_buf(backend, act, Shape{s.batch, s.seq, out});
  DeviceBuffer dy_buf(backend, act, Shape{s.batch, s.seq, out});
  DeviceBuffer dx_buf(backend, act, Shape{s.batch, s.seq, in});
  x_buf.upload(x);
  dy_buf.upload(dy);
  TensorView y = y_buf.view();
  TensorView dx = dx_buf.view();
  std::unique_ptr<IQkvProjection> qkv_node;
  std::unique_ptr<IOutputProjection> out_node;
  if (relu) {
    const LinearParams p = linear.params();
    ops.linear_forward(x_buf.view(), p.compute_weight, p.bias, true, y);
  } else if (kind == ProjectionSlot::Qkv) {
    qkv_node = fused_linear
                   ? std::unique_ptr<IQkvProjection>(
                         std::make_unique<FusedLinearQkvProjection>(0, linear.params(), ops))
                   : std::make_unique<GemmQkvProjection>(0, linear.params(), ops);
    qkv_node->forward(x_buf.view(), y);
    qkv_node->backward(dy_buf.view(), dx);
  } else {
    out_node = fused_linear
                   ? std::unique_ptr<IOutputProjection>(
                         std::make_unique<FusedLinearOutputProjection>(0, linear.params(), ops))
                   : std::make_unique<GemmOutputProjection>(0, linear.params(), ops);
    out_node->forward(x_buf.view(), y);
    out_node->backward(dy_buf.view(), dx);
  }

  std::vector<double> y64(static_cast<size_t>(T * out)), dx64(static_cast<size_t>(T * in));
  std::vector<double> dw64(static_cast<size_t>(in * out)), db64(static_cast<size_t>(out));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t o = 0; o < out; ++o) {
      double acc = b[static_cast<size_t>(o)];
      const double g = dy[static_cast<size_t>(t * out + o)];
      for (int64_t i = 0; i < in; ++i) {
        const double xi = x[static_cast<size_t>(t * in + i)];
        const double wio = w[static_cast<size_t>(i * out + o)];
        acc += xi * wio;
        dx64[static_cast<size_t>(t * in + i)] += g * wio;
        dw64[static_cast<size_t>(i * out + o)] += xi * g;
      }
      y64[static_cast<size_t>(t * out + o)] = relu ? std::max(acc, 0.0) : acc;
      db64[static_cast<size_t>(o)] += g;
    }
  }
  const std::string what =
      std::string(fused_linear ? "fused_linear " : "gemm ") +
      (relu ? "linear_forward relu" : kind == ProjectionSlot::Qkv ? "qkv projection" : "output projection") +
      " vs FP64, " + (bf16 ? "bf16act" : "f32") + ", " + s.text();
  // BF16 outputs carry one rounding of their scale; F32 parameter gradients
  // accumulate exact BF16 products in FP32.
  const double round = bf16 ? 2.0 * std::ldexp(1.0, -8) : 0.0;
  expect_close(y_buf.download(), y64, bf16 ? round * max_abs(y_buf.download()) : 1e-5,
               1e-5, what + " output");
  if (relu) {
    return;
  }
  expect_close(dx_buf.download(), dx64, bf16 ? round * max_abs(dx_buf.download()) : 1e-5,
               1e-5, what + " dx");
  expect_close(linear.dweight.download(), dw64, 1e-5, 1e-5, what + " dweight");
  expect_close(linear.dbias.download(), db64, 1e-5, 1e-5, what + " dbias");
}

// --- Multistream attention against the composed reference

// A position encoding with a gradient: scales Q, K and V by 0.75, so a
// variant that applies or reverses it at the wrong point differs.
class ScalingPositionEncoding final : public IPositionEncoding {
public:
  explicit ScalingPositionEncoding(const Ops &ops) : ops_(ops) {}
  const char *name() const override { return "test_scaling"; }
  bool needs_position_table() const override { return false; }
  void bind(TensorStore &, GradientStore *, const Ops &) override {}
  void set_diagnostics(TrainingDiagnosticsController *) override {}
  void apply_to_input(TensorView &) override {}
  void backward_input(const TensorView &) override {}
  void apply_to_qk(TensorView &qkv) override { ops_.mul_scalar(qkv, 0.75f, qkv); }
  void backward_qk(TensorView &dqkv) override { ops_.mul_scalar(dqkv, 0.75f, dqkv); }
  void report_probes() const override {}

private:
  const Ops &ops_;
};

struct AttentionResult {
  std::vector<float> out, dx, dwqkv, dbqkv, dwo, dbo;
};

AttentionResult run_attention(DeviceBackend &backend, const Ops &ops, bool multistream,
                              const Shape4 &s, DType act, uint32_t seed) {
  const bool bf16 = act == DType::BF16;
  const int64_t D = s.model_dim;
  const size_t tokens = static_cast<size_t>(s.batch * s.seq);
  Config cfg{};
  cfg.model.d_model = static_cast<uint32_t>(D);
  cfg.model.n_heads = static_cast<uint32_t>(s.heads);
  const AttentionDims dims{s.batch, s.seq, D, s.heads, true};
  const float w_scale = 1.0f / std::sqrt(static_cast<float>(D));

  LinearBuffers qkv(backend, act, D, 3 * D);
  LinearBuffers out(backend, act, D, D);
  qkv.upload(scaled_values(static_cast<size_t>(3 * D * D), seed, w_scale, bf16),
             random_values(static_cast<size_t>(3 * D), seed + 1, bf16));
  out.upload(scaled_values(static_cast<size_t>(D * D), seed + 2, w_scale, bf16),
             random_values(static_cast<size_t>(D), seed + 3, bf16));
  const AttentionParams params{qkv.params(), out.params()};

  std::vector<SliceRequest> requests;
  if (multistream) {
    requests = SelfAttentionFusedInplaceMultistream::slice_requests(dims);
  } else {
    requests = ComposedSelfAttention::slice_requests(dims);
    for (const SliceRequest &request : UnfusedAttentionCore::slice_requests(dims)) {
      requests.push_back(request);
    }
  }
  std::vector<std::unique_ptr<DeviceBuffer>> storage;
  NodeSlices slices;
  for (const SliceRequest &request : requests) {
    storage.push_back(std::make_unique<DeviceBuffer>(backend, act, request.shape));
    slices.add(request.name, storage.back()->view());
  }
  ScalingPositionEncoding position(ops);
  std::unique_ptr<ISelfAttention> attention;
  if (multistream) {
    attention = std::make_unique<SelfAttentionFusedInplaceMultistream>(
        0, cfg, params, ops, position, slices);
  } else {
    attention = std::make_unique<ComposedSelfAttention>(
        0, cfg, params, position, slices,
        std::make_unique<GemmQkvProjection>(0, params.qkv, ops),
        std::make_unique<UnfusedAttentionCore>(0, cfg, ops, slices),
        std::make_unique<GemmOutputProjection>(0, params.out, ops));
  }

  DeviceBuffer x_buf(backend, act, Shape{s.batch, s.seq, D});
  DeviceBuffer y_buf(backend, act, Shape{s.batch, s.seq, D});
  DeviceBuffer dy_buf(backend, act, Shape{s.batch, s.seq, D});
  DeviceBuffer dx_buf(backend, act, Shape{s.batch, s.seq, D});
  x_buf.upload(random_values(tokens * static_cast<size_t>(D), seed + 4, bf16));
  dy_buf.upload(random_values(tokens * static_cast<size_t>(D), seed + 5, bf16));
  TensorView y = y_buf.view();
  TensorView dx = dx_buf.view();
  attention->forward(x_buf.view(), y);
  attention->backward(dy_buf.view(), dx);
  return {y_buf.download(),          dx_buf.download(),
          qkv.dweight.download(),    qkv.dbias.download(),
          out.dweight.download(),    out.dbias.download()};
}

void test_multistream(DeviceBackend &backend, const Ops &ops, const Shape4 &s,
                      DType act, uint32_t seed) {
  const bool bf16 = act == DType::BF16;
  const AttentionResult want = run_attention(backend, ops, false, s, act, seed);
  const AttentionResult got = run_attention(backend, ops, true, s, act, seed);
  const std::string what = std::string("multistream vs composed reference, ") +
                           (bf16 ? "bf16act" : "f32") + ", " + s.text();
  // As the core differential: a few BF16 roundings of each tensor's scale.
  const auto check = [&](const std::vector<float> &g, const std::vector<float> &w,
                         const std::string &name) {
    expect_close(g, w, bf16 ? 4.0 * std::ldexp(1.0, -8) * max_abs(w) : 1e-5,
                 bf16 ? 0.0 : 1e-5, what + " " + name);
  };
  check(got.out, want.out, "out");
  check(got.dx, want.dx, "dx");
  check(got.dwqkv, want.dwqkv, "dWqkv");
  check(got.dbqkv, want.dbqkv, "dbqkv");
  check(got.dwo, want.dwo, "dWo");
  check(got.dbo, want.dbo, "dbo");
}

// --- GuardedSelfAttention

// Writes `count` floats of 1.0 at the start of slice "attn.w" in forward
// and backward: count above the slice's 4 elements overruns it.
class OverrunningAttention final : public ISelfAttention {
public:
  OverrunningAttention(DeviceBackend &backend, TensorView w, size_t count)
      : backend_(backend), w_(w), count_(count) {}
  void set_observer(ITrainingObserver *) override {}
  void set_diagnostics(TrainingDiagnosticsController *) override {}
  void forward(const TensorView &, TensorView &) override { write(); }
  void backward(const TensorView &, TensorView &) override { write(); }

private:
  void write() {
    const std::vector<float> ones(count_, 1.0f);
    backend_.copy_host2device(w_.data(), ones.data(), count_ * sizeof(float));
  }
  DeviceBackend &backend_;
  TensorView w_;
  size_t count_;
};

// Runs forward and backward of a guarded OverrunningAttention; between
// them, overwrites the Step slice "attn.saved" when change_saved. Returns
// the error message, or "" when nothing was caught.
std::string run_guarded(DeviceBackend &backend, size_t write_count, bool change_saved) {
  const std::vector<SliceRequest> requests{
      {"attn.saved", Shape{4}, ItemType::Activation, SliceRole::Saved, SliceLifetime::Step},
      {"attn.w", Shape{4}, ItemType::Activation, SliceRole::Workspace, SliceLifetime::Op},
  };
  // One F32 buffer, slices back to back as with_guards orders them.
  const std::vector<SliceRequest> guarded = GuardedSelfAttention::with_guards(requests);
  int64_t total = 0;
  for (const SliceRequest &request : guarded) {
    total += request.shape.numel();
  }
  DeviceBuffer arena(backend, DType::F32, Shape{total});
  NodeSlices slices;
  int64_t offset = 0;
  for (const SliceRequest &request : guarded) {
    slices.add(request.name,
               TensorView(backend.device(), DType::F32,
                          static_cast<float *>(arena.view().data()) + offset, request.shape));
    offset += request.shape.numel();
  }
  DeviceBuffer x_buf(backend, DType::F32, Shape{1, 2, 4});
  DeviceBuffer y_buf(backend, DType::F32, Shape{1, 2, 4});
  TensorView y = y_buf.view();
  GuardedSelfAttention attention(
      0, std::make_unique<OverrunningAttention>(backend, slices.at("attn.w"), write_count),
      requests, slices, backend);
  try {
    attention.forward(x_buf.view(), y);
    if (change_saved) {
      const float changed = 2.0f;
      backend.copy_host2device(slices.at("attn.saved").data(), &changed, sizeof(float));
    }
    attention.backward(y_buf.view(), y);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "";
}

void test_guards(DeviceBackend &backend) {
  const auto expect = [](bool ok, const std::string &what, const std::string &message) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what
              << (message.empty() ? "" : " (" + message + ")") << "\n";
    g_failures += ok ? 0 : 1;
  };
  const std::string clean = run_guarded(backend, 4, false);
  expect(clean.empty(), "guarded attention: clean run passes", clean);
  const std::string overrun = run_guarded(backend, 5, false);
  expect(overrun.find("write past the end of slice attn.w") != std::string::npos,
         "guarded attention: write past a slice is caught", overrun);
  const std::string changed = run_guarded(backend, 4, true);
  expect(changed.find("saved state attn.saved changed") != std::string::npos,
         "guarded attention: saved-state change is caught", changed);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <backend plugin .so> <attention_core.json>\n";
    return 2;
  }
  Config cfg{};
  cfg.backend.library = argv[1];
  std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(cfg);
  Ops ops(*backend);
  std::cout << "Attention-core contract tests on " << argv[1] << "\n";

  const std::vector<Shape4> shapes{{2, 5, 8, 2}, {1, 7, 12, 3}, {3, 9, 16, 4}};
  if (backend->device() == Device::CPU) {
    for (size_t i = 0; i < 2; ++i) {
      test_finite_differences(*backend, ops, shapes[i], 100 + static_cast<uint32_t>(i));
    }
  }
  test_goldens(*backend, ops, argv[2]);
  const bool has_bf16 = (backend->precision_variants() & kPrecisionVariantBf16Act) != 0;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const uint32_t seed = 200 + 10 * static_cast<uint32_t>(i);
    test_differential(*backend, ops, CoreKind::FusedInplace, shapes[i], DType::F32, seed);
    if (has_bf16) {
      test_differential(*backend, ops, CoreKind::FusedInplace, shapes[i], DType::BF16, seed);
    }
  }
  for (size_t i = 0; i < shapes.size(); ++i) {
    const uint32_t seed = 300 + 10 * static_cast<uint32_t>(i);
    for (ProjectionSlot kind : {ProjectionSlot::Qkv, ProjectionSlot::Output}) {
      for (bool fused_linear : {false, true}) {
        test_projection(*backend, ops, kind, fused_linear, shapes[i], DType::F32, seed);
        if (has_bf16) {
          test_projection(*backend, ops, kind, fused_linear, shapes[i], DType::BF16, seed);
        }
      }
    }
    test_projection(*backend, ops, ProjectionSlot::Output, true, shapes[i], DType::F32,
                    seed + 7, true);
    if (has_bf16) {
      test_projection(*backend, ops, ProjectionSlot::Output, true, shapes[i], DType::BF16,
                      seed + 7, true);
    }
  }
  // The fused op needs head_dim a multiple of 8; the last shape is the
  // benchmark's head layout (head_dim 64) at a short sequence.
  if ((ops.fused_ops() & kFusedOpCausalAttentionBf16Act) != 0) {
    const std::vector<Shape4> flash_shapes{
        {2, 5, 16, 2}, {1, 7, 24, 3}, {3, 9, 64, 4}, {2, 33, 256, 4}};
    for (size_t i = 0; i < flash_shapes.size(); ++i) {
      const uint32_t seed = 500 + 10 * static_cast<uint32_t>(i);
      test_differential(*backend, ops, CoreKind::CudnnFlash, flash_shapes[i],
                        DType::BF16, seed);
      if (i < 3) {
        test_flash_vs_f64(*backend, ops, flash_shapes[i], seed + 5);
      }
    }
  } else {
    std::cout << "[SKIP] cudnn_flash: backend has no fused causal attention\n";
  }
  if (ops.supports_exec_context_iteration()) {
    for (size_t i = 0; i < shapes.size(); ++i) {
      const uint32_t seed = 400 + 10 * static_cast<uint32_t>(i);
      test_multistream(*backend, ops, shapes[i], DType::F32, seed);
      if (has_bf16) {
        test_multistream(*backend, ops, shapes[i], DType::BF16, seed);
      }
    }
  } else {
    std::cout << "[SKIP] multistream: backend has no exec contexts\n";
  }
  test_guards(*backend);

  std::cout << (g_failures == 0 ? "All attention-core contract tests passed.\n"
                                : "Attention-core contract tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
