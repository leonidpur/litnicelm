// Runs every DeviceBackend op on a backend plugin, through the plugin API
// as the host uses it, and compares the results with the host's built-in
// DefaultCpuBackend on the same inputs.
//
// Inputs are seeded random values; shapes avoid multiples of thread-block
// sizes and include rank-3/4 tensors, I32 ids and a column-slice view.
// Output buffers start out holding nonzero values, so an op that
// accumulates where the reference overwrites (or the reverse) fails.
//
// Usage: litnice_backend_ops_test <path to backend plugin .so>

#include "backend/device_backend.hpp"
#include "tensor.hpp"

#include <config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string sci(double v) {
  std::ostringstream oss;
  oss << std::scientific << v;
  return oss.str();
}

// A tensor's host values: F32 values, or I32 ids stored as floats.
struct HostTensor {
  DType dtype;
  Shape shape;
  std::vector<float> values;
};

HostTensor random_f32(Shape shape, uint32_t seed, float lo = -1.0f,
                      float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  HostTensor t{DType::F32, shape, std::vector<float>(shape.numel())};
  for (float &v : t.values) {
    v = dist(rng);
  }
  return t;
}

HostTensor random_ids(Shape shape, int64_t limit, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, limit - 1);
  HostTensor t{DType::I32, shape, std::vector<float>(shape.numel())};
  for (float &v : t.values) {
    v = static_cast<float>(dist(rng));
  }
  return t;
}

// A HostTensor uploaded to a backend's memory.
class DeviceTensor {
public:
  DeviceTensor(DeviceBackend &backend, const HostTensor &host)
      : backend_(backend), dtype_(host.dtype), shape_(host.shape),
        bytes_(nbytes(host.shape, host.dtype)) {
    data_ = backend_.alloc(bytes_, 64);
    std::vector<uint8_t> raw(static_cast<size_t>(bytes_));
    for (size_t i = 0; i < host.values.size(); ++i) {
      if (dtype_ == DType::I32) {
        const int32_t v = static_cast<int32_t>(host.values[i]);
        std::memcpy(raw.data() + i * sizeof(int32_t), &v, sizeof(v));
      } else {
        std::memcpy(raw.data() + i * sizeof(float), &host.values[i],
                    sizeof(float));
      }
    }
    backend_.copy_host2device(data_, raw.data(), bytes_);
  }
  ~DeviceTensor() { backend_.free(data_); }
  DeviceTensor(const DeviceTensor &) = delete;
  DeviceTensor &operator=(const DeviceTensor &) = delete;

  TensorView view() const {
    return TensorView(backend_.device(), dtype_, data_, shape_);
  }
  std::vector<float> download() const {
    std::vector<float> out(shape_.numel());
    backend_.copy_device2host(out.data(), data_, bytes_);
    return out;
  }

private:
  DeviceBackend &backend_;
  DType dtype_;
  Shape shape_;
  uint64_t bytes_;
  void *data_ = nullptr;
};

// One op test: the tensors it uses (inputs and outputs, in order), the call,
// which tensors are outputs to compare, and the tolerance
// |got - ref| <= abs_tol + rel_tol * |ref|.
struct OpCase {
  std::string name;
  std::vector<HostTensor> tensors;
  // Runs the op on views of the tensors; returns any scalar results.
  std::function<std::vector<double>(DeviceBackend &, std::vector<TensorView> &)>
      run;
  std::vector<size_t> outputs;
  double rel_tol;
  double abs_tol;
  // When set, the reference backend runs this instead of run (for ops only
  // the plugin implements, such as the exec-context variants).
  std::function<std::vector<double>(DeviceBackend &, std::vector<TensorView> &)>
      run_reference;
};

struct Result {
  std::vector<std::vector<float>> outputs;
  std::vector<double> scalars;
};

Result execute(DeviceBackend &backend, const OpCase &op, bool reference) {
  std::vector<std::unique_ptr<DeviceTensor>> tensors;
  std::vector<TensorView> views;
  for (const HostTensor &host : op.tensors) {
    tensors.push_back(std::make_unique<DeviceTensor>(backend, host));
    views.push_back(tensors.back()->view());
  }
  Result result;
  result.scalars = reference && op.run_reference ? op.run_reference(backend, views)
                                                 : op.run(backend, views);
  for (size_t index : op.outputs) {
    result.outputs.push_back(tensors[index]->download());
  }
  return result;
}

bool close(double got, double ref, const OpCase &op) {
  return std::fabs(got - ref) <= op.abs_tol + op.rel_tol * std::fabs(ref);
}

void compare(DeviceBackend &reference, DeviceBackend &plugin, const OpCase &op) {
  Result want;
  Result got;
  try {
    want = execute(reference, op, true);
    got = execute(plugin, op, false);
  } catch (const std::exception &e) {
    std::cout << "[FAIL] " << op.name << ": " << e.what() << "\n";
    ++g_failures;
    return;
  }
  double worst = 0.0;
  bool ok = true;
  for (size_t o = 0; o < want.outputs.size(); ++o) {
    for (size_t i = 0; i < want.outputs[o].size(); ++i) {
      const double r = want.outputs[o][i];
      const double g = got.outputs[o][i];
      worst = std::max(worst, std::fabs(g - r));
      ok = ok && close(g, r, op);
    }
  }
  for (size_t s = 0; s < want.scalars.size(); ++s) {
    worst = std::max(worst, std::fabs(got.scalars[s] - want.scalars[s]));
    ok = ok && close(got.scalars[s], want.scalars[s], op);
  }
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << op.name << " (max abs diff "
            << sci(worst) << ")\n";
  if (!ok) {
    ++g_failures;
  }
}

using Views = std::vector<TensorView>;
using Scalars = std::vector<double>;

constexpr double kExact = 1e-6;   // elementwise: same arithmetic per element
constexpr double kReduce = 1e-5;  // reductions, GEMMs, atomics: order differs

// Shapes: 37 rows x 70 cols for matrices; [3, 17, 40] for activations.
std::vector<OpCase> op_cases() {
  const Shape mat{37, 70};
  const Shape bsd{3, 17, 40};
  const Shape bias{1, 40};
  std::vector<OpCase> cases;

  cases.push_back({"copy", {random_f32(mat, 1), random_f32(mat, 2)},
                   [](DeviceBackend &b, Views &v) { b.copy(v[0], v[1]); return Scalars{}; },
                   {1}, 0.0, 0.0});
  cases.push_back({"fill", {random_f32(mat, 3)},
                   [](DeviceBackend &b, Views &v) { b.fill(v[0], 0.375f); return Scalars{}; },
                   {0}, 0.0, 0.0});
  cases.push_back({"add", {random_f32(bsd, 4), random_f32(bsd, 5), random_f32(bsd, 6)},
                   [](DeviceBackend &b, Views &v) { b.add(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kExact, kExact});
  cases.push_back({"add [B,S,D] + [S,D] (position broadcast)",
                   {random_f32(bsd, 7), random_f32(Shape{17, 40}, 8), random_f32(bsd, 9)},
                   [](DeviceBackend &b, Views &v) { b.add(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kExact, kExact});
  cases.push_back({"add on column slices (strided rows)",
                   {random_f32(Shape{3, 17, 120}, 10), random_f32(Shape{3, 17, 120}, 11)},
                   [](DeviceBackend &b, Views &v) {
                     TensorView a = v[0].subcols(40, 40);
                     TensorView out = v[1].subcols(80, 40);
                     b.add(a, v[0].subcols(0, 40), out);
                     return Scalars{};
                   },
                   {1}, kExact, kExact});
  cases.push_back({"add_inplace", {random_f32(bsd, 12), random_f32(bsd, 13)},
                   [](DeviceBackend &b, Views &v) { b.add_inplace(v[0], v[1]); return Scalars{}; },
                   {0}, kExact, kExact});
  cases.push_back({"add_bias_rowwise", {random_f32(bsd, 14), random_f32(bias, 15), random_f32(bsd, 16)},
                   [](DeviceBackend &b, Views &v) { b.add_bias_rowwise(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kExact, kExact});
  cases.push_back({"add_bias_relu_rowwise",
                   {random_f32(bsd, 17), random_f32(bias, 18), random_f32(bsd, 19)},
                   [](DeviceBackend &b, Views &v) { b.add_bias_relu_rowwise(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kExact, kExact});
  cases.push_back({"add_bias_relu_rowwise_inplace", {random_f32(bsd, 20), random_f32(bias, 21)},
                   [](DeviceBackend &b, Views &v) { b.add_bias_relu_rowwise_inplace(v[0], v[1]); return Scalars{}; },
                   {0}, kExact, kExact});
  cases.push_back({"mul_scalar", {random_f32(bsd, 22), random_f32(bsd, 23)},
                   [](DeviceBackend &b, Views &v) { b.mul_scalar(v[0], -1.75f, v[1]); return Scalars{}; },
                   {1}, kExact, kExact});
  cases.push_back({"relu", {random_f32(bsd, 24), random_f32(bsd, 25)},
                   [](DeviceBackend &b, Views &v) { b.relu(v[0], v[1]); return Scalars{}; },
                   {1}, 0.0, 0.0});
  cases.push_back({"relu_backward", {random_f32(bsd, 26), random_f32(bsd, 27), random_f32(bsd, 28)},
                   [](DeviceBackend &b, Views &v) { b.relu_backward(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, 0.0, 0.0});
  cases.push_back({"relu_backward_inplace", {random_f32(bsd, 29), random_f32(bsd, 30)},
                   [](DeviceBackend &b, Views &v) { b.relu_backward_inplace(v[0], v[1]); return Scalars{}; },
                   {1}, 0.0, 0.0});
  cases.push_back({"sum_squares_f32", {random_f32(bsd, 31)},
                   [](DeviceBackend &b, Views &v) { return Scalars{b.sum_squares_f32(v[0])}; },
                   {}, kReduce, kReduce});
  cases.push_back({"row_sum", {random_f32(mat, 32), random_f32(Shape{1, 70}, 33)},
                   [](DeviceBackend &b, Views &v) { b.row_sum(v[0], v[1]); return Scalars{}; },
                   {1}, kReduce, kReduce});
  cases.push_back({"transpose", {random_f32(mat, 34), random_f32(Shape{70, 37}, 35)},
                   [](DeviceBackend &b, Views &v) { b.transpose(v[0], v[1]); return Scalars{}; },
                   {1}, 0.0, 0.0});

  // GEMMs: [m, k] x [k, n] with m = 17 (per batch), k = 24, n = 29.
  cases.push_back({"gemm", {random_f32(Shape{37, 24}, 40), random_f32(Shape{24, 29}, 41), random_f32(Shape{37, 29}, 42)},
                   [](DeviceBackend &b, Views &v) { b.gemm(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"gemm_ranked_matrix_rhs",
                   {random_f32(Shape{3, 17, 24}, 43), random_f32(Shape{24, 29}, 44), random_f32(Shape{3, 17, 29}, 45)},
                   [](DeviceBackend &b, Views &v) { b.gemm_ranked_matrix_rhs(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"gemm_ranked_matrix_rhs_t",
                   {random_f32(Shape{3, 17, 24}, 46), random_f32(Shape{29, 24}, 47), random_f32(Shape{3, 17, 29}, 48)},
                   [](DeviceBackend &b, Views &v) { b.gemm_ranked_matrix_rhs_t(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"gemm_ranked_reduce_lhs_t",
                   {random_f32(Shape{3, 17, 24}, 49), random_f32(Shape{3, 17, 29}, 50), random_f32(Shape{24, 29}, 51)},
                   [](DeviceBackend &b, Views &v) { b.gemm_ranked_reduce_lhs_t(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"gemm_batched",
                   {random_f32(Shape{4, 17, 24}, 52), random_f32(Shape{4, 24, 29}, 53), random_f32(Shape{4, 17, 29}, 54)},
                   [](DeviceBackend &b, Views &v) { b.gemm_batched(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"gemm_batched_lhs_t",
                   {random_f32(Shape{4, 24, 17}, 55), random_f32(Shape{4, 24, 29}, 56), random_f32(Shape{4, 17, 29}, 57)},
                   [](DeviceBackend &b, Views &v) { b.gemm_batched_lhs_t(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"gemm_batched_rhs_t",
                   {random_f32(Shape{4, 17, 24}, 58), random_f32(Shape{4, 29, 24}, 59), random_f32(Shape{4, 17, 29}, 60)},
                   [](DeviceBackend &b, Views &v) { b.gemm_batched_rhs_t(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});

  cases.push_back({"layernorm_forward",
                   {random_f32(bsd, 61), random_f32(bias, 62), random_f32(bias, 63), random_f32(bsd, 64)},
                   [](DeviceBackend &b, Views &v) { b.layernorm_forward(v[0], v[1], v[2], v[3]); return Scalars{}; },
                   {3}, kReduce, kReduce});
  cases.push_back({"layernorm_backward",
                   {random_f32(bsd, 65), random_f32(bias, 66), random_f32(bsd, 67), random_f32(bsd, 68),
                    random_f32(bias, 69), random_f32(bias, 70)},
                   [](DeviceBackend &b, Views &v) {
                     b.layernorm_backward(v[0], v[1], v[2], v[3], v[4], v[5]);
                     return Scalars{};
                   },
                   {3, 4, 5}, kReduce, kReduce});

  // Embeddings and loss: vocabulary 50, ids [3, 17].
  const Shape ids{3, 17};
  cases.push_back({"embedding_lookup",
                   {random_f32(Shape{50, 40}, 71), random_ids(ids, 50, 72), random_f32(bsd, 73)},
                   [](DeviceBackend &b, Views &v) { b.embedding_lookup(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, 0.0, 0.0});
  // Accumulates into d_tok by definition, so d_tok starts nonzero on both.
  cases.push_back({"accumulate_embedding_grads",
                   {random_ids(ids, 50, 74), random_f32(bsd, 75), random_f32(Shape{50, 40}, 76)},
                   [](DeviceBackend &b, Views &v) { b.accumulate_embedding_grads(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  const Shape logits{3, 17, 50};
  cases.push_back({"cross_entropy_mean",
                   {random_f32(logits, 77, -3.0f, 3.0f), random_ids(ids, 50, 78), random_f32(Shape{1, 1}, 79)},
                   [](DeviceBackend &b, Views &v) { b.cross_entropy_mean(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"cross_entropy_mean_backward_inplace",
                   {random_f32(logits, 80, -3.0f, 3.0f), random_ids(ids, 50, 81), random_f32(Shape{1, 1}, 82)},
                   [](DeviceBackend &b, Views &v) {
                     b.cross_entropy_mean_backward_inplace(v[0], v[1], v[2]);
                     return Scalars{};
                   },
                   {0, 2}, kReduce, kReduce});
  cases.push_back({"backward_from_logits_targets",
                   {random_f32(logits, 83, -3.0f, 3.0f), random_ids(ids, 50, 84)},
                   [](DeviceBackend &b, Views &v) { b.backward_from_logits_targets(v[0], v[1]); return Scalars{}; },
                   {0}, kReduce, kReduce});
  cases.push_back({"read_scalar_f32", {random_f32(Shape{1, 1}, 85)},
                   [](DeviceBackend &b, Views &v) { return Scalars{b.read_scalar_f32(v[0])}; },
                   {}, 0.0, 0.0});

  // Softmax: rows of 70; causal variants on [2, 3, 17, 17] scores.
  cases.push_back({"softmax_rows", {random_f32(mat, 86, -4.0f, 4.0f), random_f32(mat, 87)},
                   [](DeviceBackend &b, Views &v) { b.softmax_rows(v[0], v[1]); return Scalars{}; },
                   {1}, kReduce, kReduce});
  cases.push_back({"softmax_backward_rows",
                   {random_f32(mat, 88, 0.0f, 0.1f), random_f32(mat, 89), random_f32(mat, 90)},
                   [](DeviceBackend &b, Views &v) { b.softmax_backward_rows(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  const Shape scores{2, 3, 17, 17};
  cases.push_back({"scaled_causal_softmax_rows",
                   {random_f32(scores, 91, -4.0f, 4.0f), random_f32(scores, 92)},
                   [](DeviceBackend &b, Views &v) { b.scaled_causal_softmax_rows(v[0], 0.35f, v[1]); return Scalars{}; },
                   {1}, kReduce, kReduce});
  cases.push_back({"softmax_backward_causal_rows",
                   {random_f32(scores, 93, 0.0f, 0.1f), random_f32(scores, 94), random_f32(scores, 95)},
                   [](DeviceBackend &b, Views &v) { b.softmax_backward_causal_rows(v[0], v[1], v[2]); return Scalars{}; },
                   {2}, kReduce, kReduce});
  cases.push_back({"apply_causal_mask_inplace", {random_f32(scores, 96)},
                   [](DeviceBackend &b, Views &v) { b.apply_causal_mask_inplace(v[0], -1e9f); return Scalars{}; },
                   {0}, 0.0, 0.0});

  // RoPE on [3, 17, 24]: 3 heads of 8.
  for (const bool inverse : {false, true}) {
    cases.push_back({std::string("rotary_embedding_inplace") + (inverse ? " (inverse)" : ""),
                     {random_f32(Shape{3, 17, 24}, inverse ? 98 : 97)},
                     [inverse](DeviceBackend &b, Views &v) {
                       b.rotary_embedding_inplace(v[0], 3, 10000.0f, inverse);
                       return Scalars{};
                     },
                     {0}, kReduce, kReduce});
  }

  cases.push_back({"adamw_step",
                   {random_f32(mat, 99), random_f32(mat, 100), random_f32(mat, 101),
                    random_f32(mat, 102, 0.0f, 1.0f)},
                   [](DeviceBackend &b, Views &v) {
                     b.adamw_step(v[0], v[1], v[2], v[3], 3, 1e-3f, 0.9f, 0.95f, 0.1f, true);
                     return Scalars{};
                   },
                   {0, 2, 3}, kReduce, kReduce});
  return cases;
}

// The exec-context variants exist only on backends that run attention on
// several streams; the reference runs the plain op.
std::vector<OpCase> exec_context_cases() {
  auto in_group = [](DeviceBackend &b, const std::function<void()> &op) {
    b.start_exec_context_iteration();
    b.start_exec_context_group();
    op();
    b.finish_exec_context_group();
    b.finish_exec_context_iteration();
  };
  const Shape scores{2, 3, 17, 17};
  std::vector<OpCase> cases;
  cases.push_back({"gemm_batched_exec_context",
                   {random_f32(Shape{4, 17, 24}, 110), random_f32(Shape{4, 24, 29}, 111), random_f32(Shape{4, 17, 29}, 112)},
                   [in_group](DeviceBackend &b, Views &v) {
                     in_group(b, [&] { b.gemm_batched_exec_context(v[0], v[1], v[2]); });
                     return Scalars{};
                   },
                   {2}, kReduce, kReduce,
                   [](DeviceBackend &b, Views &v) { b.gemm_batched(v[0], v[1], v[2]); return Scalars{}; }});
  cases.push_back({"gemm_batched_lhs_t_exec_context",
                   {random_f32(Shape{4, 24, 17}, 113), random_f32(Shape{4, 24, 29}, 114), random_f32(Shape{4, 17, 29}, 115)},
                   [in_group](DeviceBackend &b, Views &v) {
                     in_group(b, [&] { b.gemm_batched_lhs_t_exec_context(v[0], v[1], v[2]); });
                     return Scalars{};
                   },
                   {2}, kReduce, kReduce,
                   [](DeviceBackend &b, Views &v) { b.gemm_batched_lhs_t(v[0], v[1], v[2]); return Scalars{}; }});
  cases.push_back({"gemm_batched_rhs_t_exec_context",
                   {random_f32(Shape{4, 17, 24}, 116), random_f32(Shape{4, 29, 24}, 117), random_f32(Shape{4, 17, 29}, 118)},
                   [in_group](DeviceBackend &b, Views &v) {
                     in_group(b, [&] { b.gemm_batched_rhs_t_exec_context(v[0], v[1], v[2]); });
                     return Scalars{};
                   },
                   {2}, kReduce, kReduce,
                   [](DeviceBackend &b, Views &v) { b.gemm_batched_rhs_t(v[0], v[1], v[2]); return Scalars{}; }});
  cases.push_back({"scaled_causal_softmax_rows_exec_context",
                   {random_f32(scores, 119, -4.0f, 4.0f), random_f32(scores, 120)},
                   [in_group](DeviceBackend &b, Views &v) {
                     in_group(b, [&] { b.scaled_causal_softmax_rows_exec_context(v[0], 0.35f, v[1]); });
                     return Scalars{};
                   },
                   {1}, kReduce, kReduce,
                   [](DeviceBackend &b, Views &v) { b.scaled_causal_softmax_rows(v[0], 0.35f, v[1]); return Scalars{}; }});
  cases.push_back({"softmax_backward_causal_rows_exec_context",
                   {random_f32(scores, 121, 0.0f, 0.1f), random_f32(scores, 122), random_f32(scores, 123)},
                   [in_group](DeviceBackend &b, Views &v) {
                     in_group(b, [&] { b.softmax_backward_causal_rows_exec_context(v[0], v[1], v[2]); });
                     return Scalars{};
                   },
                   {2}, kReduce, kReduce,
                   [](DeviceBackend &b, Views &v) { b.softmax_backward_causal_rows(v[0], v[1], v[2]); return Scalars{}; }});
  return cases;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <backend plugin .so>\n";
    return 2;
  }
  Config cfg{};
  cfg.backend.library = argv[1];
  std::unique_ptr<DeviceBackend> plugin = DeviceBackend::create_instance(cfg);
  DefaultCpuBackend reference;

  std::cout << "Backend op tests on " << argv[1]
            << " against DefaultCpuBackend\n";
  for (const OpCase &op : op_cases()) {
    compare(reference, *plugin, op);
  }
  if (plugin->supports_exec_context_iteration()) {
    for (const OpCase &op : exec_context_cases()) {
      compare(reference, *plugin, op);
    }
  } else {
    std::cout << "[SKIP] exec-context ops (not supported by this backend)\n";
  }

  std::cout << (g_failures == 0 ? "All backend op tests passed.\n"
                                : "Backend op tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
