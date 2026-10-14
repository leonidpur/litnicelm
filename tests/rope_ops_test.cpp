// Checks Ops::rotary_embedding_inplace on a given backend plugin:
//   1. matches a double-precision host reference (Q/K rotated, V untouched),
//   2. inverse undoes forward,
//   3. inverse is the adjoint of forward (so it is the correct gradient),
//   4. Q.K scores depend only on the relative position i - j.
//
// Usage: litnice_rope_test <path to backend plugin .so>

#include "backend/device_backend.hpp"
#include "ops.hpp"
#include "tensor.hpp"

#include <config.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t kBatch = 2;
constexpr int64_t kSeq = 64;
constexpr int64_t kHeads = 3;
constexpr int64_t kHeadDim = 8;
constexpr int64_t kModel = kHeads * kHeadDim;
constexpr int64_t kCols = 3 * kModel;
constexpr float kBase = 10000.0f;

int g_failures = 0;

std::string sci(double v) {
  std::ostringstream oss;
  oss << std::scientific << v;
  return oss.str();
}

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

// A [B, S, 3D] qkv tensor in backend memory with host upload/download.
class DeviceQkv {
public:
  explicit DeviceQkv(DeviceBackend &backend) : backend_(backend) {
    data_ = backend_.alloc(bytes(), 64);
  }
  ~DeviceQkv() { backend_.free(data_); }

  uint64_t bytes() const {
    return static_cast<uint64_t>(kBatch * kSeq * kCols) * sizeof(float);
  }
  TensorView view() const {
    return TensorView(backend_.device(), DType::F32, data_,
                      Shape{kBatch, kSeq, kCols});
  }
  void upload(const std::vector<float> &host) {
    backend_.copy_host2device(data_, host.data(), bytes());
  }
  std::vector<float> download() const {
    std::vector<float> host(static_cast<size_t>(kBatch * kSeq * kCols));
    backend_.copy_device2host(host.data(), data_, bytes());
    return host;
  }

private:
  DeviceBackend &backend_;
  void *data_ = nullptr;
};

size_t at(int64_t b, int64_t s, int64_t c) {
  return static_cast<size_t>((b * kSeq + s) * kCols + c);
}

void rotate_qk(const Ops &ops, DeviceQkv &qkv, bool inverse) {
  TensorView full = qkv.view();
  TensorView q = full.subcols(0, kModel);
  TensorView k = full.subcols(kModel, kModel);
  ops.rotary_embedding_inplace(q, kHeads, kBase, inverse);
  ops.rotary_embedding_inplace(k, kHeads, kBase, inverse);
}

std::vector<float> random_qkv(uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> host(static_cast<size_t>(kBatch * kSeq * kCols));
  for (float &v : host) {
    v = dist(rng);
  }
  return host;
}

std::vector<float> reference_rotate(const std::vector<float> &in) {
  std::vector<float> out = in;
  const int64_t half = kHeadDim / 2;
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t s = 0; s < kSeq; ++s) {
      for (int64_t part = 0; part < 2; ++part) { // Q, K
        for (int64_t h = 0; h < kHeads; ++h) {
          for (int64_t i = 0; i < half; ++i) {
            const double inv_freq =
                std::pow(static_cast<double>(kBase),
                         -2.0 * static_cast<double>(i) / kHeadDim);
            const double angle = static_cast<double>(s) * inv_freq;
            const int64_t c1 = part * kModel + h * kHeadDim + i;
            const int64_t c2 = c1 + half;
            const double x1 = in[at(b, s, c1)];
            const double x2 = in[at(b, s, c2)];
            out[at(b, s, c1)] =
                static_cast<float>(x1 * std::cos(angle) - x2 * std::sin(angle));
            out[at(b, s, c2)] =
                static_cast<float>(x1 * std::sin(angle) + x2 * std::cos(angle));
          }
        }
      }
    }
  }
  return out;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

double dot_qk(const std::vector<float> &a, const std::vector<float> &b) {
  double sum = 0.0;
  for (int64_t bi = 0; bi < kBatch; ++bi) {
    for (int64_t s = 0; s < kSeq; ++s) {
      for (int64_t c = 0; c < 2 * kModel; ++c) {
        sum += static_cast<double>(a[at(bi, s, c)]) * b[at(bi, s, c)];
      }
    }
  }
  return sum;
}

void test_matches_reference(const Ops &ops, DeviceBackend &backend) {
  const std::vector<float> input = random_qkv(1);
  DeviceQkv qkv(backend);
  qkv.upload(input);
  rotate_qk(ops, qkv, /*inverse=*/false);
  const float diff = max_abs_diff(qkv.download(), reference_rotate(input));
  check(diff < 1e-4f,
        "matches host reference (max abs diff " + sci(diff) + ")");
}

void test_inverse_roundtrip(const Ops &ops, DeviceBackend &backend) {
  const std::vector<float> input = random_qkv(2);
  DeviceQkv qkv(backend);
  qkv.upload(input);
  rotate_qk(ops, qkv, /*inverse=*/false);
  rotate_qk(ops, qkv, /*inverse=*/true);
  const float diff = max_abs_diff(qkv.download(), input);
  check(diff < 1e-5f,
        "inverse(forward(x)) == x (max abs diff " + sci(diff) + ")");
}

// <R x, y> == <x, R^T y> with R^T applied by the inverse rotation.
void test_inverse_is_adjoint(const Ops &ops, DeviceBackend &backend) {
  const std::vector<float> x = random_qkv(3);
  const std::vector<float> y = random_qkv(4);
  DeviceQkv rx(backend);
  DeviceQkv rty(backend);
  rx.upload(x);
  rty.upload(y);
  rotate_qk(ops, rx, /*inverse=*/false);
  rotate_qk(ops, rty, /*inverse=*/true);
  const double lhs = dot_qk(rx.download(), y);
  const double rhs = dot_qk(x, rty.download());
  const double rel = std::fabs(lhs - rhs) / std::max(1.0, std::fabs(lhs));
  check(rel < 1e-5, "inverse is the adjoint of forward (rel diff " +
                        sci(rel) + ")");
}

// With the same q at every position and the same k at every position, the
// rotated score q_i . k_j must depend only on i - j.
void test_relative_position(const Ops &ops, DeviceBackend &backend) {
  const std::vector<float> seed = random_qkv(5);
  std::vector<float> input = seed;
  for (int64_t b = 0; b < kBatch; ++b) {
    for (int64_t s = 0; s < kSeq; ++s) {
      for (int64_t c = 0; c < kCols; ++c) {
        input[at(b, s, c)] = seed[at(0, 0, c)];
      }
    }
  }
  DeviceQkv qkv(backend);
  qkv.upload(input);
  rotate_qk(ops, qkv, /*inverse=*/false);
  const std::vector<float> out = qkv.download();

  auto score = [&](int64_t h, int64_t i, int64_t j) {
    double sum = 0.0;
    for (int64_t d = 0; d < kHeadDim; ++d) {
      sum += static_cast<double>(out[at(1, i, h * kHeadDim + d)]) *
             out[at(1, j, kModel + h * kHeadDim + d)];
    }
    return sum;
  };
  double worst = 0.0;
  for (int64_t h = 0; h < kHeads; ++h) {
    for (int64_t i = 0; i < kSeq; ++i) {
      for (int64_t j = 0; j <= i; ++j) {
        worst = std::max(worst, std::fabs(score(h, i, j) - score(h, i - j, 0)));
      }
    }
  }
  check(worst < 1e-4, "score(i, j) == score(i - j, 0) (max diff " +
                          sci(worst) + ")");
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
  Ops ops(*backend);

  std::cout << "RoPE op tests on " << argv[1] << "\n";
  test_matches_reference(ops, *backend);
  test_inverse_roundtrip(ops, *backend);
  test_inverse_is_adjoint(ops, *backend);
  test_relative_position(ops, *backend);

  std::cout << (g_failures == 0 ? "All RoPE op tests passed.\n"
                                : "RoPE op tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
