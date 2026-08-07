#include "ops_cpu.hpp"

#include <utils/assert.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

static float load_f32(const TensorView &t, int64_t r, int64_t c) {
  const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
  const uint8_t *p = base + r * t.stride_r_bytes() + c * t.stride_c_bytes();
  float out;
  std::memcpy(&out, p, sizeof(float));
  return out;
}

static void store_f32(const TensorView &t, int64_t r, int64_t c, float v) {
  uint8_t *base = reinterpret_cast<uint8_t *>(t.data());
  uint8_t *p = base + r * t.stride_r_bytes() + c * t.stride_c_bytes();
  std::memcpy(p, &v, sizeof(float));
}

static int32_t load_i32(const TensorView &t, int64_t r, int64_t c) {
  const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
  const uint8_t *p = base + r * t.stride_r_bytes() + c * t.stride_c_bytes();
  int32_t out;
  std::memcpy(&out, p, sizeof(int32_t));
  return out;
}

static int64_t load_index(const TensorView &t, int64_t r, int64_t c) {
  if (t.dtype() == DType::I32) {
    return static_cast<int64_t>(load_i32(t, r, c));
  }
  if (t.dtype() == DType::F32) {
    return static_cast<int64_t>(load_f32(t, r, c));
  }
  throw std::runtime_error("OpsCPU: unsupported index dtype");
}

void OpsCPU::copy(const TensorView &src, TensorView &dst) const {
  REQUIRE_DEBUG((src.shape().r == dst.shape().r && src.shape().c == dst.shape().c),
                [&]() { return std::string("OpsCPU: ") + "copy shape mismatch"; });

  const int64_t R = src.shape().r;
  const int64_t C = src.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(dst, r, c, load_f32(src, r, c));
    }
  }
}

void OpsCPU::fill(TensorView &t, float v) const {
  const int64_t R = t.shape().r;
  const int64_t C = t.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(t, r, c, v);
    }
  }
}

void OpsCPU::add(const TensorView &a, const TensorView &b, TensorView &out) const {
  REQUIRE_DEBUG((a.shape().r == b.shape().r && a.shape().c == b.shape().c),
                [&]() { return std::string("OpsCPU: ") + "add shape mismatch"; });
  REQUIRE_DEBUG((out.shape().r == a.shape().r && out.shape().c == a.shape().c),
                [&]() { return std::string("OpsCPU: ") + "add out shape mismatch"; });

  const int64_t R = a.shape().r;
  const int64_t C = a.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(out, r, c, load_f32(a, r, c) + load_f32(b, r, c));
    }
  }
}

void OpsCPU::add_inplace(TensorView &a, const TensorView &b) const {
  REQUIRE_DEBUG((a.shape().r == b.shape().r && a.shape().c == b.shape().c),
                [&]() { return std::string("OpsCPU: ") + "add_inplace shape mismatch"; });

  const int64_t R = a.shape().r;
  const int64_t C = a.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(a, r, c, load_f32(a, r, c) + load_f32(b, r, c));
    }
  }
}

void OpsCPU::add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                              TensorView &out) const {
  REQUIRE_DEBUG((bias_1xC.shape().r == 1),
                [&]() { return std::string("OpsCPU: ") + "bias must be [1,C]"; });
  REQUIRE_DEBUG((bias_1xC.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "bias C mismatch"; });
  REQUIRE_DEBUG((out.shape().r == x.shape().r && out.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "out shape mismatch"; });

  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(out, r, c, load_f32(x, r, c) + load_f32(bias_1xC, 0, c));
    }
  }
}

void OpsCPU::mul_scalar(const TensorView &x, float s, TensorView &out) const {
  REQUIRE_DEBUG((out.shape().r == x.shape().r && out.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "shape mismatch"; });

  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(out, r, c, load_f32(x, r, c) * s);
    }
  }
}

void OpsCPU::relu(const TensorView &x, TensorView &out) const {
  REQUIRE_DEBUG((out.shape().r == x.shape().r && out.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "shape mismatch"; });

  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      const float v = load_f32(x, r, c);
      store_f32(out, r, c, v > 0.0f ? v : 0.0f);
    }
  }
}

void OpsCPU::matmul(const TensorView &a, const TensorView &b, TensorView &out) const {
  const int64_t R = a.shape().r;
  const int64_t K = a.shape().c;
  REQUIRE_DEBUG((b.shape().r == K),
                [&]() { return std::string("OpsCPU: ") + "inner dim mismatch"; });
  const int64_t C = b.shape().c;
  REQUIRE_DEBUG((out.shape().r == R && out.shape().c == C),
                [&]() { return std::string("OpsCPU: ") + "out shape mismatch"; });

  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += load_f32(a, r, k) * load_f32(b, k, c);
      }
      store_f32(out, r, c, acc);
    }
  }
}

void OpsCPU::matmul_transposed(const TensorView &a, const TensorView &b,
                               TensorView &out) const {
  const int64_t R = a.shape().r;
  const int64_t K = a.shape().c;
  REQUIRE_DEBUG((b.shape().c == K),
                [&]() { return std::string("OpsCPU: ") + "inner dim mismatch"; });
  const int64_t C = b.shape().r;
  REQUIRE_DEBUG((out.shape().r == R && out.shape().c == C),
                [&]() { return std::string("OpsCPU: ") + "out shape mismatch"; });

  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += load_f32(a, r, k) * load_f32(b, c, k);
      }
      store_f32(out, r, c, acc);
    }
  }
}

void OpsCPU::transpose(const TensorView &x, TensorView &out) const {
  REQUIRE_DEBUG((out.shape().r == x.shape().c && out.shape().c == x.shape().r),
                [&]() { return std::string("OpsCPU: ") + "shape mismatch"; });

  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      store_f32(out, c, r, load_f32(x, r, c));
    }
  }
}

void OpsCPU::layernorm(const TensorView &x, const TensorView &gamma_1xC,
                       const TensorView &beta_1xC, TensorView &out) const {
  REQUIRE_DEBUG((gamma_1xC.shape().r == 1 && gamma_1xC.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "gamma must be [1,C]"; });
  REQUIRE_DEBUG((beta_1xC.shape().r == 1 && beta_1xC.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "beta must be [1,C]"; });
  REQUIRE_DEBUG((out.shape().r == x.shape().r && out.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "out shape mismatch"; });

  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  const float eps = 1e-5f;
  for (int64_t r = 0; r < R; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      mean += load_f32(x, r, c);
    }
    mean /= static_cast<double>(C);

    double var = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double d = static_cast<double>(load_f32(x, r, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(C);
    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

    for (int64_t c = 0; c < C; ++c) {
      const float xn =
          (load_f32(x, r, c) - static_cast<float>(mean)) * inv_std;
      const float y =
          xn * load_f32(gamma_1xC, 0, c) + load_f32(beta_1xC, 0, c);
      store_f32(out, r, c, y);
    }
  }
}

void OpsCPU::embedding_lookup(const TensorView &table, const TensorView &ids,
                              TensorView &out) const {
  REQUIRE_DEBUG((ids.dtype() == DType::I32 || ids.dtype() == DType::F32),
                [&]() { return std::string("OpsCPU: ") + "embedding_lookup(ids) requires I32 or F32"; });
  REQUIRE_DEBUG((out.shape().c == table.shape().c),
                [&]() { return std::string("OpsCPU: ") + "out cols must equal table cols"; });
  REQUIRE_DEBUG((ids.shape().r == out.shape().r),
                [&]() { return std::string("OpsCPU: ") + "ids rows must equal out rows"; });
  REQUIRE_DEBUG((ids.shape().c == 1 || ids.shape().c == 0),
                [&]() { return std::string("OpsCPU: ") + "ids must be [T] or [T,1]"; });

  const int64_t V = table.shape().r;
  const int64_t D = table.shape().c;
  const int64_t T = out.shape().r;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t idx = load_index(ids, t, 0);
    REQUIRE_DEBUG((idx >= 0 && idx < V),
                  [&]() { return std::string("OpsCPU: ") + "embedding id out of range"; });
#ifdef COPY_BY_F32
    for (int64_t d = 0; d < D; ++d) {
      store_f32(out, t, d, load_f32(table, idx, d));
    }
#else
    REQUIRE_DEBUG((table.stride_c_bytes() == static_cast<int64_t>(sizeof(float)) &&
                   out.stride_c_bytes() == static_cast<int64_t>(sizeof(float))),
                  [&]() { return std::string("OpsCPU: ") + "embedding_lookup memcpy path requires contiguous columns"; });
    const uint8_t *src = reinterpret_cast<const uint8_t *>(table.data()) +
                         idx * table.stride_r_bytes();
    uint8_t *dst =
        reinterpret_cast<uint8_t *>(out.data()) + t * out.stride_r_bytes();
    std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
#endif
  }
}

void OpsCPU::cross_entropy_mean(const TensorView &logits,
                                const TensorView &targets,
                                TensorView &out_loss) const {
  REQUIRE_DEBUG((targets.dtype() == DType::I32 || targets.dtype() == DType::F32),
                [&]() { return std::string("OpsCPU: ") + "cross_entropy_mean(targets) requires I32 or F32"; });
  REQUIRE_DEBUG((out_loss.shape().r == 1 && out_loss.shape().c == 1),
                [&]() { return std::string("OpsCPU: ") + "out_loss must be [1,1]"; });
  REQUIRE_DEBUG((targets.shape().r == logits.shape().r),
                [&]() { return std::string("OpsCPU: ") + "targets rows mismatch"; });
  REQUIRE_DEBUG((targets.shape().c == 1 || targets.shape().c == 0),
                [&]() { return std::string("OpsCPU: ") + "targets must be [T] or [T,1]"; });

  const int64_t T = logits.shape().r;
  const int64_t V = logits.shape().c;
  double sum = 0.0;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t y = load_index(targets, t, 0);
    REQUIRE_DEBUG((y >= 0 && y < V),
                  [&]() { return std::string("OpsCPU: ") + "target out of range"; });

    float m = load_f32(logits, t, 0);
    for (int64_t c = 1; c < V; ++c) {
      m = std::max(m, load_f32(logits, t, c));
    }
    double lse = 0.0;
    for (int64_t c = 0; c < V; ++c) {
      lse += std::exp(static_cast<double>(load_f32(logits, t, c) - m));
    }
    const double log_denom = static_cast<double>(m) + std::log(lse);
    const double nll = log_denom - static_cast<double>(load_f32(logits, t, y));
    sum += nll;
  }
  store_f32(out_loss, 0, 0, static_cast<float>(sum / static_cast<double>(T)));
}

float OpsCPU::read_scalar_f32(const TensorView &x) const {
  REQUIRE_DEBUG((x.shape().r == 1 && x.shape().c == 1),
                [&]() { return std::string("OpsCPU: ") + "x must be [1,1]"; });
  return load_f32(x, 0, 0);
}

void OpsCPU::backward_from_logits_targets(TensorView &logits,
                                          const TensorView &targets) const {
  REQUIRE_DEBUG((targets.dtype() == DType::I32 || targets.dtype() == DType::F32),
                [&]() { return std::string("OpsCPU: ") + "backward_from_logits_targets(targets) requires I32 or F32"; });
  REQUIRE_DEBUG((targets.shape().r == logits.shape().r),
                [&]() { return std::string("OpsCPU: ") + "targets rows mismatch"; });
  REQUIRE_DEBUG((targets.shape().c == 1 || targets.shape().c == 0),
                [&]() { return std::string("OpsCPU: ") + "targets must be [T] or [T,1]"; });

  const int64_t T = logits.shape().r;
  const int64_t V = logits.shape().c;
  REQUIRE_DEBUG((T > 0 && V > 0),
                [&]() { return std::string("OpsCPU: ") + "invalid logits shape"; });

  const float inv_T = 1.0f / static_cast<float>(T);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t y = load_index(targets, t, 0);
    REQUIRE_DEBUG((y >= 0 && y < V),
                  [&]() { return std::string("OpsCPU: ") + "target out of range"; });

    float m = load_f32(logits, t, 0);
    for (int64_t c = 1; c < V; ++c) {
      m = std::max(m, load_f32(logits, t, c));
    }

    double sum = 0.0;
    for (int64_t c = 0; c < V; ++c) {
      sum += std::exp(static_cast<double>(load_f32(logits, t, c) - m));
    }
    REQUIRE_DEBUG((sum > 0.0),
                  [&]() { return std::string("OpsCPU: ") + "softmax sum <= 0"; });

    for (int64_t c = 0; c < V; ++c) {
      const float p = static_cast<float>(
          std::exp(static_cast<double>(load_f32(logits, t, c) - m)) / sum);
      float g = p;
      if (c == y) {
        g -= 1.0f;
      }
      store_f32(logits, t, c, g * inv_T);
    }
  }
}

void OpsCPU::softmax_rows(const TensorView &x, TensorView &out) const {
  REQUIRE_DEBUG((out.shape().r == x.shape().r && out.shape().c == x.shape().c),
                [&]() { return std::string("OpsCPU: ") + "shape mismatch"; });

  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;

  for (int64_t r = 0; r < R; ++r) {
    float m = load_f32(x, r, 0);
    for (int64_t c = 1; c < C; ++c) {
      m = std::max(m, load_f32(x, r, c));
    }

    double sum = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const float e = std::exp(load_f32(x, r, c) - m);
      store_f32(out, r, c, e);
      sum += static_cast<double>(e);
    }
    REQUIRE_DEBUG((sum > 0.0),
                  [&]() { return std::string("OpsCPU: ") + "softmax sum <= 0"; });

    const float inv = static_cast<float>(1.0 / sum);
    for (int64_t c = 0; c < C; ++c) {
      store_f32(out, r, c, load_f32(out, r, c) * inv);
    }
  }
}

void OpsCPU::apply_causal_mask_inplace(TensorView &scores, float neg_inf) const {
  REQUIRE_DEBUG((scores.shape().r == scores.shape().c),
                [&]() { return std::string("OpsCPU: ") + "scores must be [T,T]"; });

  const int64_t T = scores.shape().r;
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = i + 1; j < T; ++j) {
      store_f32(scores, i, j, neg_inf);
    }
  }
}

#undef require
