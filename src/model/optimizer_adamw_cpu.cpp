#include "optimizer_adamw_cpu.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

static void require(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::runtime_error("OptimizerAdamWCPU: " + msg);
  }
}

static void require_f32_cpu_contig(const TensorView &t, const char *who) {
  require(t.device() == Device::CPU, std::string(who) + " CPU only");
  require(t.dtype() == DType::F32, std::string(who) + " F32 only");
  require(t.is_contiguous(), std::string(who) + " requires contiguous");
}

static float *fptr(TensorView &t) { return reinterpret_cast<float *>(t.data()); }
static const float *fptr(const TensorView &t) {
  return reinterpret_cast<const float *>(t.data());
}

void OptimizerAdamWCPU::step(const TrainingConfig &tc, TensorView &params,
                             const TensorView &grads, TensorView &m,
                             TensorView &v, uint64_t step,
                             bool apply_weight_decay) const {
  require(step >= 1, "step must be >= 1");

  require_f32_cpu_contig(params, "step(params)");
  require_f32_cpu_contig(grads, "step(grads)");
  require_f32_cpu_contig(m, "step(m)");
  require_f32_cpu_contig(v, "step(v)");

  require(params.rank() == grads.rank() && params.numel() == grads.numel(),
          "params/grads shape mismatch");
  require(params.rank() == m.rank() && params.numel() == m.numel(),
          "params/m shape mismatch");
  require(params.rank() == v.rank() && params.numel() == v.numel(),
          "params/v shape mismatch");

  const int64_t n = static_cast<int64_t>(params.numel());
  require(n >= 0, "invalid element count");

  const float lr = tc.learning_rate;
  const float beta1 = tc.beta1;
  const float beta2 = tc.beta2;
  const float wd = tc.weight_decay;
  const float eps = 1e-8f;

  require(lr > 0.0f, "learning_rate must be > 0");
  require(beta1 >= 0.0f && beta1 < 1.0f, "beta1 must be in [0,1)");
  require(beta2 >= 0.0f && beta2 < 1.0f, "beta2 must be in [0,1)");

  const double t = static_cast<double>(step);
  const float b1_corr =
      1.0f - static_cast<float>(std::pow(static_cast<double>(beta1), t));
  const float b2_corr =
      1.0f - static_cast<float>(std::pow(static_cast<double>(beta2), t));
  require(b1_corr > 0.0f && b2_corr > 0.0f,
          "invalid bias correction (check betas)");

  float *p = fptr(params);
  const float *g = fptr(grads);
  float *m1 = fptr(m);
  float *m2 = fptr(v);

  for (int64_t i = 0; i < n; ++i) {
    const float grad = g[i];

    m1[i] = beta1 * m1[i] + (1.0f - beta1) * grad;
    m2[i] = beta2 * m2[i] + (1.0f - beta2) * grad * grad;

    const float mhat = m1[i] / b1_corr;
    const float vhat = m2[i] / b2_corr;

    const float adam = mhat / (std::sqrt(vhat) + eps);
    const float decay = apply_weight_decay ? (wd * p[i]) : 0.0f;

    p[i] -= lr * (adam + decay);
  }
}
