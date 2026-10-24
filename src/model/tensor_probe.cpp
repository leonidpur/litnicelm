#include "tensor_probe.hpp"

#include <litnicelm_utils/hash_utils.hpp>
#include <litnicelm_utils/log.hpp>
#include "model_events.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace {

struct ProbeStats {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
  uint32_t hash = hash_utils::kFnv1a32Offset;
};

ProbeStats probe_stats_f32(const TensorView &t) {
  if (t.device() != Device::CPU || t.dtype() != DType::F32) {
    return {};
  }
  ProbeStats s;
  const int64_t R = t.shape().dim(0);
  const int64_t C = t.shape().dim(1);
  double sum = 0.0;
  double sum_sq = 0.0;
  bool init = false;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      const float v = t.at_f32(r, c);
      s.hash = hash_utils::fnv1a32(s.hash, &v, sizeof(float));
      const double x = static_cast<double>(v);
      if (!init) {
        s.min = s.max = x;
        init = true;
      } else {
        if (x < s.min) s.min = x;
        if (x > s.max) s.max = x;
      }
      sum += x;
      sum_sq += x * x;
    }
  }
  const double n = static_cast<double>(R * C);
  if (n > 0.0) {
    s.mean = sum / n;
    double var = (sum_sq / n) - (s.mean * s.mean);
    if (var < 0.0) {
      var = 0.0;
    }
    s.stddev = std::sqrt(var);
  }
  return s;
}

LogFields stats_fields(const std::string &prefix, const ProbeStats &s) {
  return LogFields()
      .f64(prefix + "mean", s.mean)
      .f64(prefix + "std", s.stddev)
      .f64(prefix + "min", s.min)
      .f64(prefix + "max", s.max);
}

} // namespace

namespace TensorProbe {

void log_tensor(const std::string &group, const std::string &name,
                const TensorView &tensor) {
  if (tensor.device() != Device::CPU || tensor.dtype() != DType::F32) {
    return;
  }
  const ProbeStats s = probe_stats_f32(tensor);
  std::ostringstream oss;
  oss << group << "." << name
      << " shape=" << format_shape(tensor.shape_nd())
      << std::fixed << std::setprecision(6)
      << " mean=" << s.mean
      << " std=" << s.stddev
      << " min=" << s.min
      << " max=" << s.max
      << " hash=0x" << std::hex << s.hash << std::dec;
  logger().log(model_events::kProbeTensor, oss.str(),
               stats_fields("", s).str("tensor", group + "." + name));
}

void log_loss(const TensorView &loss_scalar, const TensorView &logits,
              const TensorView &targets) {
  if (loss_scalar.device() != Device::CPU || loss_scalar.dtype() != DType::F32 ||
      logits.device() != Device::CPU || logits.dtype() != DType::F32) {
    return;
  }
  const ProbeStats logits_stats = probe_stats_f32(logits);
  const float scalar = loss_scalar.at_f32(0, 0);
  std::ostringstream oss;
  oss << "scalar=" << std::fixed << std::setprecision(6) << scalar
      << " logits_shape=" << format_shape(logits.shape_nd())
      << " logits_mean=" << logits_stats.mean
      << " logits_std=" << logits_stats.stddev
      << " logits_min=" << logits_stats.min
      << " logits_max=" << logits_stats.max
      << " logits_hash=0x" << std::hex << logits_stats.hash << std::dec
      << " targets_shape=" << format_shape(targets.shape_nd());
  logger().log(model_events::kProbeLoss, oss.str(),
               stats_fields("logits_", logits_stats).f64("loss", scalar));
}

std::string format_shape(const Shape &shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << shape.dim(i);
  }
  oss << "]";
  return oss.str();
}

} // namespace TensorProbe
