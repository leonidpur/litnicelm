#include "tensor.hpp"

#include <limits>
#include <sstream>

std::string TensorView::debug_string() const {
  std::ostringstream oss;
  oss << "TensorView(" << device_name(dev_) << "," << dtype_name(dt_) << ", "
      << shape_.r << "x" << shape_.c << ", bytes=" << bytes()
      << ", stride_r=" << stride_r_bytes_ << ", stride_c=" << stride_c_bytes_
      << ", contiguous=" << (is_contiguous_row_major() ? "yes" : "no") << ")";
  return oss.str();
}

Tensor Tensor::make_cpu(DType dt, Shape2D shape) {
  Tensor t;
  t.owns_ = true;

  if (shape.r < 0 || shape.c < 0) {
    throw std::runtime_error("Tensor::make_cpu negative shape");
  }

  const uint64_t b = nbytes(shape, dt);
  if (b > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("Tensor::make_cpu size too large for host");
  }

  t.storage_.resize(static_cast<size_t>(b), 0);
  t.view_ = TensorView(Device::CPU, dt, t.storage_.data(), shape);
  return t;
}

Tensor Tensor::wrap(Device dev, DType dt, void *data, Shape2D shape,
                    int64_t stride_c_bytes) {
  if (!data) {
    throw std::runtime_error("Tensor::wrap null data");
  }
  Tensor t;
  t.owns_ = false;
  t.view_ = TensorView(dev, dt, data, shape, stride_c_bytes);
  return t;
}

void Tensor::zero_() {
  if (view_.data() == nullptr) {
    return;
  }
  if (device() != Device::CPU) {
    throw std::runtime_error("Tensor::zero_ GPU not implemented yet");
  }
  std::memset(view_.data(), 0, static_cast<size_t>(bytes()));
}
