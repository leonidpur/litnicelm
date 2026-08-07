#include "tensor.hpp"

#include <limits>
#include <sstream>

std::string TensorView::debug_string() const {
  std::ostringstream oss;
  oss << "TensorView(" << device_name(dev_) << "," << dtype_name(dt_)
      << ", shape=[";
  for (size_t i = 0; i < shape_.rank(); ++i) {
    if (i != 0) {
      oss << "x";
    }
    oss << shape_.dim(i);
  }
  oss << "], bytes=" << bytes() << ", strides=[";
  for (size_t i = 0; i < shape_.rank(); ++i) {
    if (i != 0) {
      oss << "/";
    }
    oss << stride_bytes(i);
  }
  oss << "], contiguous=" << (is_contiguous_row_major() ? "yes" : "no")
      << ")";
  return oss.str();
}

Tensor Tensor::make_cpu(DType dt, Shape shape) {
  Tensor t;
  t.owns_ = true;

  for (size_t i = 0; i < shape.rank(); ++i) {
    if (shape.dim(i) < 0) {
      throw std::runtime_error("Tensor::make_cpu negative shape");
    }
  }

  const uint64_t b = nbytes(shape, dt);
  if (b > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::overflow_error("Tensor::make_cpu size too large for host");
  }

  t.storage_.resize(static_cast<size_t>(b), 0);
  t.view_ = TensorView(Device::CPU, dt, t.storage_.data(), shape);
  return t;
}

Tensor Tensor::wrap(Device dev, DType dt, void *data, Shape shape,
                    int64_t stride_c_bytes) {
  if (!data) {
    throw std::runtime_error("Tensor::wrap null data");
  }
  Tensor t;
  t.owns_ = false;
  t.view_ = TensorView(dev, dt, data, shape, stride_c_bytes, 0);
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
