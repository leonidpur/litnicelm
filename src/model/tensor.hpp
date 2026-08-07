#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <types.hpp>
#include <utility>
#include <vector>

enum class DType : uint8_t {
  F32 = 0,
  I32 = 1,
};

struct Shape2D {
  union {
    int64_t r;
    int64_t rows;
  };
  union {
    int64_t c;
    int64_t cols;
  };
};

using Shape = Shape2D;

inline constexpr size_t dtype_size(DType dt) {
  switch (dt) {
  case DType::F32:
    return 4;
  case DType::I32:
    return 4;
  default:
    return 0;
  }
}

inline constexpr const char *dtype_name(DType dt) {
  switch (dt) {
  case DType::F32:
    return "f32";
  case DType::I32:
    return "i32";
  default:
    return "unknown";
  }
}

inline constexpr const char *device_name(Device d) {
  switch (d) {
  case Device::CPU:
    return "cpu";
  case Device::GPU:
    return "gpu";
  default:
    return "unknown";
  }
}

inline uint64_t numel(Shape2D s) {
  if (s.r < 0 || s.c < 0) {
    throw std::runtime_error("Shape2D has negative dimension");
  }
  return static_cast<uint64_t>(s.r) * static_cast<uint64_t>(s.c);
}

inline uint64_t nbytes(Shape2D s, DType dt) {
  const uint64_t elems = numel(s);
  const uint64_t esz = static_cast<uint64_t>(dtype_size(dt));
  if (esz == 0) {
    throw std::runtime_error("Unknown dtype");
  }
  if (elems != 0 && esz > (UINT64_MAX / elems)) {
    throw std::overflow_error("nbytes overflow");
  }
  return elems * esz;
}

class TensorView {
public:
  TensorView() = default;

  TensorView(Device dev, DType dt, void *data, Shape2D shape,
             int64_t stride_c_bytes = 0, int64_t stride_r_bytes = 0)
      : dev_(dev), dt_(dt), data_(data), shape_(shape) {
    const int64_t elem = static_cast<int64_t>(dtype_size(dt_));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
    stride_c_bytes_ = (stride_c_bytes == 0) ? elem : stride_c_bytes;
    stride_r_bytes_ = (stride_r_bytes == 0)
                          ? static_cast<int64_t>(shape_.c) * stride_c_bytes_
                          : stride_r_bytes;
  }

  Device device() const { return dev_; }
  DType dtype() const { return dt_; }
  Shape2D shape() const { return shape_; }

  int64_t stride_r_bytes() const { return stride_r_bytes_; }
  int64_t stride_c_bytes() const { return stride_c_bytes_; }

  void *data() const { return data_; }

  uint64_t bytes() const { return nbytes(shape_, dt_); }

  bool is_contiguous_row_major() const {
    const int64_t elem = static_cast<int64_t>(dtype_size(dt_));
    return stride_c_bytes_ == elem && stride_r_bytes_ == shape_.c * elem;
  }

  bool is_contiguous() const { return is_contiguous_row_major(); }

  TensorView subcols(int64_t col_offset, int64_t sub_cols) const {
    if (col_offset < 0 || sub_cols < 0 || col_offset + sub_cols > shape_.c) {
      throw std::out_of_range("TensorView::subcols out of bounds");
    }
    uint8_t *base = reinterpret_cast<uint8_t *>(data_);
    void *sub_ptr = base + col_offset * stride_c_bytes_;
    return TensorView(dev_, dt_, sub_ptr, Shape2D{shape_.r, sub_cols},
                      stride_c_bytes_, stride_r_bytes_);
  }

  TensorView subrows(int64_t row_offset, int64_t sub_rows) const {
    if (row_offset < 0 || sub_rows < 0 || row_offset + sub_rows > shape_.r) {
      throw std::out_of_range("TensorView::subrows out of bounds");
    }
    uint8_t *base = reinterpret_cast<uint8_t *>(data_);
    void *sub_ptr = base + row_offset * stride_r_bytes_;
    return TensorView(dev_, dt_, sub_ptr, Shape2D{sub_rows, shape_.c},
                      stride_c_bytes_, stride_r_bytes_);
  }

  float *f32() const {
    if (dt_ != DType::F32) {
      throw std::runtime_error("TensorView: not f32");
    }
    return reinterpret_cast<float *>(data_);
  }

  float at_f32(int64_t r, int64_t c) const {
    if (dev_ != Device::CPU) {
      throw std::runtime_error("TensorView::at_f32 CPU only");
    }
    if (dt_ != DType::F32) {
      throw std::runtime_error("TensorView::at_f32 requires f32");
    }
    if (r < 0 || c < 0 || r >= shape_.r || c >= shape_.c) {
      throw std::out_of_range("at_f32 oob");
    }
    const uint8_t *base = reinterpret_cast<const uint8_t *>(data_);
    const uint8_t *p = base + r * stride_r_bytes_ + c * stride_c_bytes_;
    float out;
    std::memcpy(&out, p, sizeof(float));
    return out;
  }

  void set_f32(int64_t r, int64_t c, float v) const {
    if (dev_ != Device::CPU) {
      throw std::runtime_error("TensorView::set_f32 CPU only");
    }
    if (dt_ != DType::F32) {
      throw std::runtime_error("TensorView::set_f32 requires f32");
    }
    if (r < 0 || c < 0 || r >= shape_.r || c >= shape_.c) {
      throw std::out_of_range("set_f32 oob");
    }
    uint8_t *base = reinterpret_cast<uint8_t *>(data_);
    uint8_t *p = base + r * stride_r_bytes_ + c * stride_c_bytes_;
    std::memcpy(p, &v, sizeof(float));
  }

  std::string debug_string() const;

private:
  Device dev_ = Device::CPU;
  DType dt_ = DType::F32;
  void *data_ = nullptr;
  Shape2D shape_{0, 0};
  int64_t stride_r_bytes_ = 0;
  int64_t stride_c_bytes_ = 0;
};

class Tensor {
public:
  Tensor() = default;

  static Tensor make_cpu(DType dt, Shape2D shape);
  static Tensor wrap(Device dev, DType dt, void *data, Shape2D shape,
                     int64_t stride_c_bytes = 0);

  Device device() const { return view_.device(); }
  DType dtype() const { return view_.dtype(); }
  Shape2D shape() const { return view_.shape(); }
  uint64_t bytes() const { return view_.bytes(); }

  TensorView view() const { return view_; }
  TensorView &view_mut() { return view_; }

  float *f32() { return view_.f32(); }
  const float *f32() const { return view_.f32(); }

  bool owns_memory() const { return owns_; }

  void zero_();

private:
  TensorView view_{};
  std::vector<uint8_t> storage_;
  bool owns_ = false;
};
