#pragma once

#include <cstdint>
#include <array>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <types.hpp>
#include <utility>
#include <vector>

enum class DType : uint8_t {
  F32 = 0,
  I32 = 1,
};

constexpr size_t kMaxTensorRank = 6;

class Shape {
public:
  Shape() = default;

  Shape(std::initializer_list<int64_t> dims) {
    init_(dims.begin(), dims.size());
  }

  explicit Shape(const std::vector<int64_t> &dims) {
    init_(dims.begin(), dims.size());
  }

private:
  template <typename Iter>
  void init_(Iter begin, size_t size) {
    if (size > kMaxTensorRank) {
      throw std::runtime_error("Shape rank exceeds kMaxTensorRank");
    }
    rank_ = static_cast<uint8_t>(size);
    size_t i = 0;
    for (Iter it = begin; i < size; ++it) {
      dims_[i++] = *it;
    }
  }

public:
  static Shape scalar() { return Shape{}; }

  size_t rank() const { return rank_; }

  int64_t dim(size_t axis) const {
    if (axis >= rank_) {
      throw std::out_of_range("Shape::dim axis out of range");
    }
    return dims_[axis];
  }

  uint64_t numel() const {
    uint64_t out = 1;
    for (size_t i = 0; i < rank_; ++i) {
      if (dims_[i] < 0) {
        throw std::runtime_error("Shape has negative dimension");
      }
      out *= static_cast<uint64_t>(dims_[i]);
    }
    return out;
  }

  const std::array<int64_t, kMaxTensorRank> &dims() const { return dims_; }

  uint8_t rank_ = 0;
  std::array<int64_t, kMaxTensorRank> dims_{};
};

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

inline uint64_t numel(const Shape &s) { return s.numel(); }

inline uint64_t nbytes(const Shape &s, DType dt) {
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

  TensorView(Device dev, DType dt, void *data, Shape shape)
      : dev_(dev), dt_(dt), data_(data), shape_(shape) {
    const int64_t elem = static_cast<int64_t>(dtype_size(dt_));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
    strides_bytes_ = compute_contiguous_strides_(shape_, elem);
  }

  TensorView(Device dev, DType dt, void *data, Shape shape,
             int64_t stride_c_bytes, int64_t stride_r_bytes)
      : dev_(dev), dt_(dt), data_(data), shape_(shape) {
    const int64_t elem = static_cast<int64_t>(dtype_size(dt_));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
    strides_bytes_ = compute_contiguous_strides_(shape_, elem);
    if (shape_.rank() >= 1) {
      strides_bytes_[shape_.rank() - 1] =
          (stride_c_bytes == 0) ? elem : stride_c_bytes;
    }
    if (shape_.rank() >= 2) {
      strides_bytes_[shape_.rank() - 2] =
          (stride_r_bytes == 0)
              ? static_cast<int64_t>(shape_.dim(shape_.rank() - 1)) *
                    strides_bytes_[shape_.rank() - 1]
              : stride_r_bytes;
      for (size_t i = shape_.rank() - 2; i > 0; --i) {
        strides_bytes_[i - 1] = strides_bytes_[i] * shape_.dim(i);
      }
    }
  }

  TensorView(Device dev, DType dt, void *data, Shape shape,
             const std::array<int64_t, kMaxTensorRank> &strides_bytes)
      : dev_(dev), dt_(dt), data_(data), shape_(shape),
        strides_bytes_(strides_bytes) {
    const int64_t elem = static_cast<int64_t>(dtype_size(dt_));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
  }

  Device device() const { return dev_; }
  DType dtype() const { return dt_; }
  const Shape &shape() const { return shape_; }
  const Shape &logical_shape() const { return shape_; }
  size_t rank() const { return shape_.rank(); }
  int64_t dim(size_t axis) const { return shape_.dim(axis); }
  Shape shape_nd() const { return shape_; }

  int64_t stride_bytes(size_t axis) const {
    if (axis >= shape_.rank()) {
      throw std::out_of_range("TensorView::stride_bytes axis out of range");
    }
    return strides_bytes_[axis];
  }

  void *data() const { return data_; }

  uint64_t bytes() const { return nbytes(shape_, dt_); }
  uint64_t numel() const { return shape_.numel(); }

  bool is_contiguous_row_major() const {
    const int64_t elem = static_cast<int64_t>(dtype_size(dt_));
    return strides_bytes_ == compute_contiguous_strides_(shape_, elem);
  }

  bool is_contiguous() const { return is_contiguous_row_major(); }

  TensorView slice(size_t axis, int64_t start, int64_t len) const {
    if (axis >= shape_.rank()) {
      throw std::out_of_range("TensorView::slice axis out of range");
    }
    if (start < 0 || len < 0 || start + len > shape_.dim(axis)) {
      throw std::out_of_range("TensorView::slice bounds");
    }
    std::vector<int64_t> rebuilt;
    rebuilt.reserve(shape_.rank());
    for (size_t i = 0; i < shape_.rank(); ++i) {
      rebuilt.push_back((i == axis) ? len : shape_.dim(i));
    }
    Shape sliced(rebuilt);
    uint8_t *base = reinterpret_cast<uint8_t *>(data_);
    void *sub_ptr = base + start * strides_bytes_[axis];
    return TensorView(dev_, dt_, sub_ptr, sliced, strides_bytes_);
  }

  TensorView select(size_t axis, int64_t index) const {
    if (axis >= shape_.rank()) {
      throw std::out_of_range("TensorView::select axis out of range");
    }
    if (index < 0 || index >= shape_.dim(axis)) {
      throw std::out_of_range("TensorView::select bounds");
    }
    std::vector<int64_t> rebuilt;
    rebuilt.reserve(shape_.rank() > 0 ? shape_.rank() - 1 : 0);
    for (size_t i = 0; i < shape_.rank(); ++i) {
      if (i != axis) {
        rebuilt.push_back(shape_.dim(i));
      }
    }
    Shape selected(rebuilt);
    uint8_t *base = reinterpret_cast<uint8_t *>(data_);
    void *sub_ptr = base + index * strides_bytes_[axis];
    std::array<int64_t, kMaxTensorRank> selected_strides{};
    size_t out_axis = 0;
    for (size_t i = 0; i < shape_.rank(); ++i) {
      if (i != axis) {
        selected_strides[out_axis++] = strides_bytes_[i];
      }
    }
    return TensorView(dev_, dt_, sub_ptr, selected, selected_strides);
  }

  TensorView subcols(int64_t col_offset, int64_t sub_cols) const {
    if (shape_.rank() == 0) {
      throw std::out_of_range("TensorView::subcols requires rank >= 1");
    }
    const size_t axis = shape_.rank() - 1;
    if (col_offset < 0 || sub_cols < 0 ||
        col_offset + sub_cols > shape_.dim(axis)) {
      throw std::out_of_range("TensorView::subcols out of bounds");
    }
    return slice(axis, col_offset, sub_cols);
  }

  TensorView subrows(int64_t row_offset, int64_t sub_rows) const {
    if (shape_.rank() == 0) {
      throw std::out_of_range("TensorView::subrows requires rank >= 1");
    }
    if (row_offset < 0 || sub_rows < 0 || row_offset + sub_rows > shape_.dim(0)) {
      throw std::out_of_range("TensorView::subrows out of bounds");
    }
    return slice(0, row_offset, sub_rows);
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
    if (shape_.rank() != 2 || r < 0 || c < 0 || r >= shape_.dim(0) ||
        c >= shape_.dim(1)) {
      throw std::out_of_range("at_f32 oob");
    }
    const uint8_t *base = reinterpret_cast<const uint8_t *>(data_);
    const uint8_t *p =
        base + r * stride_bytes(0) + c * stride_bytes(1);
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
    if (shape_.rank() != 2 || r < 0 || c < 0 || r >= shape_.dim(0) ||
        c >= shape_.dim(1)) {
      throw std::out_of_range("set_f32 oob");
    }
    uint8_t *base = reinterpret_cast<uint8_t *>(data_);
    uint8_t *p = base + r * stride_bytes(0) + c * stride_bytes(1);
    std::memcpy(p, &v, sizeof(float));
  }

  std::string debug_string() const;

private:
  static std::array<int64_t, kMaxTensorRank>
  compute_contiguous_strides_(const Shape &shape, int64_t elem_bytes) {
    std::array<int64_t, kMaxTensorRank> strides{};
    int64_t stride = elem_bytes;
    for (size_t i = shape.rank(); i > 0; --i) {
      strides[i - 1] = stride;
      stride *= shape.dim(i - 1);
    }
    return strides;
  }

  Device dev_ = Device::CPU;
  DType dt_ = DType::F32;
  void *data_ = nullptr;
  Shape shape_{};
  std::array<int64_t, kMaxTensorRank> strides_bytes_{};
};

class Tensor {
public:
  Tensor() = default;

  static Tensor make_cpu(DType dt, Shape shape);
  static Tensor wrap(Device dev, DType dt, void *data, Shape shape,
                     int64_t stride_c_bytes = 0);

  Device device() const { return view_.device(); }
  DType dtype() const { return view_.dtype(); }
  const Shape &shape() const { return view_.shape(); }
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
