#pragma once

#include <cstdint>
#include <array>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <litnicelm_utils/device.hpp>
#include <litnicelm_utils/dtype.hpp>
#include <litnicelm_utils/shape.hpp>
#include <litnicelm_utils/tensor_view_struct.h>
#include <type_traits>
#include <utility>
#include <vector>

// A non-owning view of tensor memory. It is a TensorViewStruct (the ABI
// form) with methods and adds no data, so a TensorView can be handed to a
// plugin as a TensorViewStruct pointer and used again as a TensorView.
class TensorView : public TensorViewStruct {
public:
  TensorView() : TensorViewStruct{} {}

  TensorView(Device dev, DType dt, void *data, Shape shape) : TensorViewStruct{} {
    set_(dev, dt, data, shape);
    const int64_t elem = static_cast<int64_t>(dtype_size(dt));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
    set_strides_(compute_contiguous_strides_(shape, elem));
  }

  TensorView(Device dev, DType dt, void *data, Shape shape,
             int64_t stride_c_bytes, int64_t stride_r_bytes)
      : TensorViewStruct{} {
    set_(dev, dt, data, shape);
    const int64_t elem = static_cast<int64_t>(dtype_size(dt));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
    std::array<int64_t, kMaxTensorRank> strides = compute_contiguous_strides_(shape, elem);
    if (shape.rank() >= 1) {
      strides[shape.rank() - 1] = (stride_c_bytes == 0) ? elem : stride_c_bytes;
    }
    if (shape.rank() >= 2) {
      strides[shape.rank() - 2] =
          (stride_r_bytes == 0)
              ? static_cast<int64_t>(shape.dim(shape.rank() - 1)) *
                    strides[shape.rank() - 1]
              : stride_r_bytes;
      for (size_t i = shape.rank() - 2; i > 0; --i) {
        strides[i - 1] = strides[i] * shape.dim(i);
      }
    }
    set_strides_(strides);
  }

  TensorView(Device dev, DType dt, void *data, Shape shape,
             const std::array<int64_t, kMaxTensorRank> &strides_bytes)
      : TensorViewStruct{} {
    set_(dev, dt, data, shape);
    set_strides_(strides_bytes);
    const int64_t elem = static_cast<int64_t>(dtype_size(dt));
    if (elem <= 0) {
      throw std::runtime_error("TensorView: invalid dtype size");
    }
  }

  Device device() const { return static_cast<Device>(device_id); }
  DType dtype() const { return static_cast<DType>(dtype_id); }
  Shape shape() const { return Shape(ndim, dims); }
  Shape logical_shape() const { return shape(); }
  size_t rank() const { return ndim; }
  int64_t dim(size_t axis) const {
    if (axis >= ndim) {
      throw std::out_of_range("Shape::dim axis out of range");
    }
    return dims[axis];
  }
  Shape shape_nd() const { return shape(); }

  int64_t stride_bytes(size_t axis) const {
    if (axis >= ndim) {
      throw std::out_of_range("TensorView::stride_bytes axis out of range");
    }
    return strides_bytes[axis];
  }

  void *data() const { return ptr; }

  uint64_t bytes() const { return nbytes(shape(), dtype()); }
  uint64_t numel() const { return shape().numel(); }

  bool is_contiguous_row_major() const {
    const int64_t elem = static_cast<int64_t>(dtype_size(dtype()));
    const std::array<int64_t, kMaxTensorRank> expected =
        compute_contiguous_strides_(shape(), elem);
    for (size_t i = 0; i < kMaxTensorRank; ++i) {
      if (strides_bytes[i] != expected[i]) {
        return false;
      }
    }
    return true;
  }

  bool is_contiguous() const { return is_contiguous_row_major(); }

  TensorView slice(size_t axis, int64_t start, int64_t len) const {
    if (axis >= ndim) {
      throw std::out_of_range("TensorView::slice axis out of range");
    }
    if (start < 0 || len < 0 || start + len > dims[axis]) {
      throw std::out_of_range("TensorView::slice bounds");
    }
    std::vector<int64_t> rebuilt;
    rebuilt.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
      rebuilt.push_back((i == axis) ? len : dims[i]);
    }
    Shape sliced(rebuilt);
    uint8_t *base = reinterpret_cast<uint8_t *>(ptr);
    void *sub_ptr = base + start * strides_bytes[axis];
    return TensorView(device(), dtype(), sub_ptr, sliced, strides_array_());
  }

  TensorView select(size_t axis, int64_t index) const {
    if (axis >= ndim) {
      throw std::out_of_range("TensorView::select axis out of range");
    }
    if (index < 0 || index >= dims[axis]) {
      throw std::out_of_range("TensorView::select bounds");
    }
    std::vector<int64_t> rebuilt;
    rebuilt.reserve(ndim > 0 ? ndim - 1 : 0);
    for (size_t i = 0; i < ndim; ++i) {
      if (i != axis) {
        rebuilt.push_back(dims[i]);
      }
    }
    Shape selected(rebuilt);
    uint8_t *base = reinterpret_cast<uint8_t *>(ptr);
    void *sub_ptr = base + index * strides_bytes[axis];
    std::array<int64_t, kMaxTensorRank> selected_strides{};
    size_t out_axis = 0;
    for (size_t i = 0; i < ndim; ++i) {
      if (i != axis) {
        selected_strides[out_axis++] = strides_bytes[i];
      }
    }
    return TensorView(device(), dtype(), sub_ptr, selected, selected_strides);
  }

  TensorView subcols(int64_t col_offset, int64_t sub_cols) const {
    if (ndim == 0) {
      throw std::out_of_range("TensorView::subcols requires rank >= 1");
    }
    const size_t axis = ndim - 1;
    if (col_offset < 0 || sub_cols < 0 || col_offset + sub_cols > dims[axis]) {
      throw std::out_of_range("TensorView::subcols out of bounds");
    }
    return slice(axis, col_offset, sub_cols);
  }

  TensorView subrows(int64_t row_offset, int64_t sub_rows) const {
    if (ndim == 0) {
      throw std::out_of_range("TensorView::subrows requires rank >= 1");
    }
    if (row_offset < 0 || sub_rows < 0 || row_offset + sub_rows > dims[0]) {
      throw std::out_of_range("TensorView::subrows out of bounds");
    }
    return slice(0, row_offset, sub_rows);
  }

  float *f32() const {
    if (dtype() != DType::F32) {
      throw std::runtime_error("TensorView: not f32");
    }
    return reinterpret_cast<float *>(ptr);
  }

  float at_f32(int64_t r, int64_t c) const {
    if (device() != Device::CPU) {
      throw std::runtime_error("TensorView::at_f32 CPU only");
    }
    if (dtype() != DType::F32) {
      throw std::runtime_error("TensorView::at_f32 requires f32");
    }
    if (ndim != 2 || r < 0 || c < 0 || r >= dims[0] || c >= dims[1]) {
      throw std::out_of_range("at_f32 oob");
    }
    const uint8_t *base = reinterpret_cast<const uint8_t *>(ptr);
    const uint8_t *p = base + r * stride_bytes(0) + c * stride_bytes(1);
    float out;
    std::memcpy(&out, p, sizeof(float));
    return out;
  }

  void set_f32(int64_t r, int64_t c, float v) const {
    if (device() != Device::CPU) {
      throw std::runtime_error("TensorView::set_f32 CPU only");
    }
    if (dtype() != DType::F32) {
      throw std::runtime_error("TensorView::set_f32 requires f32");
    }
    if (ndim != 2 || r < 0 || c < 0 || r >= dims[0] || c >= dims[1]) {
      throw std::out_of_range("set_f32 oob");
    }
    uint8_t *base = reinterpret_cast<uint8_t *>(ptr);
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

  void set_(Device dev, DType dt, void *data, const Shape &shape) {
    ptr = data;
    device_id = static_cast<uint32_t>(dev);
    dtype_id = static_cast<uint32_t>(dt);
    ndim = static_cast<uint32_t>(shape.rank());
    for (size_t i = 0; i < shape.rank(); ++i) {
      dims[i] = shape.dim(i);
    }
  }

  void set_strides_(const std::array<int64_t, kMaxTensorRank> &strides) {
    for (size_t i = 0; i < kMaxTensorRank; ++i) {
      strides_bytes[i] = strides[i];
    }
  }

  std::array<int64_t, kMaxTensorRank> strides_array_() const {
    std::array<int64_t, kMaxTensorRank> out{};
    for (size_t i = 0; i < kMaxTensorRank; ++i) {
      out[i] = strides_bytes[i];
    }
    return out;
  }
};

// A TensorView is exactly its TensorViewStruct: same size, standard layout.
static_assert(sizeof(TensorView) == sizeof(TensorViewStruct),
              "TensorView must add no data to TensorViewStruct");
static_assert(std::is_standard_layout_v<TensorView>,
              "TensorView must keep the C layout");

// A plugin's view of a TensorViewStruct it received across the ABI: the same
// memory, checked for a sane rank and leading dims. who names the receiver.
inline const TensorView &as_tensor_view(const TensorViewStruct &s, const char *who) {
  if (s.ndim > kMaxTensorRank) {
    throw std::runtime_error(std::string(who) + ": rank exceeds backend max");
  }
  if (s.ndim >= 1 && s.dims[0] <= 0) {
    throw std::runtime_error(std::string(who) + ": invalid leading dim");
  }
  if (s.ndim >= 2 && s.dims[1] <= 0) {
    throw std::runtime_error(std::string(who) + ": invalid second dim");
  }
  return static_cast<const TensorView &>(s);
}

class Tensor {
public:
  Tensor() = default;

  static Tensor make_cpu(DType dt, Shape shape);
  static Tensor wrap(Device dev, DType dt, void *data, Shape shape,
                     int64_t stride_c_bytes = 0);

  Device device() const { return view_.device(); }
  DType dtype() const { return view_.dtype(); }
  Shape shape() const { return view_.shape(); }
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
