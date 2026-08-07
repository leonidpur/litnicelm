#pragma once

#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"
#include "cuda_checks.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace cuda_cublas_plugin {

struct KernelTensorView {
  DType dtype;
  void *data;
  int32_t rank;
  int64_t dims[kMaxTensorRank];
  int64_t strides_bytes[kMaxTensorRank];
  int64_t rows;
  int64_t cols;
  int64_t stride_r_bytes;
  int64_t stride_c_bytes;
};

TensorView to_tensor_view(const BackendTensorView &abi);
KernelTensorView to_kernel_tensor_view(const TensorView &view);
KernelTensorView to_kernel_index_vector_view(const TensorView &view);

bool is_power_of_two(uint32_t x);
int64_t tensor_cols(const TensorView &view);
int64_t tensor_rows(const TensorView &view);
uint64_t row_bytes(const TensorView &view);
uint64_t span_bytes(const TensorView &view);
bool has_storage(const TensorView &view);
bool is_cuda_row_major(const TensorView &view);
bool is_cuda_f32_row_major(const TensorView &view);
bool is_cuda_f32_contiguous_row_major(const TensorView &view);
uint64_t logical_prefix_count(const TensorView &view, size_t suffix_rank);
int64_t prefix_matrix_byte_offset(const TensorView &view, uint64_t prefix,
                                  size_t suffix_rank);
float *prefix_matrix_ptr(TensorView &view, uint64_t prefix);
const float *prefix_matrix_ptr(const TensorView &view, uint64_t prefix);
void require_cuda_row_major(const TensorView &view, const char *what);
void require_cuda_f32_row_major(const TensorView &view, const char *what);
int leading_dim_f32(const TensorView &view);
void copy_tensor_2d(const TensorView &src, const TensorView &dst,
                    cudaMemcpyKind kind, const char *what);

struct HostTensorStage {
  std::vector<uint8_t> storage;
  TensorView view;

  explicit HostTensorStage(const TensorView &like);
};

__device__ inline int64_t kernel_prefix_offset_bytes(const KernelTensorView &t,
                                                     int64_t prefix,
                                                     int32_t prefix_rank) {
  int64_t offset = 0;
  for (int32_t axis = prefix_rank - 1; axis >= 0; --axis) {
    const int64_t dim = t.dims[axis];
    const int64_t coord = dim > 0 ? prefix % dim : 0;
    prefix = dim > 0 ? prefix / dim : 0;
    offset += coord * t.strides_bytes[axis];
  }
  return offset;
}

__device__ inline const char *tensor_ptr(const KernelTensorView &t, int64_t r,
                                         int64_t c) {
  if (t.rank >= 2) {
    const int32_t last_axis = t.rank - 1;
    const int64_t prefix_offset =
        kernel_prefix_offset_bytes(t, r, last_axis);
    return reinterpret_cast<const char *>(t.data) + prefix_offset +
           c * t.strides_bytes[last_axis];
  }
  return reinterpret_cast<const char *>(t.data) + r * t.stride_r_bytes +
         c * t.stride_c_bytes;
}

__device__ inline char *tensor_ptr_mut(const KernelTensorView &t, int64_t r,
                                       int64_t c) {
  if (t.rank >= 2) {
    const int32_t last_axis = t.rank - 1;
    const int64_t prefix_offset =
        kernel_prefix_offset_bytes(t, r, last_axis);
    return reinterpret_cast<char *>(t.data) + prefix_offset +
           c * t.strides_bytes[last_axis];
  }
  return reinterpret_cast<char *>(t.data) + r * t.stride_r_bytes +
         c * t.stride_c_bytes;
}

__device__ inline float load_f32(const KernelTensorView &t, int64_t r,
                                 int64_t c) {
  return *reinterpret_cast<const float *>(tensor_ptr(t, r, c));
}

__device__ inline void store_f32(const KernelTensorView &t, int64_t r, int64_t c,
                                 float value) {
  *reinterpret_cast<float *>(tensor_ptr_mut(t, r, c)) = value;
}

__device__ inline int32_t load_i32(const KernelTensorView &t, int64_t r,
                                   int64_t c) {
  return *reinterpret_cast<const int32_t *>(tensor_ptr(t, r, c));
}

__device__ inline int64_t load_index(const KernelTensorView &t, int64_t r,
                                     int64_t c) {
  if (t.dtype == DType::I32) {
    return static_cast<int64_t>(load_i32(t, r, c));
  }
  return static_cast<int64_t>(load_f32(t, r, c));
}

} // namespace cuda_cublas_plugin
