#pragma once

#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"
#include "cuda_checks.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
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

const TensorView &to_tensor_view(const TensorViewStruct &abi);
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
// Address of the prefix-th matrix (last two dims) of view.
void *prefix_matrix_ptr(TensorView &view, uint64_t prefix);
const void *prefix_matrix_ptr(const TensorView &view, uint64_t prefix);
// Capability: a GPU tensor with storage whose rows hold contiguous elements
// (rank >= 2). what names the operand, e.g. "gemm(a)".
void check_cuda_row_major(const TensorView &view, const char *what);
// The same, and the dtype is one of dtypes.
void check_cuda_row_major(const TensorView &view,
                          std::initializer_list<DType> dtypes,
                          const char *what);
// Row stride in elements of the view's dtype.
int leading_dim(const TensorView &view);
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

// Byte address of row r, rows running over all leading dims. The elements
// of a row are contiguous (check_cuda_row_major), so a kernel indexes the
// row through a pointer of the tensor's element type.
__device__ inline const char *row_ptr(const KernelTensorView &t, int64_t r) {
  if (t.rank >= 2) {
    return static_cast<const char *>(t.data) +
           kernel_prefix_offset_bytes(t, r, t.rank - 1);
  }
  return static_cast<const char *>(t.data) + r * t.stride_r_bytes;
}

__device__ inline char *row_ptr_mut(const KernelTensorView &t, int64_t r) {
  return const_cast<char *>(row_ptr(t, r));
}

} // namespace cuda_cublas_plugin
