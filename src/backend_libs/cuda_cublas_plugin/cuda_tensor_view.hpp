#pragma once

#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"
#include "cuda_checks.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace cuda_cublas_plugin {

// What a kernel needs of a tensor: rows (all leading dims flattened) of
// cols contiguous elements, stride_r_bytes apart.
struct KernelTensorView {
  void *data;
  int64_t rows;
  int64_t cols;
  int64_t stride_r_bytes;
};

const TensorView &to_tensor_view(const TensorViewStruct &abi);
KernelTensorView to_kernel_tensor_view(const TensorView &view);
KernelTensorView to_kernel_index_vector_view(const TensorView &view);

bool is_power_of_two(uint32_t x);
int64_t tensor_cols(const TensorView &view);
int64_t tensor_rows(const TensorView &view);
uint64_t row_bytes(const TensorView &view);
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
// The kernel variant an op runs, set by the dtype of its activation
// operand: <op>_bf16act when that is BF16 (activations and activation
// gradients BF16; parameters, their gradients and optimizer state F32),
// otherwise <op>_f32.
enum class Variant { F32, Bf16Act };
Variant variant_of(const TensorView &activation);
// The dtype activation operands have in variant v.
DType activation_dtype(Variant v);

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

// Byte address of row r, rows running over all leading dims. Rows are
// stride_r_bytes apart (to_kernel_tensor_view refuses views whose rows are
// not evenly spaced) and the elements of a row are contiguous
// (check_cuda_row_major), so a kernel indexes the row through a pointer of
// the tensor's element type.
__device__ inline const char *row_ptr(const KernelTensorView &t, int64_t r) {
  return static_cast<const char *>(t.data) + r * t.stride_r_bytes;
}

__device__ inline char *row_ptr_mut(const KernelTensorView &t, int64_t r) {
  return const_cast<char *>(row_ptr(t, r));
}

} // namespace cuda_cublas_plugin
