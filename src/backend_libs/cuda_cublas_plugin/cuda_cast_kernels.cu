#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cstdint>

namespace cuda_cublas_plugin {
namespace {

// BF16 values are stored as their 16-bit patterns. The rounding is written
// out rather than taken from __float2bfloat16 so that NaN keeps its sign and
// payload bits exactly as the host's f32_to_bf16_bits does.
__global__ void cast_f32_to_bf16(KernelTensorView src, KernelTensorView dst) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = dst.rows * dst.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / dst.cols;
  const int64_t c = idx % dst.cols;
  const float *src_row = reinterpret_cast<const float *>(row_ptr(src, r));
  uint16_t *dst_row = reinterpret_cast<uint16_t *>(row_ptr_mut(dst, r));
  const uint32_t bits = __float_as_uint(src_row[c]);
  const bool nan = (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0;
  const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
  dst_row[c] = static_cast<uint16_t>(nan ? ((bits >> 16) | 0x0040u) : rounded);
}

__global__ void cast_bf16_to_f32(KernelTensorView src, KernelTensorView dst) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = dst.rows * dst.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / dst.cols;
  const int64_t c = idx % dst.cols;
  const uint16_t *src_row = reinterpret_cast<const uint16_t *>(row_ptr(src, r));
  float *dst_row = reinterpret_cast<float *>(row_ptr_mut(dst, r));
  dst_row[c] = __uint_as_float(static_cast<uint32_t>(src_row[c]) << 16);
}

} // namespace

void launch_cast_f32_to_bf16(const TensorView &src, TensorView &dst) {
  const int64_t total = tensor_rows(dst) * tensor_cols(dst);
  cast_f32_to_bf16<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                               kThreadsPerBlock),
                     kThreadsPerBlock>>>(to_kernel_tensor_view(src),
                                         to_kernel_tensor_view(dst));
  check_kernel_launch("cast_f32_to_bf16");
}

void launch_cast_bf16_to_f32(const TensorView &src, TensorView &dst) {
  const int64_t total = tensor_rows(dst) * tensor_cols(dst);
  cast_bf16_to_f32<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                               kThreadsPerBlock),
                     kThreadsPerBlock>>>(to_kernel_tensor_view(src),
                                         to_kernel_tensor_view(dst));
  check_kernel_launch("cast_bf16_to_f32");
}

} // namespace cuda_cublas_plugin
