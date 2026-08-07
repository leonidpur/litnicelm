#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <algorithm>

namespace cuda_cublas_plugin {
namespace {

__global__ void sum_squares_f32_kernel(KernelTensorView x, float *out_sum_sq) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = x.rows * x.cols;

  float local_sum = 0.0f;
  for (int64_t i = idx; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / x.cols;
    const int64_t c = i % x.cols;
    const float v = load_f32(x, r, c);
    local_sum += v * v;
  }

  extern __shared__ float scratch[];
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(out_sum_sq, scratch[0]);
  }
}

__global__ void row_sum_kernel(KernelTensorView x, KernelTensorView out_1xC) {
  const int64_t col = blockIdx.x;
  if (col >= x.cols) {
    return;
  }

  extern __shared__ float row_sum_scratch[];
  float local_sum = 0.0f;
  for (int64_t row = threadIdx.x; row < x.rows; row += blockDim.x) {
    local_sum += load_f32(x, row, col);
  }
  row_sum_scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      row_sum_scratch[threadIdx.x] += row_sum_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    store_f32(out_1xC, 0, col, row_sum_scratch[0]);
  }
}

} // namespace

void launch_sum_squares_f32(const TensorView &x, float *device_sum_sq) {
  const int64_t total = tensor_rows(x) * tensor_cols(x);
  const unsigned int blocks = static_cast<unsigned int>(
      std::min<int64_t>((total + kThreadsPerBlock - 1) / kThreadsPerBlock,
                        1024));
  sum_squares_f32_kernel<<<
      blocks, kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), device_sum_sq);
  check_kernel_launch("sum_squares_f32_kernel");
}

void launch_row_sum(const TensorView &x, TensorView &out_1xC) {
  row_sum_kernel<<<static_cast<unsigned int>(tensor_cols(x)), kThreadsPerBlock,
                   static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(out_1xC));
  check_kernel_launch("row_sum_kernel");
}

} // namespace cuda_cublas_plugin
