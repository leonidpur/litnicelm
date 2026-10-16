#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cfloat>
#include <cmath>

namespace cuda_cublas_plugin {
namespace {

__global__ void cross_entropy_mean_backward_inplace_f32(
    KernelTensorView logits, KernelTensorView targets, KernelTensorView out_loss) {
  const int64_t row = blockIdx.x;
  if (row >= logits.rows) {
    return;
  }
  float *logits_row = reinterpret_cast<float *>(row_ptr_mut(logits, row));
  float *loss = reinterpret_cast<float *>(out_loss.data);

  extern __shared__ float scratch[];

  __shared__ int64_t target;
  if (threadIdx.x == 0) {
    target = reinterpret_cast<const int32_t *>(row_ptr(targets, row))[0];
  }
  __syncthreads();
  if (target < 0 || target >= logits.cols) {
    for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
      logits_row[c] = NAN;
    }
    if (threadIdx.x == 0) {
      atomicAdd(loss, NAN);
    }
    return;
  }

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    local_max = fmaxf(local_max, logits_row[c]);
  }
  scratch[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = scratch[0];
  // Every thread has read the result before scratch is reused.
  __syncthreads();
  if (!isfinite(row_max)) {
    for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
      logits_row[c] = NAN;
    }
    if (threadIdx.x == 0) {
      atomicAdd(loss, NAN);
    }
    return;
  }

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    local_sum += expf(logits_row[c] - row_max);
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float sum = scratch[0];
  if (!(sum > 0.0f) || !isfinite(sum)) {
    for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
      logits_row[c] = NAN;
    }
    if (threadIdx.x == 0) {
      atomicAdd(loss, NAN);
    }
    return;
  }
  const float inv_sum = 1.0f / sum;
  const float inv_token_rows = 1.0f / static_cast<float>(logits.rows);

  if (threadIdx.x == 0) {
    const float target_logit = logits_row[target];
    const float row_loss =
        (logf(sum) + row_max - target_logit) * inv_token_rows;
    atomicAdd(loss, row_loss);
  }

  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    const float p = expf(logits_row[c] - row_max) * inv_sum;
    float gradient = p;
    if (c == target) {
      gradient -= 1.0f;
    }
    logits_row[c] = gradient * inv_token_rows;
  }
}

} // namespace

void launch_cross_entropy_mean_backward_inplace(TensorView &logits,
                                                const TensorView &targets,
                                                TensorView &out_loss) {
  cross_entropy_mean_backward_inplace_f32<<<
      static_cast<unsigned int>(tensor_rows(logits)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(logits), to_kernel_index_vector_view(targets),
      to_kernel_tensor_view(out_loss));
  check_kernel_launch("cross_entropy_mean_backward_inplace_f32");
}

} // namespace cuda_cublas_plugin
