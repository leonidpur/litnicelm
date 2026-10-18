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
  // The loss needs the target's logit, which the gradient below overwrites
  // in place: thread 0 reads it now, before any write (the reductions'
  // barriers order every write after this read).
  const float target_logit = threadIdx.x == 0 ? logits_row[target] : 0.0f;

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

// The mean cross-entropy loss alone, accumulated into the zeroed out_loss;
// logits are left as they are. An out-of-range target or a non-finite row
// makes the loss NaN, as in cross_entropy_mean_backward_inplace_f32.
__global__ void cross_entropy_mean_f32(KernelTensorView logits,
                                       KernelTensorView targets,
                                       KernelTensorView out_loss) {
  const int64_t row = blockIdx.x;
  if (row >= logits.rows) {
    return;
  }
  const float *logits_row =
      reinterpret_cast<const float *>(row_ptr(logits, row));
  float *loss = reinterpret_cast<float *>(out_loss.data);

  extern __shared__ float ce_scratch[];

  __shared__ int64_t target;
  if (threadIdx.x == 0) {
    target = reinterpret_cast<const int32_t *>(row_ptr(targets, row))[0];
  }
  __syncthreads();
  if (target < 0 || target >= logits.cols) {
    if (threadIdx.x == 0) {
      atomicAdd(loss, NAN);
    }
    return;
  }

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    local_max = fmaxf(local_max, logits_row[c]);
  }
  ce_scratch[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ce_scratch[threadIdx.x] =
          fmaxf(ce_scratch[threadIdx.x], ce_scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = ce_scratch[0];
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    local_sum += expf(logits_row[c] - row_max);
  }
  ce_scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ce_scratch[threadIdx.x] += ce_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float sum = ce_scratch[0];
    const float row_loss =
        isfinite(row_max) && sum > 0.0f && isfinite(sum)
            ? (logf(sum) + row_max - logits_row[target]) /
                  static_cast<float>(logits.rows)
            : NAN;
    atomicAdd(loss, row_loss);
  }
}

} // namespace

void launch_cross_entropy_mean_backward_inplace_f32(TensorView &logits,
                                                    const TensorView &targets,
                                                    TensorView &out_loss) {
  cross_entropy_mean_backward_inplace_f32<<<
      static_cast<unsigned int>(tensor_rows(logits)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(logits), to_kernel_index_vector_view(targets),
      to_kernel_tensor_view(out_loss));
  check_kernel_launch("cross_entropy_mean_backward_inplace_f32");
}

void launch_cross_entropy_mean_f32(const TensorView &logits,
                                   const TensorView &targets,
                                   TensorView &out_loss) {
  cross_entropy_mean_f32<<<
      static_cast<unsigned int>(tensor_rows(logits)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(logits), to_kernel_index_vector_view(targets),
      to_kernel_tensor_view(out_loss));
  check_kernel_launch("cross_entropy_mean_f32");
}

} // namespace cuda_cublas_plugin
