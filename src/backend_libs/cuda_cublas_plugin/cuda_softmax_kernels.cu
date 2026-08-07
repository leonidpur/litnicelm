#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cfloat>

namespace cuda_cublas_plugin {
namespace {

__global__ void softmax_rows_kernel(KernelTensorView x, KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }

  extern __shared__ float scratch[];

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_max = fmaxf(local_max, load_f32(x, row, c));
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

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float exponent = expf(load_f32(x, row, c) - row_max);
    store_f32(out, row, c, exponent);
    local_sum += exponent;
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / scratch[0];

  for (int64_t c = threadIdx.x; c < out.cols; c += blockDim.x) {
    store_f32(out, row, c, load_f32(out, row, c) * inv_sum);
  }
}

__global__ void softmax_backward_rows_kernel(KernelTensorView softmax,
                                             KernelTensorView dout,
                                             KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }

  extern __shared__ float softmax_bw_scratch[];

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c < softmax.cols; c += blockDim.x) {
    local_dot += load_f32(softmax, row, c) * load_f32(dout, row, c);
  }
  softmax_bw_scratch[threadIdx.x] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      softmax_bw_scratch[threadIdx.x] += softmax_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float dot = softmax_bw_scratch[0];

  for (int64_t c = threadIdx.x; c < dx.cols; c += blockDim.x) {
    const float s = load_f32(softmax, row, c);
    const float g = s * (load_f32(dout, row, c) - dot);
    store_f32(dx, row, c, g);
  }
}

__global__ void scaled_causal_softmax_rows_kernel(KernelTensorView scores,
                                                  float scale,
                                                  KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= scores.rows) {
    return;
  }

  extern __shared__ float scratch[];
  const int64_t local_row = row % scores.cols;

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_max = fmaxf(local_max, load_f32(scores, row, c) * scale);
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

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < scores.cols; c += blockDim.x) {
    float exponent = 0.0f;
    if (c <= local_row) {
      exponent = expf(load_f32(scores, row, c) * scale - row_max);
      local_sum += exponent;
    }
    store_f32(out, row, c, exponent);
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / scratch[0];

  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    store_f32(out, row, c, load_f32(out, row, c) * inv_sum);
  }
}

__global__ void softmax_backward_causal_rows_kernel(KernelTensorView softmax,
                                                    KernelTensorView dout,
                                                    KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }

  extern __shared__ float scratch[];
  const int64_t local_row = row % softmax.cols;

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_dot += load_f32(softmax, row, c) * load_f32(dout, row, c);
  }
  scratch[threadIdx.x] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float dot = scratch[0];

  for (int64_t c = threadIdx.x; c < dx.cols; c += blockDim.x) {
    float grad = 0.0f;
    if (c <= local_row) {
      const float s = load_f32(softmax, row, c);
      grad = s * (load_f32(dout, row, c) - dot);
    }
    store_f32(dx, row, c, grad);
  }
}

} // namespace

void launch_softmax_rows(const TensorView &x, TensorView &out) {
  softmax_rows_kernel<<<static_cast<unsigned int>(tensor_rows(x)),
                        kThreadsPerBlock,
                        static_cast<size_t>(kThreadsPerBlock *
                                            sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(out));
  check_kernel_launch("softmax_rows_kernel");
}

void launch_softmax_backward_rows(const TensorView &softmax,
                                  const TensorView &dout, TensorView &dx) {
  softmax_backward_rows_kernel<<<
      static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("softmax_backward_rows_kernel");
}

void launch_scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                       TensorView &out) {
  scaled_causal_softmax_rows_kernel<<<
      static_cast<unsigned int>(tensor_rows(scores)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(scores), scale, to_kernel_tensor_view(out));
  check_kernel_launch("scaled_causal_softmax_rows_kernel");
}

void launch_softmax_backward_causal_rows(const TensorView &softmax,
                                         const TensorView &dout,
                                         TensorView &dx) {
  softmax_backward_causal_rows_kernel<<<
      static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("softmax_backward_causal_rows_kernel");
}

} // namespace cuda_cublas_plugin
