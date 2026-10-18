#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cuda_bf16.h>

#include <cfloat>

namespace cuda_cublas_plugin {
namespace {

__global__ void softmax_rows_f32(KernelTensorView x, KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, row));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, row));

  extern __shared__ float scratch[];

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_max = fmaxf(local_max, x_row[c]);
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

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float exponent = expf(x_row[c] - row_max);
    out_row[c] = exponent;
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
    out_row[c] *= inv_sum;
  }
}

__global__ void softmax_backward_rows_f32(KernelTensorView softmax,
                                          KernelTensorView dout,
                                          KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }
  const float *s_row = reinterpret_cast<const float *>(row_ptr(softmax, row));
  const float *dout_row = reinterpret_cast<const float *>(row_ptr(dout, row));
  float *dx_row = reinterpret_cast<float *>(row_ptr_mut(dx, row));

  extern __shared__ float softmax_bw_scratch[];

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c < softmax.cols; c += blockDim.x) {
    local_dot = __fmaf_rn(s_row[c], dout_row[c], local_dot);
  }
  softmax_bw_scratch[threadIdx.x] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      softmax_bw_scratch[threadIdx.x] +=
          softmax_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float dot = softmax_bw_scratch[0];

  for (int64_t c = threadIdx.x; c < dx.cols; c += blockDim.x) {
    dx_row[c] = s_row[c] * (dout_row[c] - dot);
  }
}

__global__ void scaled_causal_softmax_rows_f32(KernelTensorView scores,
                                               float scale,
                                               KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= scores.rows) {
    return;
  }
  const float *scores_row =
      reinterpret_cast<const float *>(row_ptr(scores, row));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, row));

  extern __shared__ float scratch[];
  const int64_t local_row = row % scores.cols;

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_max = fmaxf(local_max, scores_row[c] * scale);
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

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < scores.cols; c += blockDim.x) {
    float exponent = 0.0f;
    if (c <= local_row) {
      exponent = expf(__fmaf_rn(scores_row[c], scale, -row_max));
      local_sum += exponent;
    }
    out_row[c] = exponent;
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
    out_row[c] *= inv_sum;
  }
}

__global__ void softmax_backward_causal_rows_f32(KernelTensorView softmax,
                                                 KernelTensorView dout,
                                                 KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }
  const float *s_row = reinterpret_cast<const float *>(row_ptr(softmax, row));
  const float *dout_row = reinterpret_cast<const float *>(row_ptr(dout, row));
  float *dx_row = reinterpret_cast<float *>(row_ptr_mut(dx, row));

  extern __shared__ float scratch[];
  const int64_t local_row = row % softmax.cols;

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_dot = __fmaf_rn(s_row[c], dout_row[c], local_dot);
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
      grad = s_row[c] * (dout_row[c] - dot);
    }
    dx_row[c] = grad;
  }
}

// _bf16act: BF16 scores, probabilities and gradients; max, sums and dot
// products in FP32 registers. The forward kernels recompute each exponent
// for the output instead of storing it and rescaling, so every BF16 value
// is rounded once: the _f32 result rounded to BF16. Mul-adds are explicit
// FMAs, as in the _f32 kernels.

__global__ void softmax_rows_bf16act(KernelTensorView x, KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, row));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, row));

  extern __shared__ float softmax_bf16_scratch[];
  float *scratch = softmax_bf16_scratch;

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_max = fmaxf(local_max, __bfloat162float(x_row[c]));
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

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += expf(__bfloat162float(x_row[c]) - row_max);
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
    out_row[c] =
        __float2bfloat16(expf(__bfloat162float(x_row[c]) - row_max) * inv_sum);
  }
}

__global__ void softmax_backward_rows_bf16act(KernelTensorView softmax,
                                              KernelTensorView dout,
                                              KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }
  const __nv_bfloat16 *s_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(softmax, row));
  const __nv_bfloat16 *dout_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(dout, row));
  __nv_bfloat16 *dx_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(dx, row));

  extern __shared__ float softmax_bw_bf16_scratch[];

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c < softmax.cols; c += blockDim.x) {
    local_dot = __fmaf_rn(__bfloat162float(s_row[c]),
                          __bfloat162float(dout_row[c]), local_dot);
  }
  softmax_bw_bf16_scratch[threadIdx.x] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      softmax_bw_bf16_scratch[threadIdx.x] +=
          softmax_bw_bf16_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float dot = softmax_bw_bf16_scratch[0];

  for (int64_t c = threadIdx.x; c < dx.cols; c += blockDim.x) {
    dx_row[c] = __float2bfloat16(__bfloat162float(s_row[c]) *
                                 (__bfloat162float(dout_row[c]) - dot));
  }
}

__global__ void scaled_causal_softmax_rows_bf16act(KernelTensorView scores,
                                                   float scale,
                                                   KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= scores.rows) {
    return;
  }
  const __nv_bfloat16 *scores_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(scores, row));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, row));

  extern __shared__ float causal_bf16_scratch[];
  float *scratch = causal_bf16_scratch;
  const int64_t local_row = row % scores.cols;

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_max = fmaxf(local_max, __bfloat162float(scores_row[c]) * scale);
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

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_sum +=
        expf(__fmaf_rn(__bfloat162float(scores_row[c]), scale, -row_max));
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

  for (int64_t c = threadIdx.x; c < scores.cols; c += blockDim.x) {
    float p = 0.0f;
    if (c <= local_row) {
      p = expf(__fmaf_rn(__bfloat162float(scores_row[c]), scale, -row_max)) *
          inv_sum;
    }
    out_row[c] = __float2bfloat16(p);
  }
}

__global__ void softmax_backward_causal_rows_bf16act(KernelTensorView softmax,
                                                     KernelTensorView dout,
                                                     KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }
  const __nv_bfloat16 *s_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(softmax, row));
  const __nv_bfloat16 *dout_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(dout, row));
  __nv_bfloat16 *dx_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(dx, row));

  extern __shared__ float causal_bw_bf16_scratch[];
  float *scratch = causal_bw_bf16_scratch;
  const int64_t local_row = row % softmax.cols;

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c <= local_row; c += blockDim.x) {
    local_dot = __fmaf_rn(__bfloat162float(s_row[c]),
                          __bfloat162float(dout_row[c]), local_dot);
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
      grad = __bfloat162float(s_row[c]) * (__bfloat162float(dout_row[c]) - dot);
    }
    dx_row[c] = __float2bfloat16(grad);
  }
}

} // namespace

void launch_softmax_rows_f32(const TensorView &x, TensorView &out) {
  softmax_rows_f32<<<static_cast<unsigned int>(tensor_rows(x)),
                     kThreadsPerBlock,
                     static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(out));
  check_kernel_launch("softmax_rows_f32");
}

void launch_softmax_rows_bf16act(const TensorView &x, TensorView &out) {
  softmax_rows_bf16act<<<
      static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(out));
  check_kernel_launch("softmax_rows_bf16act");
}

void launch_softmax_backward_rows_f32(const TensorView &softmax,
                                      const TensorView &dout, TensorView &dx) {
  softmax_backward_rows_f32<<<
      static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("softmax_backward_rows_f32");
}

void launch_softmax_backward_rows_bf16act(const TensorView &softmax,
                                          const TensorView &dout,
                                          TensorView &dx) {
  softmax_backward_rows_bf16act<<<
      static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("softmax_backward_rows_bf16act");
}

void launch_scaled_causal_softmax_rows_f32(const TensorView &scores,
                                           float scale, TensorView &out,
                                           cudaStream_t stream) {
  scaled_causal_softmax_rows_f32<<<
      static_cast<unsigned int>(tensor_rows(scores)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float)), stream>>>(
      to_kernel_tensor_view(scores), scale, to_kernel_tensor_view(out));
  check_kernel_launch("scaled_causal_softmax_rows_f32");
}

void launch_scaled_causal_softmax_rows_bf16act(const TensorView &scores,
                                               float scale, TensorView &out,
                                               cudaStream_t stream) {
  scaled_causal_softmax_rows_bf16act<<<
      static_cast<unsigned int>(tensor_rows(scores)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float)), stream>>>(
      to_kernel_tensor_view(scores), scale, to_kernel_tensor_view(out));
  check_kernel_launch("scaled_causal_softmax_rows_bf16act");
}

void launch_softmax_backward_causal_rows_f32(const TensorView &softmax,
                                             const TensorView &dout,
                                             TensorView &dx,
                                             cudaStream_t stream) {
  softmax_backward_causal_rows_f32<<<
      static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float)), stream>>>(
      to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("softmax_backward_causal_rows_f32");
}

void launch_softmax_backward_causal_rows_bf16act(const TensorView &softmax,
                                                 const TensorView &dout,
                                                 TensorView &dx,
                                                 cudaStream_t stream) {
  softmax_backward_causal_rows_bf16act<<<
      static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float)), stream>>>(
      to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("softmax_backward_causal_rows_bf16act");
}

} // namespace cuda_cublas_plugin
