// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cuda_bf16.h>

namespace cuda_cublas_plugin {
namespace {

constexpr float kLayerNormEps = 1e-5f;

// layernorm_param_grads tiles: kParamGradCols columns per block, each
// reduced over rows by kParamGradRowLanes lanes.
constexpr int kParamGradCols = 32;
constexpr int kParamGradRowLanes = 16;

// Row-warp kernels: one warp per row, kRowWarpRowsPerBlock rows per block.
// Lane l keeps columns l, l + 32, ... in registers, so a row of at most
// kRowWarpMaxCols columns is read from memory once. Each reduction is a
// fixed xor-shuffle tree, so results are bitwise reproducible, and the f32
// and bf16act kernels share one arithmetic. Wider rows use the block
// kernels above.
constexpr int kWarpSize = 32;
constexpr int kRowWarpThreadsPerBlock = 128;
constexpr int kRowWarpRowsPerBlock = kRowWarpThreadsPerBlock / kWarpSize;
constexpr int kRowWarpColsPerLane = 32;
constexpr int64_t kRowWarpMaxCols = kWarpSize * kRowWarpColsPerLane;

__global__ void layernorm_forward_rowwarp_f32(KernelTensorView x,
                                              KernelTensorView gamma,
                                              KernelTensorView beta,
                                              KernelTensorView out);
__global__ void layernorm_forward_rowwarp_bf16act(KernelTensorView x,
                                                  KernelTensorView gamma,
                                                  KernelTensorView beta,
                                                  KernelTensorView out);
// Write dx and each row's (mean, inv_std) like layernorm_backward_*.
__global__ void layernorm_backward_rowwarp_f32(KernelTensorView x,
                                               KernelTensorView gamma,
                                               KernelTensorView dout,
                                               KernelTensorView dx,
                                               float *row_stats);
__global__ void layernorm_backward_rowwarp_bf16act(KernelTensorView x,
                                                   KernelTensorView gamma,
                                                   KernelTensorView dout,
                                                   KernelTensorView dx,
                                                   float *row_stats);

// The backward kernels write dx and each row's (mean, inv_std); dgamma and
// dbeta follow in layernorm_param_grads_*. All accumulation is FP32 in a
// fixed order, so both are bitwise reproducible.
__global__ void layernorm_backward_f32(KernelTensorView x,
                                      KernelTensorView gamma,
                                      KernelTensorView dout,
                                      KernelTensorView dx,
                                      float *row_stats) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const float *dout_row = reinterpret_cast<const float *>(row_ptr(dout, row));
  float *dx_row = reinterpret_cast<float *>(row_ptr_mut(dx, row));

  // Two blockDim-sized halves: the second one is used only for the pair of
  // gradient sums, which are reduced together.
  extern __shared__ float ln_bw_scratch[];
  float *sum_a = ln_bw_scratch;
  float *sum_b = ln_bw_scratch + blockDim.x;

  // mean and inv_std exactly as layernorm_forward computes them.
  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += x_row[c];
  }
  sum_a[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_a[threadIdx.x] += sum_a[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = sum_a[0] / static_cast<float>(x.cols);
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float delta = x_row[c] - mean;
    local_var = __fmaf_rn(delta, delta, local_var);
  }
  sum_a[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_a[threadIdx.x] += sum_a[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std =
      rsqrtf(sum_a[0] / static_cast<float>(x.cols) + kLayerNormEps);
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  // Explicit roundings: xhat must equal the one layernorm_param_grads
  // recomputes from row_stats, and no product may be contracted into an add.
  float local_sum_dxhat = 0.0f;
  float local_sum_dxhat_xhat = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float xhat = __fmul_rn(__fsub_rn(x_row[c], mean), inv_std);
    const float dxhat = __fmul_rn(dout_row[c], gamma_row[c]);
    local_sum_dxhat = __fadd_rn(local_sum_dxhat, dxhat);
    local_sum_dxhat_xhat = __fmaf_rn(dxhat, xhat, local_sum_dxhat_xhat);
  }
  sum_a[threadIdx.x] = local_sum_dxhat;
  sum_b[threadIdx.x] = local_sum_dxhat_xhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_a[threadIdx.x] += sum_a[threadIdx.x + stride];
      sum_b[threadIdx.x] += sum_b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float sum_dxhat = sum_a[0];
  const float sum_dxhat_xhat = sum_b[0];
  const float n = static_cast<float>(x.cols);
  const float scale = inv_std / n;

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float xhat = __fmul_rn(__fsub_rn(x_row[c], mean), inv_std);
    const float dxhat = __fmul_rn(dout_row[c], gamma_row[c]);
    const float dxi = __fmul_rn(
        scale, __fmaf_rn(-xhat, sum_dxhat_xhat,
                         __fmaf_rn(n, dxhat, -sum_dxhat)));
    dx_row[c] = dxi;
  }
  if (threadIdx.x == 0) {
    row_stats[2 * row] = mean;
    row_stats[2 * row + 1] = inv_std;
  }
}

__global__ void layernorm_param_grads_f32(KernelTensorView x,
                                         KernelTensorView dout,
                                         const float *row_stats,
                                         KernelTensorView dgamma,
                                         KernelTensorView dbeta) {
  __shared__ float gamma_partial[kParamGradRowLanes][kParamGradCols];
  __shared__ float beta_partial[kParamGradRowLanes][kParamGradCols];
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kParamGradCols + threadIdx.x;

  // Lane threadIdx.y sums rows threadIdx.y, threadIdx.y + lanes, ... in
  // order; the lanes are then combined by a fixed tree.
  float acc_gamma = 0.0f;
  float acc_beta = 0.0f;
  if (col < x.cols) {
    for (int64_t r = threadIdx.y; r < x.rows; r += kParamGradRowLanes) {
      const float *x_row = reinterpret_cast<const float *>(row_ptr(x, r));
      const float *dout_row = reinterpret_cast<const float *>(row_ptr(dout, r));
      const float dyi = dout_row[col];
      const float xhat = __fmul_rn(
          __fsub_rn(x_row[col], row_stats[2 * r]), row_stats[2 * r + 1]);
      acc_gamma = __fmaf_rn(dyi, xhat, acc_gamma);
      acc_beta = __fadd_rn(acc_beta, dyi);
    }
  }
  gamma_partial[threadIdx.y][threadIdx.x] = acc_gamma;
  beta_partial[threadIdx.y][threadIdx.x] = acc_beta;
  __syncthreads();

  for (int stride = kParamGradRowLanes / 2; stride > 0; stride >>= 1) {
    if (threadIdx.y < stride) {
      gamma_partial[threadIdx.y][threadIdx.x] +=
          gamma_partial[threadIdx.y + stride][threadIdx.x];
      beta_partial[threadIdx.y][threadIdx.x] +=
          beta_partial[threadIdx.y + stride][threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.y == 0 && col < x.cols) {
    reinterpret_cast<float *>(row_ptr_mut(dgamma, 0))[col] = gamma_partial[0][threadIdx.x];
    reinterpret_cast<float *>(row_ptr_mut(dbeta, 0))[col] = beta_partial[0][threadIdx.x];
  }
}

__global__ void layernorm_forward_f32(KernelTensorView x,
                                      KernelTensorView gamma,
                                      KernelTensorView beta,
                                      KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const float *beta_row = reinterpret_cast<const float *>(row_ptr(beta, 0));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, row));

  extern __shared__ float scratch[];

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += x_row[c];
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(x.cols);
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float delta = x_row[c] - mean;
    local_var = __fmaf_rn(delta, delta, local_var);
  }
  scratch[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std =
      rsqrtf(scratch[0] / static_cast<float>(x.cols) + kLayerNormEps);

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float normalized = (x_row[c] - mean) * inv_std;
    out_row[c] = __fmaf_rn(normalized, gamma_row[c], beta_row[c]);
  }
}

// _bf16act: x, out, dout and dx BF16; gamma, beta, dgamma and dbeta F32.
// The same arithmetic as the _f32 kernels, with one rounding per BF16
// output.
__global__ void layernorm_backward_bf16act(KernelTensorView x,
                                          KernelTensorView gamma,
                                          KernelTensorView dout,
                                          KernelTensorView dx,
                                          float *row_stats) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }
  const __nv_bfloat16 *x_row = reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const __nv_bfloat16 *dout_row = reinterpret_cast<const __nv_bfloat16 *>(row_ptr(dout, row));
  __nv_bfloat16 *dx_row = reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(dx, row));

  // Two blockDim-sized halves: the second one is used only for the pair of
  // gradient sums, which are reduced together.
  extern __shared__ float ln_bw16_scratch[];
  float *sum_a = ln_bw16_scratch;
  float *sum_b = ln_bw16_scratch + blockDim.x;

  // mean and inv_std exactly as layernorm_forward computes them.
  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += __bfloat162float(x_row[c]);
  }
  sum_a[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_a[threadIdx.x] += sum_a[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = sum_a[0] / static_cast<float>(x.cols);
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float delta = __bfloat162float(x_row[c]) - mean;
    local_var = __fmaf_rn(delta, delta, local_var);
  }
  sum_a[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_a[threadIdx.x] += sum_a[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std =
      rsqrtf(sum_a[0] / static_cast<float>(x.cols) + kLayerNormEps);
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  // Explicit roundings: xhat must equal the one layernorm_param_grads
  // recomputes from row_stats, and no product may be contracted into an add.
  float local_sum_dxhat = 0.0f;
  float local_sum_dxhat_xhat = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float xhat = __fmul_rn(__fsub_rn(__bfloat162float(x_row[c]), mean), inv_std);
    const float dxhat = __fmul_rn(__bfloat162float(dout_row[c]), gamma_row[c]);
    local_sum_dxhat = __fadd_rn(local_sum_dxhat, dxhat);
    local_sum_dxhat_xhat = __fmaf_rn(dxhat, xhat, local_sum_dxhat_xhat);
  }
  sum_a[threadIdx.x] = local_sum_dxhat;
  sum_b[threadIdx.x] = local_sum_dxhat_xhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_a[threadIdx.x] += sum_a[threadIdx.x + stride];
      sum_b[threadIdx.x] += sum_b[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float sum_dxhat = sum_a[0];
  const float sum_dxhat_xhat = sum_b[0];
  const float n = static_cast<float>(x.cols);
  const float scale = inv_std / n;

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float xhat = __fmul_rn(__fsub_rn(__bfloat162float(x_row[c]), mean), inv_std);
    const float dxhat = __fmul_rn(__bfloat162float(dout_row[c]), gamma_row[c]);
    const float dxi = __fmul_rn(
        scale, __fmaf_rn(-xhat, sum_dxhat_xhat,
                         __fmaf_rn(n, dxhat, -sum_dxhat)));
    dx_row[c] = __float2bfloat16_rn(dxi);
  }
  if (threadIdx.x == 0) {
    row_stats[2 * row] = mean;
    row_stats[2 * row + 1] = inv_std;
  }
}

__global__ void layernorm_param_grads_bf16act(KernelTensorView x,
                                             KernelTensorView dout,
                                             const float *row_stats,
                                             KernelTensorView dgamma,
                                             KernelTensorView dbeta) {
  __shared__ float gamma_partial[kParamGradRowLanes][kParamGradCols];
  __shared__ float beta_partial[kParamGradRowLanes][kParamGradCols];
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kParamGradCols + threadIdx.x;

  // Lane threadIdx.y sums rows threadIdx.y, threadIdx.y + lanes, ... in
  // order; the lanes are then combined by a fixed tree.
  float acc_gamma = 0.0f;
  float acc_beta = 0.0f;
  if (col < x.cols) {
    for (int64_t r = threadIdx.y; r < x.rows; r += kParamGradRowLanes) {
      const __nv_bfloat16 *x_row = reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, r));
      const __nv_bfloat16 *dout_row = reinterpret_cast<const __nv_bfloat16 *>(row_ptr(dout, r));
      const float dyi = __bfloat162float(dout_row[col]);
      const float xhat = __fmul_rn(
          __fsub_rn(__bfloat162float(x_row[col]), row_stats[2 * r]), row_stats[2 * r + 1]);
      acc_gamma = __fmaf_rn(dyi, xhat, acc_gamma);
      acc_beta = __fadd_rn(acc_beta, dyi);
    }
  }
  gamma_partial[threadIdx.y][threadIdx.x] = acc_gamma;
  beta_partial[threadIdx.y][threadIdx.x] = acc_beta;
  __syncthreads();

  for (int stride = kParamGradRowLanes / 2; stride > 0; stride >>= 1) {
    if (threadIdx.y < stride) {
      gamma_partial[threadIdx.y][threadIdx.x] +=
          gamma_partial[threadIdx.y + stride][threadIdx.x];
      beta_partial[threadIdx.y][threadIdx.x] +=
          beta_partial[threadIdx.y + stride][threadIdx.x];
    }
    __syncthreads();
  }
  if (threadIdx.y == 0 && col < x.cols) {
    reinterpret_cast<float *>(row_ptr_mut(dgamma, 0))[col] = gamma_partial[0][threadIdx.x];
    reinterpret_cast<float *>(row_ptr_mut(dbeta, 0))[col] = beta_partial[0][threadIdx.x];
  }
}

__global__ void layernorm_forward_bf16act(KernelTensorView x,
                                      KernelTensorView gamma,
                                      KernelTensorView beta,
                                      KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const float *beta_row = reinterpret_cast<const float *>(row_ptr(beta, 0));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, row));

  extern __shared__ float ln_fw_bf16_scratch[];
  float *scratch = ln_fw_bf16_scratch;

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += __bfloat162float(x_row[c]);
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(x.cols);
  // Every thread has read the result before scratch is reused.
  __syncthreads();

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float delta = __bfloat162float(x_row[c]) - mean;
    local_var = __fmaf_rn(delta, delta, local_var);
  }
  scratch[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std =
      rsqrtf(scratch[0] / static_cast<float>(x.cols) + kLayerNormEps);

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float normalized = (__bfloat162float(x_row[c]) - mean) * inv_std;
    out_row[c] =
        __float2bfloat16(__fmaf_rn(normalized, gamma_row[c], beta_row[c]));
  }
}

// The sum of v over the warp, the same bits in every lane: each xor step
// adds the same two partial sums in every lane of a pair.
__device__ float warp_allreduce_sum(float v) {
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, mask));
  }
  return v;
}

// This thread's row under the row-warp launch, or -1 past the last row.
// A warp's lanes share the row, so a warp leaves or stays as a whole and
// the full-mask shuffles see all 32 lanes.
__device__ int64_t rowwarp_row(int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowWarpRowsPerBlock +
                      threadIdx.x / kWarpSize;
  return row < rows ? row : -1;
}

__global__ void __launch_bounds__(kRowWarpThreadsPerBlock, 6)
layernorm_forward_rowwarp_f32(KernelTensorView x,
                                              KernelTensorView gamma,
                                              KernelTensorView beta,
                                              KernelTensorView out) {
  const int64_t row = rowwarp_row(x.rows);
  if (row < 0) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const float *beta_row = reinterpret_cast<const float *>(row_ptr(beta, 0));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, row));
  const float n = static_cast<float>(x.cols);

  float xv[kRowWarpColsPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      xv[i] = x_row[c];
      local_sum = __fadd_rn(local_sum, xv[i]);
    }
  }
  const float mean = warp_allreduce_sum(local_sum) / n;

  float local_var = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      const float delta = __fsub_rn(xv[i], mean);
      local_var = __fmaf_rn(delta, delta, local_var);
    }
  }
  const float inv_std = rsqrtf(warp_allreduce_sum(local_var) / n + kLayerNormEps);

#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      const float xhat = __fmul_rn(__fsub_rn(xv[i], mean), inv_std);
      out_row[c] = __fmaf_rn(xhat, gamma_row[c], beta_row[c]);
    }
  }
}

__global__ void __launch_bounds__(kRowWarpThreadsPerBlock, 6)
layernorm_forward_rowwarp_bf16act(KernelTensorView x,
                                                  KernelTensorView gamma,
                                                  KernelTensorView beta,
                                                  KernelTensorView out) {
  const int64_t row = rowwarp_row(x.rows);
  if (row < 0) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const float *beta_row = reinterpret_cast<const float *>(row_ptr(beta, 0));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, row));
  const float n = static_cast<float>(x.cols);

  float xv[kRowWarpColsPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      xv[i] = __bfloat162float(x_row[c]);
      local_sum = __fadd_rn(local_sum, xv[i]);
    }
  }
  const float mean = warp_allreduce_sum(local_sum) / n;

  float local_var = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      const float delta = __fsub_rn(xv[i], mean);
      local_var = __fmaf_rn(delta, delta, local_var);
    }
  }
  const float inv_std = rsqrtf(warp_allreduce_sum(local_var) / n + kLayerNormEps);

#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      const float xhat = __fmul_rn(__fsub_rn(xv[i], mean), inv_std);
      out_row[c] =
          __float2bfloat16_rn(__fmaf_rn(xhat, gamma_row[c], beta_row[c]));
    }
  }
}

// mean and inv_std exactly as layernorm_forward_rowwarp_* computes them;
// xhat as layernorm_param_grads_* recomputes it from row_stats.
__global__ void __launch_bounds__(kRowWarpThreadsPerBlock, 4)
layernorm_backward_rowwarp_f32(KernelTensorView x,
                                               KernelTensorView gamma,
                                               KernelTensorView dout,
                                               KernelTensorView dx,
                                               float *row_stats) {
  const int64_t row = rowwarp_row(x.rows);
  if (row < 0) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const float *dout_row = reinterpret_cast<const float *>(row_ptr(dout, row));
  float *dx_row = reinterpret_cast<float *>(row_ptr_mut(dx, row));
  const float n = static_cast<float>(x.cols);

  // xv holds x, then xhat; dxhat holds dout * gamma.
  float xv[kRowWarpColsPerLane];
  float dxhat[kRowWarpColsPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      xv[i] = x_row[c];
      local_sum = __fadd_rn(local_sum, xv[i]);
    }
  }
  const float mean = warp_allreduce_sum(local_sum) / n;

  float local_var = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      const float delta = __fsub_rn(xv[i], mean);
      local_var = __fmaf_rn(delta, delta, local_var);
    }
  }
  const float inv_std = rsqrtf(warp_allreduce_sum(local_var) / n + kLayerNormEps);

  float local_sum_dxhat = 0.0f;
  float local_sum_dxhat_xhat = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      xv[i] = __fmul_rn(__fsub_rn(xv[i], mean), inv_std);
      dxhat[i] = __fmul_rn(dout_row[c], gamma_row[c]);
      local_sum_dxhat = __fadd_rn(local_sum_dxhat, dxhat[i]);
      local_sum_dxhat_xhat = __fmaf_rn(dxhat[i], xv[i], local_sum_dxhat_xhat);
    }
  }
  const float sum_dxhat = warp_allreduce_sum(local_sum_dxhat);
  const float sum_dxhat_xhat = warp_allreduce_sum(local_sum_dxhat_xhat);
  const float scale = inv_std / n;

#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      dx_row[c] = __fmul_rn(
          scale, __fmaf_rn(-xv[i], sum_dxhat_xhat,
                           __fmaf_rn(n, dxhat[i], -sum_dxhat)));
    }
  }
  if (lane == 0) {
    row_stats[2 * row] = mean;
    row_stats[2 * row + 1] = inv_std;
  }
}

__global__ void __launch_bounds__(kRowWarpThreadsPerBlock, 4)
layernorm_backward_rowwarp_bf16act(KernelTensorView x,
                                                   KernelTensorView gamma,
                                                   KernelTensorView dout,
                                                   KernelTensorView dx,
                                                   float *row_stats) {
  const int64_t row = rowwarp_row(x.rows);
  if (row < 0) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x % kWarpSize);
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, row));
  const float *gamma_row = reinterpret_cast<const float *>(row_ptr(gamma, 0));
  const __nv_bfloat16 *dout_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(dout, row));
  __nv_bfloat16 *dx_row = reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(dx, row));
  const float n = static_cast<float>(x.cols);

  // xv holds x, then xhat; dxhat holds dout * gamma.
  float xv[kRowWarpColsPerLane];
  float dxhat[kRowWarpColsPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      xv[i] = __bfloat162float(x_row[c]);
      local_sum = __fadd_rn(local_sum, xv[i]);
    }
  }
  const float mean = warp_allreduce_sum(local_sum) / n;

  float local_var = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      const float delta = __fsub_rn(xv[i], mean);
      local_var = __fmaf_rn(delta, delta, local_var);
    }
  }
  const float inv_std = rsqrtf(warp_allreduce_sum(local_var) / n + kLayerNormEps);

  float local_sum_dxhat = 0.0f;
  float local_sum_dxhat_xhat = 0.0f;
#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      xv[i] = __fmul_rn(__fsub_rn(xv[i], mean), inv_std);
      dxhat[i] = __fmul_rn(__bfloat162float(dout_row[c]), gamma_row[c]);
      local_sum_dxhat = __fadd_rn(local_sum_dxhat, dxhat[i]);
      local_sum_dxhat_xhat = __fmaf_rn(dxhat[i], xv[i], local_sum_dxhat_xhat);
    }
  }
  const float sum_dxhat = warp_allreduce_sum(local_sum_dxhat);
  const float sum_dxhat_xhat = warp_allreduce_sum(local_sum_dxhat_xhat);
  const float scale = inv_std / n;

#pragma unroll
  for (int i = 0; i < kRowWarpColsPerLane; ++i) {
    const int64_t c = lane + static_cast<int64_t>(i) * kWarpSize;
    if (c < x.cols) {
      dx_row[c] = __float2bfloat16_rn(__fmul_rn(
          scale, __fmaf_rn(-xv[i], sum_dxhat_xhat,
                           __fmaf_rn(n, dxhat[i], -sum_dxhat))));
    }
  }
  if (lane == 0) {
    row_stats[2 * row] = mean;
    row_stats[2 * row + 1] = inv_std;
  }
}

// Rows narrow enough for the row-warp kernels. The choice depends on the
// width alone, so a backward pass recomputes its forward pass's statistics.
bool use_rowwarp_kernels(int64_t cols) { return cols <= kRowWarpMaxCols; }

unsigned int rowwarp_blocks(int64_t rows) {
  return static_cast<unsigned int>((rows + kRowWarpRowsPerBlock - 1) /
                                   kRowWarpRowsPerBlock);
}
} // namespace

void launch_layernorm_forward_f32(const TensorView &x,
                                     const TensorView &gamma,
                                     const TensorView &beta, TensorView &out) {
  if (use_rowwarp_kernels(tensor_cols(x))) {
    layernorm_forward_rowwarp_f32<<<rowwarp_blocks(tensor_rows(x)),
                                    kRowWarpThreadsPerBlock>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
        to_kernel_tensor_view(beta), to_kernel_tensor_view(out));
    check_kernel_launch("layernorm_forward_rowwarp_f32");
    return;
  }
  layernorm_forward_f32<<<
      static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
      to_kernel_tensor_view(beta), to_kernel_tensor_view(out));
  check_kernel_launch("layernorm_forward_f32");
}

void launch_layernorm_backward_f32(const TensorView &x,
                                 const TensorView &gamma,
                                 const TensorView &dout, TensorView &dx,
                                 float *row_stats) {
  if (use_rowwarp_kernels(tensor_cols(x))) {
    layernorm_backward_rowwarp_f32<<<rowwarp_blocks(tensor_rows(x)),
                                    kRowWarpThreadsPerBlock>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
        to_kernel_tensor_view(dout), to_kernel_tensor_view(dx), row_stats);
    check_kernel_launch("layernorm_backward_rowwarp_f32");
    return;
  }
  layernorm_backward_f32<<<
      static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
      static_cast<size_t>(2 * kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
      to_kernel_tensor_view(dout), to_kernel_tensor_view(dx), row_stats);
  check_kernel_launch("layernorm_backward_f32");
}

void launch_layernorm_param_grads_f32(const TensorView &x,
                                     const TensorView &dout,
                                     const float *row_stats,
                                     TensorView &dgamma, TensorView &dbeta) {
  const auto blocks = static_cast<unsigned int>(
      (tensor_cols(x) + kParamGradCols - 1) / kParamGradCols);
  layernorm_param_grads_f32<<<blocks, dim3(kParamGradCols, kParamGradRowLanes)>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(dout), row_stats,
      to_kernel_tensor_view(dgamma), to_kernel_tensor_view(dbeta));
  check_kernel_launch("layernorm_param_grads_f32");
}

void launch_layernorm_forward_bf16act(const TensorView &x,
                                      const TensorView &gamma,
                                      const TensorView &beta, TensorView &out) {
  if (use_rowwarp_kernels(tensor_cols(x))) {
    layernorm_forward_rowwarp_bf16act<<<rowwarp_blocks(tensor_rows(x)),
                                    kRowWarpThreadsPerBlock>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
        to_kernel_tensor_view(beta), to_kernel_tensor_view(out));
    check_kernel_launch("layernorm_forward_rowwarp_bf16act");
    return;
  }
  layernorm_forward_bf16act<<<
      static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
      to_kernel_tensor_view(beta), to_kernel_tensor_view(out));
  check_kernel_launch("layernorm_forward_bf16act");
}

void launch_layernorm_backward_bf16act(const TensorView &x,
                                     const TensorView &gamma,
                                     const TensorView &dout, TensorView &dx,
                                     float *row_stats) {
  if (use_rowwarp_kernels(tensor_cols(x))) {
    layernorm_backward_rowwarp_bf16act<<<rowwarp_blocks(tensor_rows(x)),
                                    kRowWarpThreadsPerBlock>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
        to_kernel_tensor_view(dout), to_kernel_tensor_view(dx), row_stats);
    check_kernel_launch("layernorm_backward_rowwarp_bf16act");
    return;
  }
  layernorm_backward_bf16act<<<
      static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
      static_cast<size_t>(2 * kThreadsPerBlock * sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
      to_kernel_tensor_view(dout), to_kernel_tensor_view(dx), row_stats);
  check_kernel_launch("layernorm_backward_bf16act");
}

void launch_layernorm_param_grads_bf16act(const TensorView &x,
                                         const TensorView &dout,
                                         const float *row_stats,
                                         TensorView &dgamma, TensorView &dbeta) {
  const auto blocks = static_cast<unsigned int>(
      (tensor_cols(x) + kParamGradCols - 1) / kParamGradCols);
  layernorm_param_grads_bf16act<<<blocks, dim3(kParamGradCols, kParamGradRowLanes)>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(dout), row_stats,
      to_kernel_tensor_view(dgamma), to_kernel_tensor_view(dbeta));
  check_kernel_launch("layernorm_param_grads_bf16act");
}

} // namespace cuda_cublas_plugin
