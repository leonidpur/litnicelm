#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

#include <cuda_bf16.h>

namespace cuda_cublas_plugin {
namespace {

__global__ void fill_f32(KernelTensorView t, float value) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= t.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= t.rows) {
    return;
  }
  reinterpret_cast<float *>(row_ptr_mut(t, r))[c] = value;
}

__global__ void add_f32(KernelTensorView a, KernelTensorView b,
                        KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const float *a_row = reinterpret_cast<const float *>(row_ptr(a, r));
  const float *b_row = reinterpret_cast<const float *>(row_ptr(b, r));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  out_row[c] = a_row[c] + b_row[c];
}

__global__ void add_batch_seq_plus_pos_f32(KernelTensorView a,
                                           KernelTensorView b,
                                           KernelTensorView out,
                                           int64_t seq_len) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const int pos = seq_len > 0 ? r % static_cast<int>(seq_len) : r;
  const float *a_row = reinterpret_cast<const float *>(row_ptr(a, r));
  const float *b_row = reinterpret_cast<const float *>(row_ptr(b, pos));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  out_row[c] = a_row[c] + b_row[c];
}

__global__ void add_inplace_f32(KernelTensorView a, KernelTensorView b) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= a.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= a.rows) {
    return;
  }
  float *a_row = reinterpret_cast<float *>(row_ptr_mut(a, r));
  const float *b_row = reinterpret_cast<const float *>(row_ptr(b, r));
  a_row[c] = a_row[c] + b_row[c];
}

__global__ void add_bias_rowwise_f32(KernelTensorView x, KernelTensorView bias,
                                     KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, r));
  const float *bias_row = reinterpret_cast<const float *>(row_ptr(bias, 0));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  out_row[c] = x_row[c] + bias_row[c];
}

__global__ void add_bias_relu_rowwise_f32(KernelTensorView x,
                                          KernelTensorView bias,
                                          KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, r));
  const float *bias_row = reinterpret_cast<const float *>(row_ptr(bias, 0));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  const float value = x_row[c] + bias_row[c];
  out_row[c] = value > 0.0f ? value : 0.0f;
}

__global__ void add_bias_relu_rowwise_inplace_f32(KernelTensorView x,
                                                  KernelTensorView bias) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= x.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= x.rows) {
    return;
  }
  float *x_row = reinterpret_cast<float *>(row_ptr_mut(x, r));
  const float *bias_row = reinterpret_cast<const float *>(row_ptr(bias, 0));
  const float value = x_row[c] + bias_row[c];
  x_row[c] = value > 0.0f ? value : 0.0f;
}

__global__ void mul_scalar_f32(KernelTensorView x, float scale,
                               KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, r));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  out_row[c] = x_row[c] * scale;
}

__global__ void relu_f32(KernelTensorView x, KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, r));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, r));
  const float value = x_row[c];
  out_row[c] = value > 0.0f ? value : 0.0f;
}

__global__ void relu_backward_f32(KernelTensorView preact,
                                  KernelTensorView dout,
                                  KernelTensorView dx) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dx.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= dx.rows) {
    return;
  }
  const float *preact_row =
      reinterpret_cast<const float *>(row_ptr(preact, r));
  const float *dout_row = reinterpret_cast<const float *>(row_ptr(dout, r));
  float *dx_row = reinterpret_cast<float *>(row_ptr_mut(dx, r));
  dx_row[c] = preact_row[c] > 0.0f ? dout_row[c] : 0.0f;
}

__global__ void relu_backward_inplace_f32(KernelTensorView preact,
                                          KernelTensorView dout_dx) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dout_dx.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= dout_dx.rows) {
    return;
  }
  const float *preact_row =
      reinterpret_cast<const float *>(row_ptr(preact, r));
  float *dout_dx_row = reinterpret_cast<float *>(row_ptr_mut(dout_dx, r));
  dout_dx_row[c] = preact_row[c] > 0.0f ? dout_dx_row[c] : 0.0f;
}

__global__ void transpose_f32(KernelTensorView x, KernelTensorView out) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= x.rows || c >= x.cols) {
    return;
  }
  const float *x_row = reinterpret_cast<const float *>(row_ptr(x, r));
  float *out_row = reinterpret_cast<float *>(row_ptr_mut(out, c));
  out_row[r] = x_row[c];
}

__global__ void apply_causal_mask_inplace_f32(KernelTensorView scores,
                                              float neg_inf) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= scores.cols) {
    return;
  }
  const int cols = static_cast<int>(scores.cols);
  const int r = grid_row();
  if (r >= scores.rows) {
    return;
  }
  // Rows are (..., query); columns beyond the query position are masked.
  if (c > r % cols) {
    reinterpret_cast<float *>(row_ptr_mut(scores, r))[c] = neg_inf;
  }
}

// Columns are rotation pairs: pair p rotates index p % half of head
// p / half.
__global__ void rotary_embedding_inplace_f32(KernelTensorView x,
                                             int64_t seq_len, int64_t n_heads,
                                             int64_t head_dim, float base,
                                             float sign) {
  const int half = static_cast<int>(head_dim / 2);
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= static_cast<int>(n_heads) * half) {
    return;
  }
  const int i = pair % half;
  const int h = pair / half;
  const float inv_freq = powf(base, -2.0f * static_cast<float>(i) /
                                        static_cast<float>(head_dim));
  const int c1 = h * static_cast<int>(head_dim) + i;
  const int c2 = c1 + half;
  const int row = grid_row();
  if (row >= x.rows) {
    return;
  }
  const float pos = static_cast<float>(row % static_cast<int>(seq_len));
  float s = 0.0f;
  float c = 0.0f;
  sincosf(pos * inv_freq, &s, &c);
  s *= sign;
  float *x_row = reinterpret_cast<float *>(row_ptr_mut(x, row));
  const float x1 = x_row[c1];
  const float x2 = x_row[c2];
  // Explicit FMAs: the same rounding in both variants, whatever the
  // compiler would contract.
  x_row[c1] = __fmaf_rn(x1, c, -(x2 * s));
  x_row[c2] = __fmaf_rn(x1, s, x2 * c);
}

// _bf16act: activations are BF16 (math in FP32 registers, one rounding per
// stored value); bias, the position table and add_inplace's accumulator
// stay F32.

__global__ void fill_bf16act(KernelTensorView t, float value) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= t.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= t.rows) {
    return;
  }
  reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(t, r))[c] =
      __float2bfloat16(value);
}

__global__ void add_bf16act(KernelTensorView a, KernelTensorView b,
                            KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const __nv_bfloat16 *a_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(a, r));
  const __nv_bfloat16 *b_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(b, r));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  out_row[c] =
      __float2bfloat16(__bfloat162float(a_row[c]) +
                       __bfloat162float(b_row[c]));
}

// a, out BF16 [B, S, D]; b the F32 position table [S, D].
__global__ void add_batch_seq_plus_pos_bf16act(KernelTensorView a,
                                               KernelTensorView b,
                                               KernelTensorView out,
                                               int64_t seq_len) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const int pos = seq_len > 0 ? r % static_cast<int>(seq_len) : r;
  const __nv_bfloat16 *a_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(a, r));
  const float *b_row = reinterpret_cast<const float *>(row_ptr(b, pos));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  out_row[c] = __float2bfloat16(__bfloat162float(a_row[c]) + b_row[c]);
}

// a: an F32 parameter gradient; b: a BF16 activation gradient.
__global__ void add_inplace_bf16act(KernelTensorView a, KernelTensorView b) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= a.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= a.rows) {
    return;
  }
  float *a_row = reinterpret_cast<float *>(row_ptr_mut(a, r));
  const __nv_bfloat16 *b_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(b, r));
  a_row[c] = a_row[c] + __bfloat162float(b_row[c]);
}

__global__ void add_bias_rowwise_bf16act(KernelTensorView x,
                                         KernelTensorView bias,
                                         KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, r));
  const float *bias_row = reinterpret_cast<const float *>(row_ptr(bias, 0));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  out_row[c] = __float2bfloat16(__bfloat162float(x_row[c]) + bias_row[c]);
}

__global__ void add_bias_relu_rowwise_bf16act(KernelTensorView x,
                                              KernelTensorView bias,
                                              KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, r));
  const float *bias_row = reinterpret_cast<const float *>(row_ptr(bias, 0));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  const float value = __bfloat162float(x_row[c]) + bias_row[c];
  out_row[c] = __float2bfloat16(value > 0.0f ? value : 0.0f);
}

__global__ void add_bias_relu_rowwise_inplace_bf16act(KernelTensorView x,
                                                      KernelTensorView bias) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= x.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= x.rows) {
    return;
  }
  __nv_bfloat16 *x_row = reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(x, r));
  const float *bias_row = reinterpret_cast<const float *>(row_ptr(bias, 0));
  const float value = __bfloat162float(x_row[c]) + bias_row[c];
  x_row[c] = __float2bfloat16(value > 0.0f ? value : 0.0f);
}

__global__ void mul_scalar_bf16act(KernelTensorView x, float scale,
                                   KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, r));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  out_row[c] = __float2bfloat16(__bfloat162float(x_row[c]) * scale);
}

__global__ void relu_bf16act(KernelTensorView x, KernelTensorView out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= out.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= out.rows) {
    return;
  }
  const __nv_bfloat16 *x_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(x, r));
  __nv_bfloat16 *out_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(out, r));
  const float value = __bfloat162float(x_row[c]);
  out_row[c] = __float2bfloat16(value > 0.0f ? value : 0.0f);
}

__global__ void relu_backward_bf16act(KernelTensorView preact,
                                      KernelTensorView dout,
                                      KernelTensorView dx) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dx.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= dx.rows) {
    return;
  }
  const __nv_bfloat16 *preact_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(preact, r));
  const __nv_bfloat16 *dout_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(dout, r));
  __nv_bfloat16 *dx_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(dx, r));
  dx_row[c] = __bfloat162float(preact_row[c]) > 0.0f ? dout_row[c]
                                                     : __float2bfloat16(0.0f);
}

__global__ void relu_backward_inplace_bf16act(KernelTensorView preact,
                                              KernelTensorView dout_dx) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= dout_dx.cols) {
    return;
  }
  const int r = grid_row();
  if (r >= dout_dx.rows) {
    return;
  }
  const __nv_bfloat16 *preact_row =
      reinterpret_cast<const __nv_bfloat16 *>(row_ptr(preact, r));
  __nv_bfloat16 *dout_dx_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(dout_dx, r));
  if (!(__bfloat162float(preact_row[c]) > 0.0f)) {
    dout_dx_row[c] = __float2bfloat16(0.0f);
  }
}

__global__ void apply_causal_mask_inplace_bf16act(KernelTensorView scores,
                                                  float neg_inf) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= scores.cols) {
    return;
  }
  const int cols = static_cast<int>(scores.cols);
  const int r = grid_row();
  if (r >= scores.rows) {
    return;
  }
  // Rows are (..., query); columns beyond the query position are masked.
  if (c > r % cols) {
    reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(scores, r))[c] =
        __float2bfloat16(neg_inf);
  }
}

// Columns are rotation pairs: pair p rotates index p % half of head
// p / half.
__global__ void rotary_embedding_inplace_bf16act(KernelTensorView x,
                                                 int64_t seq_len,
                                                 int64_t n_heads,
                                                 int64_t head_dim, float base,
                                                 float sign) {
  const int half = static_cast<int>(head_dim / 2);
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= static_cast<int>(n_heads) * half) {
    return;
  }
  const int i = pair % half;
  const int h = pair / half;
  const float inv_freq = powf(base, -2.0f * static_cast<float>(i) /
                                        static_cast<float>(head_dim));
  const int c1 = h * static_cast<int>(head_dim) + i;
  const int c2 = c1 + half;
  const int row = grid_row();
  if (row >= x.rows) {
    return;
  }
  const float pos = static_cast<float>(row % static_cast<int>(seq_len));
  float s = 0.0f;
  float c = 0.0f;
  sincosf(pos * inv_freq, &s, &c);
  s *= sign;
  __nv_bfloat16 *x_row =
      reinterpret_cast<__nv_bfloat16 *>(row_ptr_mut(x, row));
  const float x1 = __bfloat162float(x_row[c1]);
  const float x2 = __bfloat162float(x_row[c2]);
  // Explicit FMAs, as in the _f32 variant.
  x_row[c1] = __float2bfloat16(__fmaf_rn(x1, c, -(x2 * s)));
  x_row[c2] = __float2bfloat16(__fmaf_rn(x1, s, x2 * c));
}

} // namespace

void launch_fill_f32(TensorView &t, float value) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(t), tensor_cols(t));
  fill_f32<<<launch.grid, launch.block>>>(to_kernel_tensor_view(t), value);
  check_kernel_launch("fill_f32");
}

void launch_add_f32(const TensorView &a, const TensorView &b, TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_f32<<<launch.grid, launch.block>>>(to_kernel_tensor_view(a),
                                         to_kernel_tensor_view(b),
                                         to_kernel_tensor_view(out));
  check_kernel_launch("add_f32");
}

void launch_add_batch_seq_plus_pos_f32(const TensorView &a, const TensorView &b,
                                       TensorView &out, int64_t seq_len) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_batch_seq_plus_pos_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(a), to_kernel_tensor_view(b),
      to_kernel_tensor_view(out), seq_len);
  check_kernel_launch("add_batch_seq_plus_pos_f32");
}

void launch_add_inplace_f32(TensorView &a, const TensorView &b) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(a), tensor_cols(a));
  add_inplace_f32<<<launch.grid, launch.block>>>(to_kernel_tensor_view(a),
                                                 to_kernel_tensor_view(b));
  check_kernel_launch("add_inplace_f32");
}

void launch_add_bias_rowwise_f32(const TensorView &x, const TensorView &bias,
                                 TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_bias_rowwise_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(bias),
      to_kernel_tensor_view(out));
  check_kernel_launch("add_bias_rowwise_f32");
}

void launch_add_bias_relu_rowwise_f32(const TensorView &x,
                                      const TensorView &bias, TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_bias_relu_rowwise_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(bias),
      to_kernel_tensor_view(out));
  check_kernel_launch("add_bias_relu_rowwise_f32");
}

void launch_add_bias_relu_rowwise_inplace_f32(TensorView &x,
                                              const TensorView &bias) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(x), tensor_cols(x));
  add_bias_relu_rowwise_inplace_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(bias));
  check_kernel_launch("add_bias_relu_rowwise_inplace_f32");
}

void launch_mul_scalar_f32(const TensorView &x, float scale, TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  mul_scalar_f32<<<launch.grid, launch.block>>>(to_kernel_tensor_view(x), scale,
                                                to_kernel_tensor_view(out));
  check_kernel_launch("mul_scalar_f32");
}

void launch_relu_f32(const TensorView &x, TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  relu_f32<<<launch.grid, launch.block>>>(to_kernel_tensor_view(x),
                                          to_kernel_tensor_view(out));
  check_kernel_launch("relu_f32");
}

void launch_relu_backward_f32(const TensorView &preact, const TensorView &dout,
                              TensorView &dx) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(dx), tensor_cols(dx));
  relu_backward_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(preact), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("relu_backward_f32");
}

void launch_relu_backward_inplace_f32(const TensorView &preact,
                                      TensorView &dout_dx) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(dout_dx), tensor_cols(dout_dx));
  relu_backward_inplace_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(preact), to_kernel_tensor_view(dout_dx));
  check_kernel_launch("relu_backward_inplace_f32");
}

void launch_transpose(const TensorView &x, TensorView &out) {
  const dim3 block(kTileDim, kTileDim);
  const dim3 grid(
      static_cast<unsigned int>((tensor_cols(x) + block.x - 1) / block.x),
      static_cast<unsigned int>((tensor_rows(x) + block.y - 1) / block.y));
  transpose_f32<<<grid, block>>>(to_kernel_tensor_view(x),
                                 to_kernel_tensor_view(out));
  check_kernel_launch("transpose_f32");
}

void launch_apply_causal_mask_inplace_f32(TensorView &scores, float neg_inf) {
  // At most 16 wide: score rows are short and half masked.
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(scores), tensor_cols(scores), kTileDim);
  apply_causal_mask_inplace_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(scores), neg_inf);
  check_kernel_launch("apply_causal_mask_inplace_f32");
}

void launch_rotary_embedding_inplace_f32(TensorView &x, int64_t n_heads,
                                         float base, bool inverse) {
  const int64_t seq_len = x.dim(x.rank() - 2);
  const int64_t head_dim = tensor_cols(x) / n_heads;
  const int64_t pairs = n_heads * (head_dim / 2);
  if (pairs == 0) {
    return;
  }
  const RowsColsLaunch launch = rows_cols_launch(tensor_rows(x), pairs);
  rotary_embedding_inplace_f32<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), seq_len, n_heads, head_dim, base,
      inverse ? -1.0f : 1.0f);
  check_kernel_launch("rotary_embedding_inplace_f32");
}

void launch_fill_bf16act(TensorView &t, float value) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(t), tensor_cols(t));
  fill_bf16act<<<launch.grid, launch.block>>>(to_kernel_tensor_view(t), value);
  check_kernel_launch("fill_bf16act");
}

void launch_add_bf16act(const TensorView &a, const TensorView &b,
                        TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_bf16act<<<launch.grid, launch.block>>>(to_kernel_tensor_view(a),
                                             to_kernel_tensor_view(b),
                                             to_kernel_tensor_view(out));
  check_kernel_launch("add_bf16act");
}

void launch_add_batch_seq_plus_pos_bf16act(const TensorView &a,
                                           const TensorView &b, TensorView &out,
                                           int64_t seq_len) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_batch_seq_plus_pos_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(a), to_kernel_tensor_view(b),
      to_kernel_tensor_view(out), seq_len);
  check_kernel_launch("add_batch_seq_plus_pos_bf16act");
}

void launch_add_inplace_bf16act(TensorView &a, const TensorView &b) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(a), tensor_cols(a));
  add_inplace_bf16act<<<launch.grid, launch.block>>>(to_kernel_tensor_view(a),
                                                     to_kernel_tensor_view(b));
  check_kernel_launch("add_inplace_bf16act");
}

void launch_add_bias_rowwise_bf16act(const TensorView &x,
                                     const TensorView &bias, TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_bias_rowwise_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(bias),
      to_kernel_tensor_view(out));
  check_kernel_launch("add_bias_rowwise_bf16act");
}

void launch_add_bias_relu_rowwise_bf16act(const TensorView &x,
                                          const TensorView &bias,
                                          TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  add_bias_relu_rowwise_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(bias),
      to_kernel_tensor_view(out));
  check_kernel_launch("add_bias_relu_rowwise_bf16act");
}

void launch_add_bias_relu_rowwise_inplace_bf16act(TensorView &x,
                                                  const TensorView &bias) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(x), tensor_cols(x));
  add_bias_relu_rowwise_inplace_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(bias));
  check_kernel_launch("add_bias_relu_rowwise_inplace_bf16act");
}

void launch_mul_scalar_bf16act(const TensorView &x, float scale,
                               TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  mul_scalar_bf16act<<<launch.grid, launch.block>>>(to_kernel_tensor_view(x),
                                                    scale,
                                                    to_kernel_tensor_view(out));
  check_kernel_launch("mul_scalar_bf16act");
}

void launch_relu_bf16act(const TensorView &x, TensorView &out) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(out), tensor_cols(out));
  relu_bf16act<<<launch.grid, launch.block>>>(to_kernel_tensor_view(x),
                                              to_kernel_tensor_view(out));
  check_kernel_launch("relu_bf16act");
}

void launch_relu_backward_bf16act(const TensorView &preact,
                                  const TensorView &dout, TensorView &dx) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(dx), tensor_cols(dx));
  relu_backward_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(preact), to_kernel_tensor_view(dout),
      to_kernel_tensor_view(dx));
  check_kernel_launch("relu_backward_bf16act");
}

void launch_relu_backward_inplace_bf16act(const TensorView &preact,
                                          TensorView &dout_dx) {
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(dout_dx), tensor_cols(dout_dx));
  relu_backward_inplace_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(preact), to_kernel_tensor_view(dout_dx));
  check_kernel_launch("relu_backward_inplace_bf16act");
}

void launch_apply_causal_mask_inplace_bf16act(TensorView &scores,
                                              float neg_inf) {
  // At most 16 wide: score rows are short and half masked.
  const RowsColsLaunch launch =
      rows_cols_launch(tensor_rows(scores), tensor_cols(scores), kTileDim);
  apply_causal_mask_inplace_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(scores), neg_inf);
  check_kernel_launch("apply_causal_mask_inplace_bf16act");
}

void launch_rotary_embedding_inplace_bf16act(TensorView &x, int64_t n_heads,
                                             float base, bool inverse) {
  const int64_t seq_len = x.dim(x.rank() - 2);
  const int64_t head_dim = tensor_cols(x) / n_heads;
  const int64_t pairs = n_heads * (head_dim / 2);
  if (pairs == 0) {
    return;
  }
  const RowsColsLaunch launch = rows_cols_launch(tensor_rows(x), pairs);
  rotary_embedding_inplace_bf16act<<<launch.grid, launch.block>>>(
      to_kernel_tensor_view(x), seq_len, n_heads, head_dim, base,
      inverse ? -1.0f : 1.0f);
  check_kernel_launch("rotary_embedding_inplace_bf16act");
}

} // namespace cuda_cublas_plugin
