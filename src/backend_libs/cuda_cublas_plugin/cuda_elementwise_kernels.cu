#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

namespace cuda_cublas_plugin {
namespace {

__global__ void fill_kernel(KernelTensorView t, float value) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = t.rows * t.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / t.cols;
  const int64_t c = idx % t.cols;
  store_f32(t, r, c, value);
}

__global__ void add_kernel(KernelTensorView a, KernelTensorView b,
                           KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  store_f32(out, r, c, load_f32(a, r, c) + load_f32(b, r, c));
}

__global__ void add_batch_seq_plus_pos_kernel(KernelTensorView a,
                                              KernelTensorView b,
                                              KernelTensorView out,
                                              int64_t seq_len) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const int64_t pos = seq_len > 0 ? r % seq_len : r;
  store_f32(out, r, c, load_f32(a, r, c) + load_f32(b, pos, c));
}

__global__ void add_inplace_kernel(KernelTensorView a, KernelTensorView b) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = a.rows * a.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / a.cols;
  const int64_t c = idx % a.cols;
  store_f32(a, r, c, load_f32(a, r, c) + load_f32(b, r, c));
}

__global__ void add_bias_rowwise_kernel(KernelTensorView x, KernelTensorView bias,
                                        KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  store_f32(out, r, c, load_f32(x, r, c) + load_f32(bias, 0, c));
}

__global__ void add_bias_relu_rowwise_kernel(KernelTensorView x,
                                             KernelTensorView bias,
                                             KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const float value = load_f32(x, r, c) + load_f32(bias, 0, c);
  store_f32(out, r, c, value > 0.0f ? value : 0.0f);
}

__global__ void add_bias_relu_rowwise_inplace_kernel(KernelTensorView x,
                                                     KernelTensorView bias) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = x.rows * x.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / x.cols;
  const int64_t c = idx % x.cols;
  const float value = load_f32(x, r, c) + load_f32(bias, 0, c);
  store_f32(x, r, c, value > 0.0f ? value : 0.0f);
}

__global__ void mul_scalar_kernel(KernelTensorView x, float scale,
                                  KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  store_f32(out, r, c, load_f32(x, r, c) * scale);
}

__global__ void relu_kernel(KernelTensorView x, KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const float value = load_f32(x, r, c);
  store_f32(out, r, c, value > 0.0f ? value : 0.0f);
}

__global__ void relu_backward_kernel(KernelTensorView preact,
                                     KernelTensorView dout,
                                     KernelTensorView dx) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = dx.rows * dx.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / dx.cols;
  const int64_t c = idx % dx.cols;
  const float grad = load_f32(preact, r, c) > 0.0f ? load_f32(dout, r, c) : 0.0f;
  store_f32(dx, r, c, grad);
}

__global__ void relu_backward_inplace_kernel(KernelTensorView preact,
                                             KernelTensorView dout_dx) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = dout_dx.rows * dout_dx.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / dout_dx.cols;
  const int64_t c = idx % dout_dx.cols;
  const float grad =
      load_f32(preact, r, c) > 0.0f ? load_f32(dout_dx, r, c) : 0.0f;
  store_f32(dout_dx, r, c, grad);
}

__global__ void transpose_kernel(KernelTensorView x, KernelTensorView out) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= x.rows || c >= x.cols) {
    return;
  }
  store_f32(out, c, r, load_f32(x, r, c));
}

__global__ void apply_causal_mask_inplace_kernel(KernelTensorView scores,
                                                 float neg_inf) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= scores.rows || c >= scores.cols) {
    return;
  }
  const int64_t local_r = r % scores.cols;
  if (c > local_r) {
    store_f32(scores, r, c, neg_inf);
  }
}

} // namespace

void launch_fill(TensorView &t, float value) {
  const int64_t total = tensor_rows(t) * tensor_cols(t);
  fill_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                          kThreadsPerBlock),
                kThreadsPerBlock>>>(to_kernel_tensor_view(t), value);
  check_kernel_launch("fill_kernel");
}

void launch_add(const TensorView &a, const TensorView &b, TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  add_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                         kThreadsPerBlock),
               kThreadsPerBlock>>>(to_kernel_tensor_view(a),
                                   to_kernel_tensor_view(b),
                                   to_kernel_tensor_view(out));
  check_kernel_launch("add_kernel");
}

void launch_add_batch_seq_plus_pos(const TensorView &a, const TensorView &b,
                                   TensorView &out, int64_t seq_len) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  add_batch_seq_plus_pos_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(a), to_kernel_tensor_view(b),
                          to_kernel_tensor_view(out), seq_len);
  check_kernel_launch("add_batch_seq_plus_pos_kernel");
}

void launch_add_inplace(TensorView &a, const TensorView &b) {
  const int64_t total = tensor_rows(a) * tensor_cols(a);
  add_inplace_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(a), to_kernel_tensor_view(b));
  check_kernel_launch("add_inplace_kernel");
}

void launch_add_bias_rowwise(const TensorView &x, const TensorView &bias,
                             TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  add_bias_rowwise_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(x), to_kernel_tensor_view(bias),
                          to_kernel_tensor_view(out));
  check_kernel_launch("add_bias_rowwise_kernel");
}

void launch_add_bias_relu_rowwise(const TensorView &x, const TensorView &bias,
                                  TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  add_bias_relu_rowwise_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(x), to_kernel_tensor_view(bias),
                          to_kernel_tensor_view(out));
  check_kernel_launch("add_bias_relu_rowwise_kernel");
}

void launch_add_bias_relu_rowwise_inplace(TensorView &x,
                                          const TensorView &bias) {
  const int64_t total = tensor_rows(x) * tensor_cols(x);
  add_bias_relu_rowwise_inplace_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(x), to_kernel_tensor_view(bias));
  check_kernel_launch("add_bias_relu_rowwise_inplace_kernel");
}

void launch_mul_scalar(const TensorView &x, float scale, TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  mul_scalar_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                                kThreadsPerBlock),
                      kThreadsPerBlock>>>(to_kernel_tensor_view(x), scale,
                                          to_kernel_tensor_view(out));
  check_kernel_launch("mul_scalar_kernel");
}

void launch_relu(const TensorView &x, TensorView &out) {
  const int64_t total = tensor_rows(out) * tensor_cols(out);
  relu_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                          kThreadsPerBlock),
                kThreadsPerBlock>>>(to_kernel_tensor_view(x),
                                    to_kernel_tensor_view(out));
  check_kernel_launch("relu_kernel");
}

void launch_relu_backward(const TensorView &preact, const TensorView &dout,
                          TensorView &dx) {
  const int64_t total = tensor_rows(dx) * tensor_cols(dx);
  relu_backward_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(preact),
                          to_kernel_tensor_view(dout),
                          to_kernel_tensor_view(dx));
  check_kernel_launch("relu_backward_kernel");
}

void launch_relu_backward_inplace(const TensorView &preact,
                                  TensorView &dout_dx) {
  const int64_t total = tensor_rows(dout_dx) * tensor_cols(dout_dx);
  relu_backward_inplace_kernel<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(to_kernel_tensor_view(preact),
                          to_kernel_tensor_view(dout_dx));
  check_kernel_launch("relu_backward_inplace_kernel");
}

void launch_transpose(const TensorView &x, TensorView &out) {
  const dim3 block(kTileDim, kTileDim);
  const dim3 grid(
      static_cast<unsigned int>((tensor_cols(x) + block.x - 1) / block.x),
      static_cast<unsigned int>((tensor_rows(x) + block.y - 1) / block.y));
  transpose_kernel<<<grid, block>>>(to_kernel_tensor_view(x),
                                    to_kernel_tensor_view(out));
  check_kernel_launch("transpose_kernel");
}

void launch_apply_causal_mask_inplace(TensorView &scores, float neg_inf) {
  const dim3 block(kTileDim, kTileDim);
  const dim3 grid(
      static_cast<unsigned int>((tensor_cols(scores) + block.x - 1) / block.x),
      static_cast<unsigned int>((tensor_rows(scores) + block.y - 1) / block.y));
  apply_causal_mask_inplace_kernel<<<grid, block>>>(
      to_kernel_tensor_view(scores), neg_inf);
  check_kernel_launch("apply_causal_mask_inplace_kernel");
}

} // namespace cuda_cublas_plugin
