#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

namespace cuda_cublas_plugin {
namespace {

constexpr float kLayerNormEps = 1e-5f;

__global__ void layernorm_backward_kernel(KernelTensorView x,
                                          KernelTensorView gamma,
                                          KernelTensorView dout,
                                          KernelTensorView dx,
                                          KernelTensorView dgamma,
                                          KernelTensorView dbeta) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }

  extern __shared__ double ln_bw_scratch[];

  double local_sum = 0.0;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += static_cast<double>(load_f32(x, row, c));
  }
  ln_bw_scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double mean = ln_bw_scratch[0] / static_cast<double>(x.cols);

  double local_var = 0.0;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const double delta = static_cast<double>(load_f32(x, row, c)) - mean;
    local_var += delta * delta;
  }
  ln_bw_scratch[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  constexpr double kEps = 1e-5;
  const double inv_std =
      rsqrt(static_cast<double>(ln_bw_scratch[0] / static_cast<double>(x.cols)) + kEps);

  double local_sum_dxhat = 0.0;
  double local_sum_dxhat_xhat = 0.0;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const double xhat =
        (static_cast<double>(load_f32(x, row, c)) - mean) * inv_std;
    const double g = static_cast<double>(load_f32(gamma, 0, c));
    const double dyi = static_cast<double>(load_f32(dout, row, c));
    const double dxhat = dyi * g;
    local_sum_dxhat += dxhat;
    local_sum_dxhat_xhat += dxhat * xhat;
    atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(dgamma, 0, c)),
              static_cast<float>(dyi * xhat));
    atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(dbeta, 0, c)),
              static_cast<float>(dyi));
  }
  ln_bw_scratch[threadIdx.x] = local_sum_dxhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double sum_dxhat = ln_bw_scratch[0];

  ln_bw_scratch[threadIdx.x] = local_sum_dxhat_xhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double sum_dxhat_xhat = ln_bw_scratch[0];
  const double n = static_cast<double>(x.cols);

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const double xhat =
        (static_cast<double>(load_f32(x, row, c)) - mean) * inv_std;
    const double g = static_cast<double>(load_f32(gamma, 0, c));
    const double dyi = static_cast<double>(load_f32(dout, row, c));
    const double dxhat = dyi * g;
    const double dxi =
        (inv_std / n) * (n * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
    store_f32(dx, row, c, static_cast<float>(dxi));
  }
}

__global__ void layernorm_forward_kernel(KernelTensorView x,
                                         KernelTensorView gamma,
                                         KernelTensorView beta,
                                         KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }

  extern __shared__ float scratch[];

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += load_f32(x, row, c);
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

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float delta = load_f32(x, row, c) - mean;
    local_var += delta * delta;
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
    const float normalized = (load_f32(x, row, c) - mean) * inv_std;
    const float value =
        normalized * load_f32(gamma, 0, c) + load_f32(beta, 0, c);
    store_f32(out, row, c, value);
  }
}

} // namespace

void launch_layernorm_forward(const TensorView &x, const TensorView &gamma,
                              const TensorView &beta, TensorView &out) {
  layernorm_forward_kernel<<<static_cast<unsigned int>(tensor_rows(x)),
                             kThreadsPerBlock,
                             static_cast<size_t>(kThreadsPerBlock *
                                                 sizeof(float))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
      to_kernel_tensor_view(beta), to_kernel_tensor_view(out));
  check_kernel_launch("layernorm_forward_kernel");
}

void launch_layernorm_backward(const TensorView &x, const TensorView &gamma,
                               const TensorView &dout, TensorView &dx,
                               TensorView &dgamma, TensorView &dbeta) {
  layernorm_backward_kernel<<<
      static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
      static_cast<size_t>(kThreadsPerBlock * sizeof(double))>>>(
      to_kernel_tensor_view(x), to_kernel_tensor_view(gamma),
      to_kernel_tensor_view(dout), to_kernel_tensor_view(dx),
      to_kernel_tensor_view(dgamma), to_kernel_tensor_view(dbeta));
  check_kernel_launch("layernorm_backward_kernel");
}

} // namespace cuda_cublas_plugin
