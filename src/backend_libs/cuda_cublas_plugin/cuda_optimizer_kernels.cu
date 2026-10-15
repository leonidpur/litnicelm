#include "cuda_kernel_launchers.hpp"
#include "cuda_checks.hpp"
#include "cuda_kernel_config.hpp"

namespace cuda_cublas_plugin {
namespace {

__global__ void adamw_step_f32(KernelTensorView params, KernelTensorView grads,
                               KernelTensorView m, KernelTensorView v,
                               float learning_rate, float beta1, float beta2,
                               float weight_decay, float inv_b1_corr,
                               float inv_b2_corr, uint32_t apply_weight_decay) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = params.rows * params.cols;
  if (idx >= total) {
    return;
  }

  const int64_t r = idx / params.cols;
  const int64_t c = idx % params.cols;
  constexpr float kEps = 1e-8f;

  float *params_row = reinterpret_cast<float *>(row_ptr_mut(params, r));
  const float *grads_row = reinterpret_cast<const float *>(row_ptr(grads, r));
  float *m_row = reinterpret_cast<float *>(row_ptr_mut(m, r));
  float *v_row = reinterpret_cast<float *>(row_ptr_mut(v, r));

  const float grad = grads_row[c];
  const float old_m = m_row[c];
  const float old_v = v_row[c];
  const float new_m = beta1 * old_m + (1.0f - beta1) * grad;
  const float new_v = beta2 * old_v + (1.0f - beta2) * grad * grad;
  const float mhat = new_m * inv_b1_corr;
  const float vhat = new_v * inv_b2_corr;
  const float param = params_row[c];
  const float decay = apply_weight_decay != 0 ? (weight_decay * param) : 0.0f;
  const float adam = mhat / (sqrtf(vhat) + kEps);

  m_row[c] = new_m;
  v_row[c] = new_v;
  params_row[c] = param - learning_rate * (adam + decay);
}

} // namespace

void launch_adamw_step(TensorView &params, const TensorView &grads,
                       TensorView &m, TensorView &v, float learning_rate,
                       float beta1, float beta2, float weight_decay,
                       float inv_b1_corr, float inv_b2_corr,
                       uint32_t apply_weight_decay) {
  const int64_t total = tensor_rows(params) * tensor_cols(params);
  adamw_step_f32<<<
      static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                kThreadsPerBlock),
      kThreadsPerBlock>>>(
      to_kernel_tensor_view(params), to_kernel_tensor_view(grads),
      to_kernel_tensor_view(m), to_kernel_tensor_view(v), learning_rate, beta1,
      beta2, weight_decay, inv_b1_corr, inv_b2_corr, apply_weight_decay);
  check_kernel_launch("adamw_step_f32");
}

} // namespace cuda_cublas_plugin
