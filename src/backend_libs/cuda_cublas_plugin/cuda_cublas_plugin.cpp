#include "cuda_backend.hpp"
#include "cuda_tensor_view.hpp"
#include "../../model/backend/backend_plugin_api.hpp"

namespace cuda_cublas_plugin {
namespace {

void *plugin_create(uint32_t device) {
  return create_backend(device);
}

void plugin_destroy(void *backend) {
  delete reinterpret_cast<DeviceBackend *>(backend);
}

uint32_t plugin_device(void *backend) {
  return static_cast<uint32_t>(backend_from_opaque(backend).device());
}

void *plugin_alloc(void *backend, uint64_t bytes, uint32_t alignment) {
  return backend_from_opaque(backend).alloc(bytes, alignment);
}

void plugin_free(void *backend, void *ptr) {
  backend_from_opaque(backend).free(ptr);
}

void plugin_copy_host2device(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  backend_from_opaque(backend).copy_host2device(dst, src, bytes);
}

void plugin_copy_device2host(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  backend_from_opaque(backend).copy_device2host(dst, src, bytes);
}

void plugin_copy(void *backend, const BackendTensorView *src,
                 const BackendTensorView *dst) {
  TensorView dst_view = to_tensor_view(*dst);
  backend_from_opaque(backend).copy(to_tensor_view(*src), dst_view);
}

void plugin_fill(void *backend, const BackendTensorView *t, float v) {
  TensorView t_view = to_tensor_view(*t);
  backend_from_opaque(backend).fill(t_view, v);
}

void plugin_add(void *backend, const BackendTensorView *a,
                const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).add(to_tensor_view(*a), to_tensor_view(*b),
                                 out_view);
}

void plugin_add_inplace(void *backend, const BackendTensorView *a,
                        const BackendTensorView *b) {
  TensorView a_view = to_tensor_view(*a);
  backend_from_opaque(backend).add_inplace(a_view, to_tensor_view(*b));
}

void plugin_add_bias_rowwise(void *backend, const BackendTensorView *x,
                             const BackendTensorView *bias_1xC,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).add_bias_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_add_bias_relu_rowwise(void *backend, const BackendTensorView *x,
                                  const BackendTensorView *bias_1xC,
                                  const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).add_bias_relu_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_add_bias_relu_rowwise_inplace(void *backend,
                                          const BackendTensorView *x,
                                          const BackendTensorView *bias_1xC) {
  TensorView x_view = to_tensor_view(*x);
  backend_from_opaque(backend).add_bias_relu_rowwise_inplace(
      x_view, to_tensor_view(*bias_1xC));
}

void plugin_mul_scalar(void *backend, const BackendTensorView *x, float s,
                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).mul_scalar(to_tensor_view(*x), s, out_view);
}

float plugin_sum_squares_f32(void *backend, const BackendTensorView *x) {
  return backend_from_opaque(backend).sum_squares_f32(to_tensor_view(*x));
}

void plugin_relu(void *backend, const BackendTensorView *x,
                 const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).relu(to_tensor_view(*x), out_view);
}

void plugin_relu_backward(void *backend, const BackendTensorView *preact,
                          const BackendTensorView *dout,
                          const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  backend_from_opaque(backend).relu_backward(to_tensor_view(*preact),
                                           to_tensor_view(*dout), dx_view);
}

void plugin_relu_backward_inplace(void *backend,
                                  const BackendTensorView *preact,
                                  const BackendTensorView *dout_dx) {
  TensorView dout_dx_view = to_tensor_view(*dout_dx);
  backend_from_opaque(backend).relu_backward_inplace(to_tensor_view(*preact),
                                                   dout_dx_view);
}

void plugin_row_sum(void *backend, const BackendTensorView *x,
                    const BackendTensorView *out_1xC) {
  TensorView out_view = to_tensor_view(*out_1xC);
  backend_from_opaque(backend).row_sum(to_tensor_view(*x), out_view);
}

void plugin_matmul(void *backend, const BackendTensorView *a,
                   const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).matmul(to_tensor_view(*a), to_tensor_view(*b),
                                    out_view);
}

void plugin_matmul_left_transposed(void *backend, const BackendTensorView *a,
                                   const BackendTensorView *b,
                                   const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).matmul_left_transposed(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_matmul_right_transposed(void *backend, const BackendTensorView *a,
                                    const BackendTensorView *b,
                                    const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).matmul_right_transposed(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_transpose(void *backend, const BackendTensorView *x,
                      const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).transpose(to_tensor_view(*x), out_view);
}

void plugin_layernorm_forward(void *backend, const BackendTensorView *x,
                              const BackendTensorView *gamma_1xC,
                              const BackendTensorView *beta_1xC,
                              const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).layernorm_forward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC),
      to_tensor_view(*beta_1xC), out_view);
}

void plugin_layernorm_backward(void *backend, const BackendTensorView *x,
                               const BackendTensorView *gamma_1xC,
                               const BackendTensorView *dout,
                               const BackendTensorView *dx,
                               const BackendTensorView *dgamma_1xC,
                               const BackendTensorView *dbeta_1xC) {
  TensorView dx_view = to_tensor_view(*dx);
  TensorView dgamma_view = to_tensor_view(*dgamma_1xC);
  TensorView dbeta_view = to_tensor_view(*dbeta_1xC);
  backend_from_opaque(backend).layernorm_backward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC), to_tensor_view(*dout),
      dx_view, dgamma_view, dbeta_view);
}

void plugin_embedding_lookup(void *backend, const BackendTensorView *table,
                             const BackendTensorView *ids,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).embedding_lookup(
      to_tensor_view(*table), to_tensor_view(*ids), out_view);
}

void plugin_accumulate_embedding_grads(void *backend,
                                       const BackendTensorView *ids,
                                       const BackendTensorView *d_cur,
                                       const BackendTensorView *d_tok,
                                       const BackendTensorView *d_pos) {
  TensorView d_tok_view = to_tensor_view(*d_tok);
  TensorView d_pos_view = to_tensor_view(*d_pos);
  backend_from_opaque(backend).accumulate_embedding_grads(
      to_tensor_view(*ids), to_tensor_view(*d_cur), d_tok_view, d_pos_view);
}

void plugin_cross_entropy_mean(void *backend, const BackendTensorView *logits,
                               const BackendTensorView *targets,
                               const BackendTensorView *out_loss) {
  TensorView out_loss_view = to_tensor_view(*out_loss);
  backend_from_opaque(backend).cross_entropy_mean(
      to_tensor_view(*logits), to_tensor_view(*targets), out_loss_view);
}

void plugin_cross_entropy_mean_backward_inplace(
    void *backend, const BackendTensorView *logits,
    const BackendTensorView *targets, const BackendTensorView *out_loss) {
  TensorView logits_view = to_tensor_view(*logits);
  TensorView out_loss_view = to_tensor_view(*out_loss);
  backend_from_opaque(backend).cross_entropy_mean_backward_inplace(
      logits_view, to_tensor_view(*targets), out_loss_view);
}

float plugin_read_scalar_f32(void *backend, const BackendTensorView *x) {
  return backend_from_opaque(backend).read_scalar_f32(to_tensor_view(*x));
}

void plugin_backward_from_logits_targets(void *backend,
                                         const BackendTensorView *logits,
                                         const BackendTensorView *targets) {
  TensorView logits_view = to_tensor_view(*logits);
  backend_from_opaque(backend).backward_from_logits_targets(
      logits_view, to_tensor_view(*targets));
}

void plugin_softmax_rows(void *backend, const BackendTensorView *x,
                         const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).softmax_rows(to_tensor_view(*x), out_view);
}

void plugin_softmax_backward_rows(void *backend,
                                  const BackendTensorView *softmax,
                                  const BackendTensorView *dout,
                                  const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  backend_from_opaque(backend).softmax_backward_rows(to_tensor_view(*softmax),
                                                   to_tensor_view(*dout),
                                                   dx_view);
}

void plugin_scaled_causal_softmax_rows(void *backend,
                                       const BackendTensorView *scores,
                                       float scale,
                                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  backend_from_opaque(backend).scaled_causal_softmax_rows(
      to_tensor_view(*scores), scale, out_view);
}

void plugin_softmax_backward_causal_rows(void *backend,
                                         const BackendTensorView *softmax,
                                         const BackendTensorView *dout,
                                         const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  backend_from_opaque(backend).softmax_backward_causal_rows(
      to_tensor_view(*softmax), to_tensor_view(*dout), dx_view);
}

void plugin_apply_causal_mask_inplace(void *backend,
                                      const BackendTensorView *scores,
                                      float neg_inf) {
  TensorView scores_view = to_tensor_view(*scores);
  backend_from_opaque(backend).apply_causal_mask_inplace(scores_view, neg_inf);
}

void plugin_adamw_step(void *backend, const BackendTensorView *params,
                       const BackendTensorView *grads,
                       const BackendTensorView *m,
                       const BackendTensorView *v, uint64_t step,
                       float learning_rate, float beta1, float beta2,
                       float weight_decay, uint32_t apply_weight_decay) {
  TensorView params_view = to_tensor_view(*params);
  TensorView m_view = to_tensor_view(*m);
  TensorView v_view = to_tensor_view(*v);
  backend_from_opaque(backend).adamw_step(params_view, to_tensor_view(*grads),
                                        m_view, v_view, step, learning_rate,
                                        beta1, beta2, weight_decay,
                                        apply_weight_decay != 0);
}

uint32_t plugin_is_file2device_read_supported(void *backend) {
  return backend_from_opaque(backend).is_file2device_read_supported() ? 1u : 0u;
}

void plugin_read_file2device(void *backend, const char *path, void *dst,
                             uint64_t size, uint64_t file_offset) {
  backend_from_opaque(backend).read_file2device(path, dst, size, file_offset);
}

BackendMemoryInfo plugin_memory_info(void *backend) {
  const DeviceMemoryInfo info = backend_from_opaque(backend).memory_info();
  return BackendMemoryInfo{info.available ? 1u : 0u, info.free_bytes,
                           info.total_bytes};
}

const BackendApiV1 kBackendApi = {
    kBackendApiVersion,
    &plugin_create,
    &plugin_destroy,
    &plugin_device,
    &plugin_alloc,
    &plugin_free,
    &plugin_copy_host2device,
    &plugin_copy_device2host,
    &plugin_copy,
    &plugin_fill,
    &plugin_add,
    &plugin_add_inplace,
    &plugin_add_bias_rowwise,
    &plugin_add_bias_relu_rowwise,
    &plugin_add_bias_relu_rowwise_inplace,
    &plugin_mul_scalar,
    &plugin_sum_squares_f32,
    &plugin_relu,
    &plugin_relu_backward,
    &plugin_relu_backward_inplace,
    &plugin_row_sum,
    &plugin_matmul,
    &plugin_matmul_left_transposed,
    &plugin_matmul_right_transposed,
    &plugin_transpose,
    &plugin_layernorm_forward,
    &plugin_layernorm_backward,
    &plugin_embedding_lookup,
    &plugin_accumulate_embedding_grads,
    &plugin_cross_entropy_mean,
    &plugin_cross_entropy_mean_backward_inplace,
    &plugin_read_scalar_f32,
    &plugin_backward_from_logits_targets,
    &plugin_softmax_rows,
    &plugin_softmax_backward_rows,
    &plugin_scaled_causal_softmax_rows,
    &plugin_softmax_backward_causal_rows,
    &plugin_apply_causal_mask_inplace,
    &plugin_adamw_step,
    &plugin_is_file2device_read_supported,
    &plugin_read_file2device,
    &plugin_memory_info,
};

} // namespace

const BackendApiV1 *backend_api() { return &kBackendApi; }

} // namespace cuda_cublas_plugin

extern "C" const BackendApiV1 *litnice_backend_get_api() {
  return cuda_cublas_plugin::backend_api();
}
