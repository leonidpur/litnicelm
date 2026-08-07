#include "backend_plugin_api.hpp"
#include "device_backend.hpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void validate_backend_tensor_view(const BackendTensorView &abi, const char *who) {
  if (abi.rank > kBackendTensorMaxRank) {
    throw std::runtime_error(std::string(who) + ": rank exceeds backend max");
  }
  if (abi.rank == 0) {
    if (abi.rows != 0 || abi.cols != 0) {
      throw std::runtime_error(std::string(who) +
                               ": rank-0 tensor must not advertise 2D extents");
    }
    return;
  }
  if (abi.rank >= 1 && abi.dims[0] <= 0) {
    throw std::runtime_error(std::string(who) + ": invalid leading dim");
  }
  if (abi.rank >= 2 && abi.dims[1] <= 0) {
    throw std::runtime_error(std::string(who) + ": invalid second dim");
  }
}

TensorView to_tensor_view(const BackendTensorView &abi) {
  validate_backend_tensor_view(abi, "backend plugin cpu");
  Shape logical_shape{};
  if (abi.rank > 0) {
    std::vector<int64_t> dims;
    dims.reserve(abi.rank);
    for (uint32_t i = 0; i < abi.rank && i < kBackendTensorMaxRank; ++i) {
      dims.push_back(abi.dims[i]);
    }
    logical_shape = Shape(dims);
  }
  const Shape shape = logical_shape.rank() == 0 ? Shape{abi.rows, abi.cols}
                                                : logical_shape;
  std::array<int64_t, kMaxTensorRank> strides{};
  for (uint32_t i = 0; i < abi.rank && i < kBackendTensorMaxRank; ++i) {
    strides[i] = abi.strides_bytes[i];
  }
  return TensorView(static_cast<Device>(abi.device),
                    static_cast<DType>(abi.dtype), abi.data, shape, strides);
}

CpuBackend &to_cpu_backend(void *backend) {
  if (backend == nullptr) {
    throw std::runtime_error("backend plugin: null backend instance");
  }
  return *reinterpret_cast<CpuBackend *>(backend);
}

void *plugin_create(uint32_t device) {
  if (static_cast<Device>(device) != Device::CPU) {
    throw std::runtime_error("backend plugin: CPU plugin only supports cpu device");
  }
  return new CpuBackend();
}

void plugin_destroy(void *backend) { delete reinterpret_cast<CpuBackend *>(backend); }

uint32_t plugin_device(void *backend) {
  return static_cast<uint32_t>(to_cpu_backend(backend).device());
}

void *plugin_alloc(void *backend, uint64_t bytes, uint32_t alignment) {
  return to_cpu_backend(backend).alloc(bytes, alignment);
}

void plugin_free(void *backend, void *ptr) { to_cpu_backend(backend).free(ptr); }

void plugin_copy_host2device(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  to_cpu_backend(backend).copy_host2device(dst, src, bytes);
}

void plugin_copy_device2host(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  to_cpu_backend(backend).copy_device2host(dst, src, bytes);
}

void plugin_copy(void *backend, const BackendTensorView *src,
                 const BackendTensorView *dst) {
  TensorView dst_view = to_tensor_view(*dst);
  to_cpu_backend(backend).copy(to_tensor_view(*src), dst_view);
}

void plugin_fill(void *backend, const BackendTensorView *t, float v) {
  TensorView t_view = to_tensor_view(*t);
  to_cpu_backend(backend).fill(t_view, v);
}

void plugin_add(void *backend, const BackendTensorView *a,
                const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).add(to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_add_inplace(void *backend, const BackendTensorView *a,
                        const BackendTensorView *b) {
  TensorView a_view = to_tensor_view(*a);
  to_cpu_backend(backend).add_inplace(a_view, to_tensor_view(*b));
}

void plugin_add_bias_rowwise(void *backend, const BackendTensorView *x,
                             const BackendTensorView *bias_1xC,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).add_bias_rowwise(to_tensor_view(*x),
                                           to_tensor_view(*bias_1xC), out_view);
}

void plugin_mul_scalar(void *backend, const BackendTensorView *x, float s,
                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).mul_scalar(to_tensor_view(*x), s, out_view);
}

float plugin_sum_squares_f32(void *backend, const BackendTensorView *x) {
  return to_cpu_backend(backend).sum_squares_f32(to_tensor_view(*x));
}

void plugin_relu(void *backend, const BackendTensorView *x,
                 const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).relu(to_tensor_view(*x), out_view);
}

void plugin_relu_backward(void *backend, const BackendTensorView *preact,
                          const BackendTensorView *dout,
                          const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_cpu_backend(backend).relu_backward(to_tensor_view(*preact),
                                        to_tensor_view(*dout), dx_view);
}

void plugin_row_sum(void *backend, const BackendTensorView *x,
                    const BackendTensorView *out_1xC) {
  TensorView out_view = to_tensor_view(*out_1xC);
  to_cpu_backend(backend).row_sum(to_tensor_view(*x), out_view);
}

void plugin_matmul(void *backend, const BackendTensorView *a,
                   const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).matmul(to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_matmul_left_transposed(void *backend, const BackendTensorView *a,
                                   const BackendTensorView *b,
                                   const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).matmul_left_transposed(to_tensor_view(*a),
                                                 to_tensor_view(*b), out_view);
}

void plugin_matmul_right_transposed(void *backend, const BackendTensorView *a,
                                    const BackendTensorView *b,
                                    const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).matmul_right_transposed(to_tensor_view(*a),
                                                  to_tensor_view(*b), out_view);
}

void plugin_transpose(void *backend, const BackendTensorView *x,
                      const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).transpose(to_tensor_view(*x), out_view);
}

void plugin_layernorm_forward(void *backend, const BackendTensorView *x,
                              const BackendTensorView *gamma_1xC,
                              const BackendTensorView *beta_1xC,
                              const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).layernorm_forward(to_tensor_view(*x),
                                            to_tensor_view(*gamma_1xC),
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
  to_cpu_backend(backend).layernorm_backward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC), to_tensor_view(*dout),
      dx_view, dgamma_view, dbeta_view);
}

void plugin_embedding_lookup(void *backend, const BackendTensorView *table,
                             const BackendTensorView *ids,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).embedding_lookup(to_tensor_view(*table),
                                           to_tensor_view(*ids), out_view);
}

void plugin_accumulate_embedding_grads(void *backend,
                                       const BackendTensorView *ids,
                                       const BackendTensorView *d_cur,
                                       const BackendTensorView *d_tok,
                                       const BackendTensorView *d_pos) {
  TensorView d_tok_view = to_tensor_view(*d_tok);
  TensorView d_pos_view = to_tensor_view(*d_pos);
  to_cpu_backend(backend).accumulate_embedding_grads(
      to_tensor_view(*ids), to_tensor_view(*d_cur), d_tok_view, d_pos_view);
}

void plugin_cross_entropy_mean(void *backend, const BackendTensorView *logits,
                               const BackendTensorView *targets,
                               const BackendTensorView *out_loss) {
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_cpu_backend(backend).cross_entropy_mean(to_tensor_view(*logits),
                                             to_tensor_view(*targets),
                                             out_loss_view);
}

void plugin_cross_entropy_mean_backward_inplace(
    void *backend, const BackendTensorView *logits,
    const BackendTensorView *targets, const BackendTensorView *out_loss) {
  TensorView logits_view = to_tensor_view(*logits);
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_cpu_backend(backend).cross_entropy_mean_backward_inplace(
      logits_view, to_tensor_view(*targets), out_loss_view);
}

float plugin_read_scalar_f32(void *backend, const BackendTensorView *x) {
  return to_cpu_backend(backend).read_scalar_f32(to_tensor_view(*x));
}

void plugin_backward_from_logits_targets(void *backend,
                                         const BackendTensorView *logits,
                                         const BackendTensorView *targets) {
  TensorView logits_view = to_tensor_view(*logits);
  to_cpu_backend(backend).backward_from_logits_targets(logits_view,
                                                       to_tensor_view(*targets));
}

void plugin_softmax_rows(void *backend, const BackendTensorView *x,
                         const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cpu_backend(backend).softmax_rows(to_tensor_view(*x), out_view);
}

void plugin_softmax_backward_rows(void *backend,
                                  const BackendTensorView *softmax,
                                  const BackendTensorView *dout,
                                  const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_cpu_backend(backend).softmax_backward_rows(to_tensor_view(*softmax),
                                                to_tensor_view(*dout),
                                                dx_view);
}

void plugin_apply_causal_mask_inplace(void *backend,
                                      const BackendTensorView *scores,
                                      float neg_inf) {
  TensorView scores_view = to_tensor_view(*scores);
  to_cpu_backend(backend).apply_causal_mask_inplace(scores_view, neg_inf);
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
  to_cpu_backend(backend).adamw_step(params_view, to_tensor_view(*grads), m_view,
                                     v_view, step, learning_rate, beta1, beta2,
                                     weight_decay,
                                     apply_weight_decay != 0);
}

uint32_t plugin_is_file2device_read_supported(void *backend) {
  return to_cpu_backend(backend).is_file2device_read_supported() ? 1u : 0u;
}

void plugin_read_file2device(void *backend, const char *path, void *dst,
                             uint64_t size, uint64_t file_offset) {
  to_cpu_backend(backend).read_file2device(path, dst, size, file_offset);
}

BackendMemoryInfo plugin_memory_info(void *backend) {
  (void)backend;
  return BackendMemoryInfo{0u, 0u, 0u};
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
    &plugin_mul_scalar,
    &plugin_sum_squares_f32,
    &plugin_relu,
    &plugin_relu_backward,
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
    &plugin_apply_causal_mask_inplace,
    &plugin_adamw_step,
    &plugin_is_file2device_read_supported,
    &plugin_read_file2device,
    &plugin_memory_info,
};

} // namespace

extern "C" const BackendApiV1 *litnice_backend_get_api() { 
  return &kBackendApi; 
}
