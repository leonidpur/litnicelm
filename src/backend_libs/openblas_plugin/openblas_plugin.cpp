#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"

#include <cblas.h>

#include <memory>
#include <stdexcept>

namespace {

TensorView to_tensor_view(const BackendTensorView &abi) {
  return TensorView(static_cast<Device>(abi.device),
                    static_cast<DType>(abi.dtype), abi.data,
                    Shape2D{abi.rows, abi.cols}, abi.stride_c_bytes,
                    abi.stride_r_bytes);
}

class OpenBlasPluginBackend final : public DeviceBackend {
public:
  void *alloc(uint64_t bytes, uint32_t alignment) override {
    return cpu_backend_.alloc(bytes, alignment);
  }

  void free(void *ptr) override { cpu_backend_.free(ptr); }

  void copy_host2device(void *dst, const void *src, uint64_t bytes) override {
    cpu_backend_.copy_host2device(dst, src, bytes);
  }

  void copy_device2host(void *dst, const void *src, uint64_t bytes) override {
    cpu_backend_.copy_device2host(dst, src, bytes);
  }

  void copy(const TensorView &src, TensorView &dst) override {
    cpu_backend_.copy(src, dst);
  }

  void fill(TensorView &t, float v) override { cpu_backend_.fill(t, v); }

  void add(const TensorView &a, const TensorView &b, TensorView &out) override {
    cpu_backend_.add(a, b, out);
  }

  void add_inplace(TensorView &a, const TensorView &b) override {
    cpu_backend_.add_inplace(a, b);
  }

  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override {
    cpu_backend_.add_bias_rowwise(x, bias_1xC, out);
  }

  void mul_scalar(const TensorView &x, float s, TensorView &out) override {
    cpu_backend_.mul_scalar(x, s, out);
  }

  void relu(const TensorView &x, TensorView &out) override {
    cpu_backend_.relu(x, out);
  }

  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override {
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32 ||
        out.dtype() != DType::F32 || !a.is_contiguous_row_major() ||
        !b.is_contiguous_row_major() || !out.is_contiguous_row_major()) {
      cpu_backend_.matmul(a, b, out);
      return;
    }

    const int m = static_cast<int>(a.shape().r);
    const int k = static_cast<int>(a.shape().c);
    const int n = static_cast<int>(b.shape().c);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), k,
                static_cast<const float *>(b.data()), n, 0.0f,
                static_cast<float *>(out.data()), n);
  }

  void matmul_left_transposed(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32 ||
        out.dtype() != DType::F32 || !a.is_contiguous_row_major() ||
        !b.is_contiguous_row_major() || !out.is_contiguous_row_major()) {
      cpu_backend_.matmul_left_transposed(a, b, out);
      return;
    }

    const int m = static_cast<int>(a.shape().c);
    const int k = static_cast<int>(a.shape().r);
    const int n = static_cast<int>(b.shape().c);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), m,
                static_cast<const float *>(b.data()), n, 0.0f,
                static_cast<float *>(out.data()), n);
  }

  void matmul_right_transposed(const TensorView &a, const TensorView &b,
                               TensorView &out) override {
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32 ||
        out.dtype() != DType::F32 || !a.is_contiguous_row_major() ||
        !b.is_contiguous_row_major() || !out.is_contiguous_row_major()) {
      cpu_backend_.matmul_right_transposed(a, b, out);
      return;
    }

    const int m = static_cast<int>(a.shape().r);
    const int k = static_cast<int>(a.shape().c);
    const int n = static_cast<int>(b.shape().r);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), k,
                static_cast<const float *>(b.data()), k, 0.0f,
                static_cast<float *>(out.data()), n);
  }

  void transpose(const TensorView &x, TensorView &out) override {
    cpu_backend_.transpose(x, out);
  }

  void layernorm_forward(const TensorView &x, const TensorView &gamma_1xC,
                         const TensorView &beta_1xC, TensorView &out) override {
    cpu_backend_.layernorm_forward(x, gamma_1xC, beta_1xC, out);
  }

  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) override {
    cpu_backend_.layernorm_backward(x, gamma_1xC, dout, dx, dgamma_1xC,
                                    dbeta_1xC);
  }

  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) override {
    cpu_backend_.embedding_lookup(table, ids, out);
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    cpu_backend_.cross_entropy_mean(logits, targets, out_loss);
  }

  float read_scalar_f32(const TensorView &x) override {
    return cpu_backend_.read_scalar_f32(x);
  }

  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override {
    cpu_backend_.backward_from_logits_targets(logits, targets);
  }

  void softmax_rows(const TensorView &x, TensorView &out) override {
    cpu_backend_.softmax_rows(x, out);
  }

  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override {
    cpu_backend_.apply_causal_mask_inplace(scores, neg_inf);
  }

  bool is_file2device_read_supported() const override {
    return cpu_backend_.is_file2device_read_supported();
  }

  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override {
    cpu_backend_.read_file2device(path, dst, size, file_offset);
  }

private:
  CpuBackend cpu_backend_;
};

OpenBlasPluginBackend &to_openblas_backend(void *backend) {
  if (backend == nullptr) {
    throw std::runtime_error("openblas_plugin: null backend instance");
  }
  return *reinterpret_cast<OpenBlasPluginBackend *>(backend);
}

void *plugin_create(uint32_t device) {
  if (static_cast<Device>(device) != Device::CPU) {
    throw std::runtime_error("openblas_plugin: only cpu device is supported");
  }
  return new OpenBlasPluginBackend();
}

void plugin_destroy(void *backend) {
  delete reinterpret_cast<OpenBlasPluginBackend *>(backend);
}

void *plugin_alloc(void *backend, uint64_t bytes, uint32_t alignment) {
  return to_openblas_backend(backend).alloc(bytes, alignment);
}

void plugin_free(void *backend, void *ptr) {
  to_openblas_backend(backend).free(ptr);
}

void plugin_copy_host2device(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  to_openblas_backend(backend).copy_host2device(dst, src, bytes);
}

void plugin_copy_device2host(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  to_openblas_backend(backend).copy_device2host(dst, src, bytes);
}

void plugin_copy(void *backend, const BackendTensorView *src,
                 const BackendTensorView *dst) {
  TensorView dst_view = to_tensor_view(*dst);
  to_openblas_backend(backend).copy(to_tensor_view(*src), dst_view);
}

void plugin_fill(void *backend, const BackendTensorView *t, float v) {
  TensorView t_view = to_tensor_view(*t);
  to_openblas_backend(backend).fill(t_view, v);
}

void plugin_add(void *backend, const BackendTensorView *a,
                const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).add(to_tensor_view(*a), to_tensor_view(*b),
                                   out_view);
}

void plugin_add_inplace(void *backend, const BackendTensorView *a,
                        const BackendTensorView *b) {
  TensorView a_view = to_tensor_view(*a);
  to_openblas_backend(backend).add_inplace(a_view, to_tensor_view(*b));
}

void plugin_add_bias_rowwise(void *backend, const BackendTensorView *x,
                             const BackendTensorView *bias_1xC,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).add_bias_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_mul_scalar(void *backend, const BackendTensorView *x, float s,
                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).mul_scalar(to_tensor_view(*x), s, out_view);
}

void plugin_relu(void *backend, const BackendTensorView *x,
                 const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).relu(to_tensor_view(*x), out_view);
}

void plugin_matmul(void *backend, const BackendTensorView *a,
                   const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).matmul(to_tensor_view(*a), to_tensor_view(*b),
                                      out_view);
}

void plugin_matmul_left_transposed(void *backend, const BackendTensorView *a,
                                   const BackendTensorView *b,
                                   const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).matmul_left_transposed(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_matmul_right_transposed(void *backend, const BackendTensorView *a,
                                    const BackendTensorView *b,
                                    const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).matmul_right_transposed(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_transpose(void *backend, const BackendTensorView *x,
                      const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).transpose(to_tensor_view(*x), out_view);
}

void plugin_layernorm_forward(void *backend, const BackendTensorView *x,
                              const BackendTensorView *gamma_1xC,
                              const BackendTensorView *beta_1xC,
                              const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).layernorm_forward(
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
  to_openblas_backend(backend).layernorm_backward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC), to_tensor_view(*dout),
      dx_view, dgamma_view, dbeta_view);
}

void plugin_embedding_lookup(void *backend, const BackendTensorView *table,
                             const BackendTensorView *ids,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).embedding_lookup(
      to_tensor_view(*table), to_tensor_view(*ids), out_view);
}

void plugin_cross_entropy_mean(void *backend, const BackendTensorView *logits,
                               const BackendTensorView *targets,
                               const BackendTensorView *out_loss) {
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_openblas_backend(backend).cross_entropy_mean(
      to_tensor_view(*logits), to_tensor_view(*targets), out_loss_view);
}

float plugin_read_scalar_f32(void *backend, const BackendTensorView *x) {
  return to_openblas_backend(backend).read_scalar_f32(to_tensor_view(*x));
}

void plugin_backward_from_logits_targets(void *backend,
                                         const BackendTensorView *logits,
                                         const BackendTensorView *targets) {
  TensorView logits_view = to_tensor_view(*logits);
  to_openblas_backend(backend).backward_from_logits_targets(
      logits_view, to_tensor_view(*targets));
}

void plugin_softmax_rows(void *backend, const BackendTensorView *x,
                         const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).softmax_rows(to_tensor_view(*x), out_view);
}

void plugin_apply_causal_mask_inplace(void *backend,
                                      const BackendTensorView *scores,
                                      float neg_inf) {
  TensorView scores_view = to_tensor_view(*scores);
  to_openblas_backend(backend).apply_causal_mask_inplace(scores_view, neg_inf);
}

uint32_t plugin_is_file2device_read_supported(void *backend) {
  return to_openblas_backend(backend).is_file2device_read_supported() ? 1u : 0u;
}

void plugin_read_file2device(void *backend, const char *path, void *dst,
                             uint64_t size, uint64_t file_offset) {
  to_openblas_backend(backend).read_file2device(path, dst, size, file_offset);
}

const BackendApiV1 kBackendApi = {
    kBackendApiVersion,
    &plugin_create,
    &plugin_destroy,
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
    &plugin_relu,
    &plugin_matmul,
    &plugin_matmul_left_transposed,
    &plugin_matmul_right_transposed,
    &plugin_transpose,
    &plugin_layernorm_forward,
    &plugin_layernorm_backward,
    &plugin_embedding_lookup,
    &plugin_cross_entropy_mean,
    &plugin_read_scalar_f32,
    &plugin_backward_from_logits_targets,
    &plugin_softmax_rows,
    &plugin_apply_causal_mask_inplace,
    &plugin_is_file2device_read_supported,
    &plugin_read_file2device,
};

} // namespace

extern "C" const BackendApiV1 *litnice_backend_get_api() { return &kBackendApi; }
