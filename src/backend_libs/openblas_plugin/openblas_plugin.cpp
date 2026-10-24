#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"

#include <litnicelm_utils/view_geometry.hpp>
#include <litnicelm_utils/plugin_log.h>

#include <cblas.h>

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

inline constexpr EventDef kCpuFallback{
    "openblas_cpu_fallback", EventType::Warning,
    "[OpenBLAS] TEMP fallback to cpu_backend_ in"};

void warn_cpu_fallback(const char *op) { plugin_log(kCpuFallback, op); }

const TensorView &to_tensor_view(const TensorViewStruct &abi) {
  return as_tensor_view(abi, "openblas plugin");
}

bool is_blas_compatible_f32_row_major(const TensorView &view) {
  return view.rank() == 2 && view.device() == Device::CPU &&
         view.dtype() == DType::F32 &&
         view.data() != nullptr &&
         view.stride_bytes(view.rank() - 1) == static_cast<int64_t>(sizeof(float)) &&
         view.stride_bytes(view.rank() - 2) >=
             view.shape().dim(1) * static_cast<int64_t>(sizeof(float));
}

bool is_contiguous_f32_row_major(const TensorView &view) {
  return view.device() == Device::CPU && view.dtype() == DType::F32 &&
         view.data() != nullptr && view.is_contiguous_row_major();
}

bool is_batched_blas_compatible_f32_row_major(const TensorView &view) {
  if (view.rank() < 3 || view.device() != Device::CPU ||
      view.dtype() != DType::F32 || view.data() == nullptr) {
    return false;
  }
  if (view.stride_bytes(view.rank() - 1) !=
      static_cast<int64_t>(sizeof(float))) {
    return false;
  }
  if (view.stride_bytes(view.rank() - 2) <
      view.shape().dim(view.rank() - 1) *
          static_cast<int64_t>(sizeof(float))) {
    return false;
  }
  for (size_t axis = view.rank() - 2; axis > 0; --axis) {
    const int64_t required = view.dim(axis) * view.stride_bytes(axis);
    if (view.stride_bytes(axis - 1) < required) {
      return false;
    }
  }
  return true;
}

float *prefix_matrix_ptr(TensorView &view, uint64_t prefix, int64_t rows,
                         int64_t cols) {
  (void)rows;
  (void)cols;
  auto *base = reinterpret_cast<uint8_t *>(view.data());
  return reinterpret_cast<float *>(
      base + view_geometry::prefix_byte_offset(view, prefix, 2));
}

const float *prefix_matrix_ptr(const TensorView &view, uint64_t prefix,
                               int64_t rows, int64_t cols) {
  (void)rows;
  (void)cols;
  auto *base = reinterpret_cast<const uint8_t *>(view.data());
  return reinterpret_cast<const float *>(
      base + view_geometry::prefix_byte_offset(view, prefix, 2));
}

int leading_dim_f32(const TensorView &view) {
  return static_cast<int>(view.stride_bytes(view.rank() - 2) / sizeof(float));
}

class OpenBlasPluginBackend final : public DeviceBackend {
public:
  Device device() const override { return Device::CPU; }

  void *alloc(uint64_t bytes, uint32_t alignment) override {
    return cpu_backend_.alloc(bytes, alignment);
  }

  void free(void *ptr) override { cpu_backend_.free(ptr); }

  DeviceMemoryInfo memory_info() const override { return {}; }

  uint32_t precision_variants() const override {
    return kPrecisionVariantF32;
  }

  void copy_host2device(void *dst, const void *src, uint64_t bytes) override {
    cpu_backend_.copy_host2device(dst, src, bytes);
  }

  void copy_device2host(void *dst, const void *src, uint64_t bytes) override {
    cpu_backend_.copy_device2host(dst, src, bytes);
  }

  void copy(const TensorView &src, TensorView &dst) override {
    cpu_backend_.copy(src, dst);
  }

  void cast(const TensorView &src, TensorView &dst) override {
    cpu_backend_.cast(src, dst);
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

  void add_bias_relu_rowwise(const TensorView &x, const TensorView &bias_1xC,
                             TensorView &out) override {
    cpu_backend_.add_bias_relu_rowwise(x, bias_1xC, out);
  }

  void add_bias_relu_rowwise_inplace(TensorView &x,
                                     const TensorView &bias_1xC) override {
    cpu_backend_.add_bias_relu_rowwise_inplace(x, bias_1xC);
  }

  void mul_scalar(const TensorView &x, float s, TensorView &out) override {
    cpu_backend_.mul_scalar(x, s, out);
  }

  float sum_squares_f32(const TensorView &x) override {
    return cpu_backend_.sum_squares_f32(x);
  }

  void relu(const TensorView &x, TensorView &out) override {
    cpu_backend_.relu(x, out);
  }

  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) override {
    cpu_backend_.relu_backward(preact, dout, dx);
  }

  void relu_backward_inplace(const TensorView &preact,
                             TensorView &dout_dx) override {
    cpu_backend_.relu_backward_inplace(preact, dout_dx);
  }

  void row_sum(const TensorView &x, TensorView &out_1xC) override {
    cpu_backend_.row_sum(x, out_1xC);
  }

  void gemm(const TensorView &a, const TensorView &b,
            TensorView &out) override {
    if (!is_blas_compatible_f32_row_major(a) ||
        !is_blas_compatible_f32_row_major(b) ||
        !is_blas_compatible_f32_row_major(out)) {
      warn_cpu_fallback("gemm");
      cpu_backend_.gemm(a, b, out);
      return;
    }

    const int m = static_cast<int>(a.shape().dim(0));
    const int k = static_cast<int>(a.shape().dim(1));
    const int n = static_cast<int>(b.shape().dim(1));
    const int lda = leading_dim_f32(a);
    const int ldb = leading_dim_f32(b);
    const int ldc = leading_dim_f32(out);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), lda,
                static_cast<const float *>(b.data()), ldb, 0.0f,
                static_cast<float *>(out.data()), ldc);
  }

  void gemm_ranked_matrix_rhs(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    if (!is_batched_blas_compatible_f32_row_major(a) ||
        !is_blas_compatible_f32_row_major(b) ||
        !is_batched_blas_compatible_f32_row_major(out)) {
      warn_cpu_fallback("gemm_ranked_matrix_rhs");
      cpu_backend_.gemm_ranked_matrix_rhs(a, b, out);
      return;
    }
    const int m = static_cast<int>(view_geometry::prefix_count(a, 2) *
                                   static_cast<uint64_t>(a.dim(a.rank() - 2)));
    const int k = static_cast<int>(a.dim(a.rank() - 1));
    const int n = static_cast<int>(b.dim(1));
    const int lda = leading_dim_f32(a);
    const int ldb = leading_dim_f32(b);
    const int ldc = leading_dim_f32(out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), lda,
                static_cast<const float *>(b.data()), ldb, 0.0f,
                static_cast<float *>(out.data()), ldc);
  }

  void gemm_ranked_matrix_rhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) override {
    if (!is_batched_blas_compatible_f32_row_major(a) ||
        !is_blas_compatible_f32_row_major(b) ||
        !is_batched_blas_compatible_f32_row_major(out)) {
      warn_cpu_fallback("gemm_ranked_matrix_rhs_t");
      cpu_backend_.gemm_ranked_matrix_rhs_t(a, b, out);
      return;
    }
    const int m = static_cast<int>(view_geometry::prefix_count(a, 2) *
                                   static_cast<uint64_t>(a.dim(a.rank() - 2)));
    const int k = static_cast<int>(a.dim(a.rank() - 1));
    const int n = static_cast<int>(b.dim(0));
    const int lda = leading_dim_f32(a);
    const int ldb = leading_dim_f32(b);
    const int ldc = leading_dim_f32(out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), lda,
                static_cast<const float *>(b.data()), ldb, 0.0f,
                static_cast<float *>(out.data()), ldc);
  }

  void gemm_ranked_reduce_lhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) override {
    if (!is_contiguous_f32_row_major(a) || !is_contiguous_f32_row_major(b) ||
        !is_contiguous_f32_row_major(out)) {
      warn_cpu_fallback("gemm_ranked_reduce_lhs_t");
      cpu_backend_.gemm_ranked_reduce_lhs_t(a, b, out);
      return;
    }
    const int m = static_cast<int>(a.dim(a.rank() - 1));
    const int k = static_cast<int>(view_geometry::prefix_count(a, 2) *
                                   static_cast<uint64_t>(a.dim(a.rank() - 2)));
    const int n = static_cast<int>(b.dim(b.rank() - 1));
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), m,
                static_cast<const float *>(b.data()), n, 0.0f,
                static_cast<float *>(out.data()), n);
  }

  void gemm_batched(const TensorView &a, const TensorView &b,
                    TensorView &out) override {
    if (is_batched_blas_compatible_f32_row_major(a) &&
        is_batched_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const uint64_t prefix_count = view_geometry::prefix_count(a, 2);
      const int m = static_cast<int>(a.dim(a.rank() - 2));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const int lda = leading_dim_f32(a);
      const int ldb = leading_dim_f32(b);
      const int ldc = leading_dim_f32(out);
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
            prefix_matrix_ptr(a, prefix, a.dim(a.rank() - 2), a.dim(a.rank() - 1)),
            lda,
            prefix_matrix_ptr(b, prefix, b.dim(b.rank() - 2), b.dim(b.rank() - 1)),
            ldb, 0.0f,
            prefix_matrix_ptr(out, prefix, out.dim(out.rank() - 2),
                              out.dim(out.rank() - 1)),
            ldc);
      }
      return;
    }
    warn_cpu_fallback("gemm_batched");
    cpu_backend_.gemm_batched(a, b, out);
  }

  void gemm_batched_lhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) override {
    if (is_batched_blas_compatible_f32_row_major(a) &&
        is_batched_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const uint64_t prefix_count = view_geometry::prefix_count(a, 2);
      const int m = static_cast<int>(a.dim(a.rank() - 1));
      const int k = static_cast<int>(a.dim(a.rank() - 2));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const int lda = leading_dim_f32(a);
      const int ldb = leading_dim_f32(b);
      const int ldc = leading_dim_f32(out);
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
            prefix_matrix_ptr(a, prefix, a.dim(a.rank() - 2), a.dim(a.rank() - 1)),
            lda,
            prefix_matrix_ptr(b, prefix, b.dim(b.rank() - 2), b.dim(b.rank() - 1)),
            ldb, 0.0f,
            prefix_matrix_ptr(out, prefix, out.dim(out.rank() - 2),
                              out.dim(out.rank() - 1)),
            ldc);
      }
      return;
    }
    warn_cpu_fallback("gemm_batched_lhs_t");
    cpu_backend_.gemm_batched_lhs_t(a, b, out);
  }

  void gemm_batched_rhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) override {
    if (is_batched_blas_compatible_f32_row_major(a) &&
        is_batched_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const uint64_t prefix_count = view_geometry::prefix_count(a, 2);
      const int m = static_cast<int>(a.dim(a.rank() - 2));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(b.rank() - 2));
      const int lda = leading_dim_f32(a);
      const int ldb = leading_dim_f32(b);
      const int ldc = leading_dim_f32(out);
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
            prefix_matrix_ptr(a, prefix, a.dim(a.rank() - 2), a.dim(a.rank() - 1)),
            lda,
            prefix_matrix_ptr(b, prefix, b.dim(b.rank() - 2), b.dim(b.rank() - 1)),
            ldb, 0.0f,
            prefix_matrix_ptr(out, prefix, out.dim(out.rank() - 2),
                              out.dim(out.rank() - 1)),
            ldc);
      }
      return;
    }
    warn_cpu_fallback("gemm_batched_rhs_t");
    cpu_backend_.gemm_batched_rhs_t(a, b, out);
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

  void accumulate_embedding_grads(const TensorView &ids,
                                  const TensorView &d_cur,
                                  TensorView &d_tok) override {
    cpu_backend_.accumulate_embedding_grads(ids, d_cur, d_tok);
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    cpu_backend_.cross_entropy_mean(logits, targets, out_loss);
  }

  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) override {
    cpu_backend_.cross_entropy_mean_backward_inplace(logits, targets, out_loss);
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

  void softmax_backward_rows(const TensorView &softmax, const TensorView &dout,
                             TensorView &dx) override {
    cpu_backend_.softmax_backward_rows(softmax, dout, dx);
  }

  void scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                  TensorView &out) override {
    cpu_backend_.scaled_causal_softmax_rows(scores, scale, out);
  }

  bool supports_exec_context_iteration() const override { return false; }

  void start_exec_context_iteration() override {
    throw std::runtime_error(
        "openblas_plugin: exec context iteration is not supported");
  }

  void finish_exec_context_iteration() override {
    throw std::runtime_error(
        "openblas_plugin: exec context iteration is not supported");
  }

  void start_exec_context_group() override {
    throw std::runtime_error(
        "openblas_plugin: exec context group is not supported");
  }

  void finish_exec_context_group() override {
    throw std::runtime_error(
        "openblas_plugin: exec context group is not supported");
  }

  void gemm_batched_exec_context(const TensorView &, const TensorView &,
                                 TensorView &) override {
    throw std::runtime_error(
        "openblas_plugin: exec context batched gemm is not supported");
  }

  void gemm_batched_rhs_t_exec_context(const TensorView &, const TensorView &,
                                       TensorView &) override {
    throw std::runtime_error(
        "openblas_plugin: exec context rhs-transposed batched gemm is not supported");
  }

  void gemm_batched_lhs_t_exec_context(const TensorView &, const TensorView &,
                                       TensorView &) override {
    throw std::runtime_error(
        "openblas_plugin: exec context lhs-transposed batched gemm is not supported");
  }

  void scaled_causal_softmax_rows_exec_context(
      const TensorView &, float, TensorView &) override {
    throw std::runtime_error(
        "openblas_plugin: exec context causal softmax is not supported");
  }

  void softmax_backward_causal_rows_exec_context(
      const TensorView &, const TensorView &, TensorView &) override {
    throw std::runtime_error(
        "openblas_plugin: exec context causal softmax backward is not supported");
  }

  void softmax_backward_causal_rows(const TensorView &softmax,
                                    const TensorView &dout,
                                    TensorView &dx) override {
    cpu_backend_.softmax_backward_causal_rows(softmax, dout, dx);
  }

  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override {
    cpu_backend_.apply_causal_mask_inplace(scores, neg_inf);
  }

  void rotary_embedding_inplace(TensorView &x, int64_t n_heads, float base,
                                bool inverse) override {
    cpu_backend_.rotary_embedding_inplace(x, n_heads, base, inverse);
  }

  void adamw_step(TensorView &params, const TensorView &grads, TensorView &m,
                  TensorView &v, uint64_t step, float learning_rate,
                  float beta1, float beta2, float weight_decay,
                  bool apply_weight_decay) override {
    cpu_backend_.adamw_step(params, grads, m, v, step, learning_rate, beta1,
                            beta2, weight_decay, apply_weight_decay);
  }

  bool is_file2device_read_supported() const override {
    return cpu_backend_.is_file2device_read_supported();
  }

  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override {
    cpu_backend_.read_file2device(path, dst, size, file_offset);
  }

private:
  DefaultCpuBackend cpu_backend_;
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

uint32_t plugin_device(void *backend) {
  return static_cast<uint32_t>(to_openblas_backend(backend).device());
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

void plugin_copy(void *backend, const TensorViewStruct *src,
                 const TensorViewStruct *dst) {
  TensorView dst_view = to_tensor_view(*dst);
  to_openblas_backend(backend).copy(to_tensor_view(*src), dst_view);
}

void plugin_fill(void *backend, const TensorViewStruct *t, float v) {
  TensorView t_view = to_tensor_view(*t);
  to_openblas_backend(backend).fill(t_view, v);
}

void plugin_add(void *backend, const TensorViewStruct *a,
                const TensorViewStruct *b, const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).add(to_tensor_view(*a), to_tensor_view(*b),
                                   out_view);
}

void plugin_add_inplace(void *backend, const TensorViewStruct *a,
                        const TensorViewStruct *b) {
  TensorView a_view = to_tensor_view(*a);
  to_openblas_backend(backend).add_inplace(a_view, to_tensor_view(*b));
}

void plugin_add_bias_rowwise(void *backend, const TensorViewStruct *x,
                             const TensorViewStruct *bias_1xC,
                             const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).add_bias_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_add_bias_relu_rowwise(void *backend, const TensorViewStruct *x,
                                  const TensorViewStruct *bias_1xC,
                                  const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).add_bias_relu_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_add_bias_relu_rowwise_inplace(void *backend,
                                          const TensorViewStruct *x,
                                          const TensorViewStruct *bias_1xC) {
  TensorView x_view = to_tensor_view(*x);
  to_openblas_backend(backend).add_bias_relu_rowwise_inplace(
      x_view, to_tensor_view(*bias_1xC));
}

void plugin_mul_scalar(void *backend, const TensorViewStruct *x, float s,
                       const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).mul_scalar(to_tensor_view(*x), s, out_view);
}

float plugin_sum_squares_f32(void *backend, const TensorViewStruct *x) {
  return to_openblas_backend(backend).sum_squares_f32(to_tensor_view(*x));
}

void plugin_relu(void *backend, const TensorViewStruct *x,
                 const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).relu(to_tensor_view(*x), out_view);
}

void plugin_relu_backward(void *backend, const TensorViewStruct *preact,
                          const TensorViewStruct *dout,
                          const TensorViewStruct *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).relu_backward(to_tensor_view(*preact),
                                             to_tensor_view(*dout), dx_view);
}

void plugin_relu_backward_inplace(void *backend,
                                  const TensorViewStruct *preact,
                                  const TensorViewStruct *dout_dx) {
  TensorView dout_dx_view = to_tensor_view(*dout_dx);
  to_openblas_backend(backend).relu_backward_inplace(to_tensor_view(*preact),
                                                     dout_dx_view);
}

void plugin_row_sum(void *backend, const TensorViewStruct *x,
                    const TensorViewStruct *out_1xC) {
  TensorView out_view = to_tensor_view(*out_1xC);
  to_openblas_backend(backend).row_sum(to_tensor_view(*x), out_view);
}

void plugin_gemm(void *backend, const TensorViewStruct *a,
                 const TensorViewStruct *b, const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm(to_tensor_view(*a), to_tensor_view(*b),
                                    out_view);
}

void plugin_gemm_ranked_matrix_rhs(void *backend, const TensorViewStruct *a,
                                   const TensorViewStruct *b,
                                   const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_ranked_matrix_rhs(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_ranked_matrix_rhs_t(void *backend, const TensorViewStruct *a,
                                     const TensorViewStruct *b,
                                     const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_ranked_matrix_rhs_t(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_ranked_reduce_lhs_t(void *backend,
                                     const TensorViewStruct *a,
                                     const TensorViewStruct *b,
                                     const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_ranked_reduce_lhs_t(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_batched(void *backend, const TensorViewStruct *a,
                         const TensorViewStruct *b,
                         const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_batched(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_batched_lhs_t(void *backend, const TensorViewStruct *a,
                               const TensorViewStruct *b,
                               const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_batched_lhs_t(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_batched_rhs_t(void *backend, const TensorViewStruct *a,
                               const TensorViewStruct *b,
                               const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_batched_rhs_t(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_transpose(void *backend, const TensorViewStruct *x,
                      const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).transpose(to_tensor_view(*x), out_view);
}

void plugin_layernorm_forward(void *backend, const TensorViewStruct *x,
                              const TensorViewStruct *gamma_1xC,
                              const TensorViewStruct *beta_1xC,
                              const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).layernorm_forward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC),
      to_tensor_view(*beta_1xC), out_view);
}

void plugin_layernorm_backward(void *backend, const TensorViewStruct *x,
                               const TensorViewStruct *gamma_1xC,
                               const TensorViewStruct *dout,
                               const TensorViewStruct *dx,
                               const TensorViewStruct *dgamma_1xC,
                               const TensorViewStruct *dbeta_1xC) {
  TensorView dx_view = to_tensor_view(*dx);
  TensorView dgamma_view = to_tensor_view(*dgamma_1xC);
  TensorView dbeta_view = to_tensor_view(*dbeta_1xC);
  to_openblas_backend(backend).layernorm_backward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC), to_tensor_view(*dout),
      dx_view, dgamma_view, dbeta_view);
}

void plugin_embedding_lookup(void *backend, const TensorViewStruct *table,
                             const TensorViewStruct *ids,
                             const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).embedding_lookup(
      to_tensor_view(*table), to_tensor_view(*ids), out_view);
}

void plugin_accumulate_embedding_grads(void *backend,
                                       const TensorViewStruct *ids,
                                       const TensorViewStruct *d_cur,
                                       const TensorViewStruct *d_tok) {
  TensorView d_tok_view = to_tensor_view(*d_tok);
  to_openblas_backend(backend).accumulate_embedding_grads(
      to_tensor_view(*ids), to_tensor_view(*d_cur), d_tok_view);
}

void plugin_cross_entropy_mean(void *backend, const TensorViewStruct *logits,
                               const TensorViewStruct *targets,
                               const TensorViewStruct *out_loss) {
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_openblas_backend(backend).cross_entropy_mean(
      to_tensor_view(*logits), to_tensor_view(*targets), out_loss_view);
}

void plugin_cross_entropy_mean_backward_inplace(
    void *backend, const TensorViewStruct *logits,
    const TensorViewStruct *targets, const TensorViewStruct *out_loss) {
  TensorView logits_view = to_tensor_view(*logits);
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_openblas_backend(backend).cross_entropy_mean_backward_inplace(
      logits_view, to_tensor_view(*targets), out_loss_view);
}

float plugin_read_scalar_f32(void *backend, const TensorViewStruct *x) {
  return to_openblas_backend(backend).read_scalar_f32(to_tensor_view(*x));
}

void plugin_backward_from_logits_targets(void *backend,
                                         const TensorViewStruct *logits,
                                         const TensorViewStruct *targets) {
  TensorView logits_view = to_tensor_view(*logits);
  to_openblas_backend(backend).backward_from_logits_targets(
      logits_view, to_tensor_view(*targets));
}

void plugin_softmax_rows(void *backend, const TensorViewStruct *x,
                         const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).softmax_rows(to_tensor_view(*x), out_view);
}

void plugin_softmax_backward_rows(void *backend,
                                  const TensorViewStruct *softmax,
                                  const TensorViewStruct *dout,
                                  const TensorViewStruct *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).softmax_backward_rows(
      to_tensor_view(*softmax), to_tensor_view(*dout), dx_view);
}

void plugin_scaled_causal_softmax_rows(void *backend,
                                       const TensorViewStruct *scores,
                                       float scale,
                                       const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).scaled_causal_softmax_rows(
      to_tensor_view(*scores), scale, out_view);
}

uint32_t plugin_supports_exec_context_iteration(void *backend) {
  return to_openblas_backend(backend).supports_exec_context_iteration() ? 1u
                                                                        : 0u;
}

void plugin_start_exec_context_iteration(void *backend) {
  to_openblas_backend(backend).start_exec_context_iteration();
}

void plugin_finish_exec_context_iteration(void *backend) {
  to_openblas_backend(backend).finish_exec_context_iteration();
}

void plugin_start_exec_context_group(void *backend) {
  to_openblas_backend(backend).start_exec_context_group();
}

void plugin_finish_exec_context_group(void *backend) {
  to_openblas_backend(backend).finish_exec_context_group();
}

void plugin_gemm_batched_exec_context(void *backend, const TensorViewStruct *a,
                                      const TensorViewStruct *b,
                                      const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_batched_exec_context(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_batched_rhs_t_exec_context(
    void *backend, const TensorViewStruct *a, const TensorViewStruct *b,
    const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_batched_rhs_t_exec_context(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_gemm_batched_lhs_t_exec_context(
    void *backend, const TensorViewStruct *a, const TensorViewStruct *b,
    const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).gemm_batched_lhs_t_exec_context(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_scaled_causal_softmax_rows_exec_context(
    void *backend, const TensorViewStruct *scores, float scale,
    const TensorViewStruct *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).scaled_causal_softmax_rows_exec_context(
      to_tensor_view(*scores), scale, out_view);
}

void plugin_softmax_backward_causal_rows_exec_context(
    void *backend, const TensorViewStruct *softmax,
    const TensorViewStruct *dout, const TensorViewStruct *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).softmax_backward_causal_rows_exec_context(
      to_tensor_view(*softmax), to_tensor_view(*dout), dx_view);
}

void plugin_softmax_backward_causal_rows(void *backend,
                                         const TensorViewStruct *softmax,
                                         const TensorViewStruct *dout,
                                         const TensorViewStruct *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).softmax_backward_causal_rows(
      to_tensor_view(*softmax), to_tensor_view(*dout), dx_view);
}

void plugin_apply_causal_mask_inplace(void *backend,
                                      const TensorViewStruct *scores,
                                      float neg_inf) {
  TensorView scores_view = to_tensor_view(*scores);
  to_openblas_backend(backend).apply_causal_mask_inplace(scores_view, neg_inf);
}

void plugin_rotary_embedding_inplace(void *backend, const TensorViewStruct *x,
                                     int64_t n_heads, float base,
                                     uint32_t inverse) {
  TensorView x_view = to_tensor_view(*x);
  to_openblas_backend(backend).rotary_embedding_inplace(x_view, n_heads, base, inverse != 0);
}

void plugin_adamw_step(void *backend, const TensorViewStruct *params,
                       const TensorViewStruct *grads,
                       const TensorViewStruct *m,
                       const TensorViewStruct *v, uint64_t step,
                       float learning_rate, float beta1, float beta2,
                       float weight_decay, uint32_t apply_weight_decay) {
  TensorView params_view = to_tensor_view(*params);
  TensorView m_view = to_tensor_view(*m);
  TensorView v_view = to_tensor_view(*v);
  to_openblas_backend(backend).adamw_step(
      params_view, to_tensor_view(*grads), m_view, v_view, step, learning_rate,
      beta1, beta2, weight_decay, apply_weight_decay != 0);
}

uint32_t plugin_is_file2device_read_supported(void *backend) {
  return to_openblas_backend(backend).is_file2device_read_supported() ? 1u : 0u;
}

void plugin_read_file2device(void *backend, const char *path, void *dst,
                             uint64_t size, uint64_t file_offset) {
  to_openblas_backend(backend).read_file2device(path, dst, size, file_offset);
}

BackendMemoryInfo plugin_memory_info(void *backend) {
  (void)backend;
  return BackendMemoryInfo{0u, 0u, 0u};
}

uint32_t plugin_precision_variants(void *backend) {
  return to_openblas_backend(backend).precision_variants();
}

void plugin_cast(void *backend, const TensorViewStruct *src,
                 const TensorViewStruct *dst) {
  TensorView dst_view = to_tensor_view(*dst);
  to_openblas_backend(backend).cast(to_tensor_view(*src), dst_view);
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
    &plugin_gemm,
    &plugin_gemm_ranked_matrix_rhs,
    &plugin_gemm_ranked_matrix_rhs_t,
    &plugin_gemm_ranked_reduce_lhs_t,
    &plugin_gemm_batched,
    &plugin_gemm_batched_lhs_t,
    &plugin_gemm_batched_rhs_t,
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
    &plugin_supports_exec_context_iteration,
    &plugin_start_exec_context_iteration,
    &plugin_finish_exec_context_iteration,
    &plugin_start_exec_context_group,
    &plugin_finish_exec_context_group,
    &plugin_gemm_batched_exec_context,
    &plugin_gemm_batched_lhs_t_exec_context,
    &plugin_gemm_batched_rhs_t_exec_context,
    &plugin_scaled_causal_softmax_rows_exec_context,
    &plugin_softmax_backward_causal_rows_exec_context,
    &plugin_softmax_backward_causal_rows,
    &plugin_apply_causal_mask_inplace,
    &plugin_rotary_embedding_inplace,
    &plugin_adamw_step,
    &plugin_is_file2device_read_supported,
    &plugin_read_file2device,
    &plugin_memory_info,
    &plugin_precision_variants,
    &plugin_cast,
};

} // namespace

extern "C" const BackendApiV1 *litnice_backend_get_api() { return &kBackendApi; }
