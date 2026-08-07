#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"

#include <cblas.h>

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void warn_cpu_fallback(const char *op) {
  std::cerr << "\033[31m"
            << "openblas_plugin TEMP fallback to cpu_backend_ in " << op
            << "\033[0m" << std::endl;
}

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
  validate_backend_tensor_view(abi, "openblas plugin");
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

uint64_t logical_prefix_count(const TensorView &view, size_t suffix_rank) {
  if (view.rank() < suffix_rank) {
    throw std::runtime_error(
        "openblas plugin: suffix rank exceeds tensor rank");
  }
  uint64_t count = 1;
  for (size_t i = 0; i + suffix_rank < view.rank(); ++i) {
    count *= static_cast<uint64_t>(view.dim(i));
  }
  return count;
}

int64_t prefix_matrix_byte_offset(const TensorView &view, uint64_t prefix,
                                  size_t suffix_rank) {
  if (view.rank() < suffix_rank) {
    throw std::runtime_error(
        "openblas plugin: suffix rank exceeds tensor rank");
  }
  const size_t prefix_rank = view.rank() - suffix_rank;
  int64_t offset = 0;
  for (size_t axis = prefix_rank; axis-- > 0;) {
    const uint64_t dim = static_cast<uint64_t>(view.dim(axis));
    const uint64_t idx = prefix % dim;
    prefix /= dim;
    offset += static_cast<int64_t>(idx) * view.stride_bytes(axis);
  }
  return offset;
}

float *prefix_matrix_ptr(TensorView &view, uint64_t prefix, int64_t rows,
                         int64_t cols) {
  (void)rows;
  (void)cols;
  auto *base = reinterpret_cast<uint8_t *>(view.data());
  return reinterpret_cast<float *>(
      base + prefix_matrix_byte_offset(view, prefix, 2));
}

const float *prefix_matrix_ptr(const TensorView &view, uint64_t prefix,
                               int64_t rows, int64_t cols) {
  (void)rows;
  (void)cols;
  auto *base = reinterpret_cast<const uint8_t *>(view.data());
  return reinterpret_cast<const float *>(
      base + prefix_matrix_byte_offset(view, prefix, 2));
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

  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override {
    if (a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank() &&
        is_batched_blas_compatible_f32_row_major(a) &&
        is_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const int m = static_cast<int>(logical_prefix_count(a, 2) *
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
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank() &&
        is_batched_blas_compatible_f32_row_major(a) &&
        is_batched_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const uint64_t prefix_count = logical_prefix_count(a, 2);
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
    if (!is_blas_compatible_f32_row_major(a) ||
        !is_blas_compatible_f32_row_major(b) ||
        !is_blas_compatible_f32_row_major(out)) {
      warn_cpu_fallback("matmul");
      cpu_backend_.matmul(a, b, out);
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

  void matmul_left_transposed(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == 2 &&
        is_contiguous_f32_row_major(a) && is_contiguous_f32_row_major(b) &&
        is_contiguous_f32_row_major(out)) {
      const int m = static_cast<int>(a.dim(a.rank() - 1));
      const int k = static_cast<int>(logical_prefix_count(a, 2) *
                                     static_cast<uint64_t>(a.dim(a.rank() - 2)));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
                  static_cast<const float *>(a.data()), m,
                  static_cast<const float *>(b.data()), n, 0.0f,
                  static_cast<float *>(out.data()), n);
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank() &&
        is_batched_blas_compatible_f32_row_major(a) &&
        is_batched_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const uint64_t prefix_count = logical_prefix_count(a, 2);
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
    if (!is_blas_compatible_f32_row_major(a) ||
        !is_blas_compatible_f32_row_major(b) ||
        !is_blas_compatible_f32_row_major(out)) {
      warn_cpu_fallback("matmul_left_transposed");
      cpu_backend_.matmul_left_transposed(a, b, out);
      return;
    }

    const int m = static_cast<int>(a.shape().dim(1));
    const int k = static_cast<int>(a.shape().dim(0));
    const int n = static_cast<int>(b.shape().dim(1));
    const int lda = leading_dim_f32(a);
    const int ldb = leading_dim_f32(b);
    const int ldc = leading_dim_f32(out);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), lda,
                static_cast<const float *>(b.data()), ldb, 0.0f,
                static_cast<float *>(out.data()), ldc);
  }

  void matmul_right_transposed(const TensorView &a, const TensorView &b,
                               TensorView &out) override {
    if (a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank() &&
        is_batched_blas_compatible_f32_row_major(a) &&
        is_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const int m = static_cast<int>(logical_prefix_count(a, 2) *
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
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank() &&
        is_batched_blas_compatible_f32_row_major(a) &&
        is_batched_blas_compatible_f32_row_major(b) &&
        is_batched_blas_compatible_f32_row_major(out)) {
      const uint64_t prefix_count = logical_prefix_count(a, 2);
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
    if (!is_blas_compatible_f32_row_major(a) ||
        !is_blas_compatible_f32_row_major(b) ||
        !is_blas_compatible_f32_row_major(out)) {
      warn_cpu_fallback("matmul_right_transposed");
      cpu_backend_.matmul_right_transposed(a, b, out);
      return;
    }

    const int m = static_cast<int>(a.shape().dim(0));
    const int k = static_cast<int>(a.shape().dim(1));
    const int n = static_cast<int>(b.shape().dim(0));
    const int lda = leading_dim_f32(a);
    const int ldb = leading_dim_f32(b);
    const int ldc = leading_dim_f32(out);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                static_cast<const float *>(a.data()), lda,
                static_cast<const float *>(b.data()), ldb, 0.0f,
                static_cast<float *>(out.data()), ldc);
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
                                  const TensorView &d_cur, TensorView &d_tok,
                                  TensorView &d_pos) override {
    cpu_backend_.accumulate_embedding_grads(ids, d_cur, d_tok, d_pos);
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

  void softmax_backward_causal_rows(const TensorView &softmax,
                                    const TensorView &dout,
                                    TensorView &dx) override {
    cpu_backend_.softmax_backward_causal_rows(softmax, dout, dx);
  }

  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override {
    cpu_backend_.apply_causal_mask_inplace(scores, neg_inf);
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

void plugin_add_bias_relu_rowwise(void *backend, const BackendTensorView *x,
                                  const BackendTensorView *bias_1xC,
                                  const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).add_bias_relu_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_add_bias_relu_rowwise_inplace(void *backend,
                                          const BackendTensorView *x,
                                          const BackendTensorView *bias_1xC) {
  TensorView x_view = to_tensor_view(*x);
  to_openblas_backend(backend).add_bias_relu_rowwise_inplace(
      x_view, to_tensor_view(*bias_1xC));
}

void plugin_mul_scalar(void *backend, const BackendTensorView *x, float s,
                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).mul_scalar(to_tensor_view(*x), s, out_view);
}

float plugin_sum_squares_f32(void *backend, const BackendTensorView *x) {
  return to_openblas_backend(backend).sum_squares_f32(to_tensor_view(*x));
}

void plugin_relu(void *backend, const BackendTensorView *x,
                 const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).relu(to_tensor_view(*x), out_view);
}

void plugin_relu_backward(void *backend, const BackendTensorView *preact,
                          const BackendTensorView *dout,
                          const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).relu_backward(to_tensor_view(*preact),
                                             to_tensor_view(*dout), dx_view);
}

void plugin_relu_backward_inplace(void *backend,
                                  const BackendTensorView *preact,
                                  const BackendTensorView *dout_dx) {
  TensorView dout_dx_view = to_tensor_view(*dout_dx);
  to_openblas_backend(backend).relu_backward_inplace(to_tensor_view(*preact),
                                                     dout_dx_view);
}

void plugin_row_sum(void *backend, const BackendTensorView *x,
                    const BackendTensorView *out_1xC) {
  TensorView out_view = to_tensor_view(*out_1xC);
  to_openblas_backend(backend).row_sum(to_tensor_view(*x), out_view);
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

void plugin_accumulate_embedding_grads(void *backend,
                                       const BackendTensorView *ids,
                                       const BackendTensorView *d_cur,
                                       const BackendTensorView *d_tok,
                                       const BackendTensorView *d_pos) {
  TensorView d_tok_view = to_tensor_view(*d_tok);
  TensorView d_pos_view = to_tensor_view(*d_pos);
  to_openblas_backend(backend).accumulate_embedding_grads(
      to_tensor_view(*ids), to_tensor_view(*d_cur), d_tok_view, d_pos_view);
}

void plugin_cross_entropy_mean(void *backend, const BackendTensorView *logits,
                               const BackendTensorView *targets,
                               const BackendTensorView *out_loss) {
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_openblas_backend(backend).cross_entropy_mean(
      to_tensor_view(*logits), to_tensor_view(*targets), out_loss_view);
}

void plugin_cross_entropy_mean_backward_inplace(
    void *backend, const BackendTensorView *logits,
    const BackendTensorView *targets, const BackendTensorView *out_loss) {
  TensorView logits_view = to_tensor_view(*logits);
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_openblas_backend(backend).cross_entropy_mean_backward_inplace(
      logits_view, to_tensor_view(*targets), out_loss_view);
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

void plugin_softmax_backward_rows(void *backend,
                                  const BackendTensorView *softmax,
                                  const BackendTensorView *dout,
                                  const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).softmax_backward_rows(
      to_tensor_view(*softmax), to_tensor_view(*dout), dx_view);
}

void plugin_scaled_causal_softmax_rows(void *backend,
                                       const BackendTensorView *scores,
                                       float scale,
                                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_openblas_backend(backend).scaled_causal_softmax_rows(
      to_tensor_view(*scores), scale, out_view);
}

void plugin_softmax_backward_causal_rows(void *backend,
                                         const BackendTensorView *softmax,
                                         const BackendTensorView *dout,
                                         const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_openblas_backend(backend).softmax_backward_causal_rows(
      to_tensor_view(*softmax), to_tensor_view(*dout), dx_view);
}

void plugin_apply_causal_mask_inplace(void *backend,
                                      const BackendTensorView *scores,
                                      float neg_inf) {
  TensorView scores_view = to_tensor_view(*scores);
  to_openblas_backend(backend).apply_causal_mask_inplace(scores_view, neg_inf);
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

extern "C" const BackendApiV1 *litnice_backend_get_api() { return &kBackendApi; }
