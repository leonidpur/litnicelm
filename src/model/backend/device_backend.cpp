#include "device_backend.hpp"
#include "backend_plugin_api.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
bool is_power_of_two(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

struct AllocationHeader {
  void *base = nullptr;
};

float load_f32_backend(const TensorView &t, int64_t r, int64_t c) {
  const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
  const uint8_t *p = base + r * t.stride_r_bytes() + c * t.stride_c_bytes();
  float out;
  std::memcpy(&out, p, sizeof(float));
  return out;
}

void store_f32_backend(const TensorView &t, int64_t r, int64_t c, float v) {
  uint8_t *base = reinterpret_cast<uint8_t *>(t.data());
  uint8_t *p = base + r * t.stride_r_bytes() + c * t.stride_c_bytes();
  std::memcpy(p, &v, sizeof(float));
}

int32_t load_i32_backend(const TensorView &t, int64_t r, int64_t c) {
  const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
  const uint8_t *p = base + r * t.stride_r_bytes() + c * t.stride_c_bytes();
  int32_t out;
  std::memcpy(&out, p, sizeof(int32_t));
  return out;
}

int64_t load_index_backend(const TensorView &t, int64_t r, int64_t c) {
  if (t.dtype() == DType::I32) {
    return static_cast<int64_t>(load_i32_backend(t, r, c));
  }
  if (t.dtype() == DType::F32) {
    return static_cast<int64_t>(load_f32_backend(t, r, c));
  }
  throw std::runtime_error("DeviceBackend: unsupported index dtype");
}

BackendTensorView to_backend_tensor_view(const TensorView &view) {
  return BackendTensorView{
      static_cast<uint32_t>(view.device()),
      static_cast<uint32_t>(view.dtype()),
      view.data(),
      view.shape().r,
      view.shape().c,
      view.stride_r_bytes(),
      view.stride_c_bytes(),
  };
}

class DynamicLibraryBackend final : public DeviceBackend {
public:
  DynamicLibraryBackend(const std::string &library_path, Device device) {
    library_handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_handle_ == nullptr) {
      throw std::runtime_error("DynamicLibraryBackend: failed to load " +
                               library_path + ": " + dlerror());
    }
    std::cout << "\033[32m[DeviceBackend] Loaded backend library: "
              << library_path << "\033[0m\n";

    auto *get_api = reinterpret_cast<BackendGetApiFn>(
        dlsym(library_handle_, "litnice_backend_get_api"));
    if (get_api == nullptr) {
      const std::string err = dlerror() != nullptr ? dlerror() : "missing symbol";
      dlclose(library_handle_);
      library_handle_ = nullptr;
      throw std::runtime_error("DynamicLibraryBackend: failed to resolve backend API: " +
                               err);
    }

    api_ = get_api();
    if (api_ == nullptr || api_->abi_version != kBackendApiVersion) {
      dlclose(library_handle_);
      library_handle_ = nullptr;
      throw std::runtime_error("DynamicLibraryBackend: backend ABI mismatch");
    }
    instance_ = api_->create(static_cast<uint32_t>(device));
    if (instance_ == nullptr) {
      dlclose(library_handle_);
      library_handle_ = nullptr;
      throw std::runtime_error("DynamicLibraryBackend: backend create failed");
    }
  }

  ~DynamicLibraryBackend() override {
    if (api_ != nullptr && instance_ != nullptr) {
      api_->destroy(instance_);
    }
    if (library_handle_ != nullptr) {
      dlclose(library_handle_);
    }
  }

  void *alloc(uint64_t bytes, uint32_t alignment) override {
    return api_->alloc(instance_, bytes, alignment);
  }

  void free(void *ptr) override { api_->free(instance_, ptr); }

  void copy_host2device(void *dst, const void *src, uint64_t bytes) override {
    api_->copy_host2device(instance_, dst, src, bytes);
  }

  void copy_device2host(void *dst, const void *src, uint64_t bytes) override {
    api_->copy_device2host(instance_, dst, src, bytes);
  }

  void copy(const TensorView &src, TensorView &dst) override {
    const BackendTensorView src_view = to_backend_tensor_view(src);
    const BackendTensorView dst_view = to_backend_tensor_view(dst);
    api_->copy(instance_, &src_view, &dst_view);
  }

  void fill(TensorView &t, float v) override {
    const BackendTensorView t_view = to_backend_tensor_view(t);
    api_->fill(instance_, &t_view, v);
  }

  void add(const TensorView &a, const TensorView &b, TensorView &out) override {
    const BackendTensorView a_view = to_backend_tensor_view(a);
    const BackendTensorView b_view = to_backend_tensor_view(b);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->add(instance_, &a_view, &b_view, &out_view);
  }

  void add_inplace(TensorView &a, const TensorView &b) override {
    const BackendTensorView a_view = to_backend_tensor_view(a);
    const BackendTensorView b_view = to_backend_tensor_view(b);
    api_->add_inplace(instance_, &a_view, &b_view);
  }

  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView bias_view = to_backend_tensor_view(bias_1xC);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->add_bias_rowwise(instance_, &x_view, &bias_view, &out_view);
  }

  void mul_scalar(const TensorView &x, float s, TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->mul_scalar(instance_, &x_view, s, &out_view);
  }

  void relu(const TensorView &x, TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->relu(instance_, &x_view, &out_view);
  }

  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override {
    const BackendTensorView a_view = to_backend_tensor_view(a);
    const BackendTensorView b_view = to_backend_tensor_view(b);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->matmul(instance_, &a_view, &b_view, &out_view);
  }

  void matmul_left_transposed(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    const BackendTensorView a_view = to_backend_tensor_view(a);
    const BackendTensorView b_view = to_backend_tensor_view(b);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->matmul_left_transposed(instance_, &a_view, &b_view, &out_view);
  }

  void matmul_right_transposed(const TensorView &a, const TensorView &b,
                               TensorView &out) override {
    const BackendTensorView a_view = to_backend_tensor_view(a);
    const BackendTensorView b_view = to_backend_tensor_view(b);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->matmul_right_transposed(instance_, &a_view, &b_view, &out_view);
  }

  void transpose(const TensorView &x, TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->transpose(instance_, &x_view, &out_view);
  }

  void layernorm_forward(const TensorView &x, const TensorView &gamma_1xC,
                         const TensorView &beta_1xC, TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView gamma_view = to_backend_tensor_view(gamma_1xC);
    const BackendTensorView beta_view = to_backend_tensor_view(beta_1xC);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->layernorm_forward(instance_, &x_view, &gamma_view, &beta_view,
                            &out_view);
  }

  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView gamma_view = to_backend_tensor_view(gamma_1xC);
    const BackendTensorView dout_view = to_backend_tensor_view(dout);
    const BackendTensorView dx_view = to_backend_tensor_view(dx);
    const BackendTensorView dgamma_view = to_backend_tensor_view(dgamma_1xC);
    const BackendTensorView dbeta_view = to_backend_tensor_view(dbeta_1xC);
    api_->layernorm_backward(instance_, &x_view, &gamma_view, &dout_view,
                             &dx_view, &dgamma_view, &dbeta_view);
  }

  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) override {
    const BackendTensorView table_view = to_backend_tensor_view(table);
    const BackendTensorView ids_view = to_backend_tensor_view(ids);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->embedding_lookup(instance_, &table_view, &ids_view, &out_view);
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    const BackendTensorView logits_view = to_backend_tensor_view(logits);
    const BackendTensorView targets_view = to_backend_tensor_view(targets);
    const BackendTensorView out_loss_view = to_backend_tensor_view(out_loss);
    api_->cross_entropy_mean(instance_, &logits_view, &targets_view,
                             &out_loss_view);
  }

  float read_scalar_f32(const TensorView &x) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    return api_->read_scalar_f32(instance_, &x_view);
  }

  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override {
    const BackendTensorView logits_view = to_backend_tensor_view(logits);
    const BackendTensorView targets_view = to_backend_tensor_view(targets);
    api_->backward_from_logits_targets(instance_, &logits_view, &targets_view);
  }

  void softmax_rows(const TensorView &x, TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->softmax_rows(instance_, &x_view, &out_view);
  }

  void apply_causal_mask_inplace(TensorView &scores, float neg_inf) override {
    const BackendTensorView scores_view = to_backend_tensor_view(scores);
    api_->apply_causal_mask_inplace(instance_, &scores_view, neg_inf);
  }

  bool is_file2device_read_supported() const override {
    return api_->is_file2device_read_supported(instance_) != 0;
  }

  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override {
    api_->read_file2device(instance_, path.c_str(), dst, size, file_offset);
  }

private:
  void *library_handle_ = nullptr;
  const BackendApiV1 *api_ = nullptr;
  void *instance_ = nullptr;
};
}

void *CpuBackend::alloc(uint64_t bytes, uint32_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument("CpuBackend::alloc: invalid alignment");
  }

  const uint64_t extra =
      static_cast<uint64_t>(alignment - 1) + sizeof(AllocationHeader);
  void *base = std::malloc(static_cast<size_t>(bytes + extra));
  if (base == nullptr) {
    throw std::runtime_error("CpuBackend::alloc failed");
  }

  const auto raw =
      reinterpret_cast<std::uintptr_t>(base) + sizeof(AllocationHeader);
  const auto aligned =
      (raw + (alignment - 1)) & ~static_cast<std::uintptr_t>(alignment - 1);
  auto *header =
      reinterpret_cast<AllocationHeader *>(aligned - sizeof(AllocationHeader));
  header->base = base;
  return reinterpret_cast<void *>(aligned);
}

void CpuBackend::free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  auto *header = reinterpret_cast<AllocationHeader *>(
      reinterpret_cast<std::uintptr_t>(ptr) - sizeof(AllocationHeader));
  std::free(header->base);
}

void CpuBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("CpuBackend::copy_host2device: null pointer");
  }
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void CpuBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("CpuBackend::copy_device2host: null pointer");
  }
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void CpuBackend::copy(const TensorView &src, TensorView &dst) {
  const int64_t row_count = src.shape().r;
  const int64_t col_count = src.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(dst, r, c, load_f32_backend(src, r, c));
    }
  }
}

void CpuBackend::fill(TensorView &t, float v) {
  const int64_t row_count = t.shape().r;
  const int64_t col_count = t.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(t, r, c, v);
    }
  }
}

void CpuBackend::add(const TensorView &a, const TensorView &b, TensorView &out) {
  const int64_t row_count = a.shape().r;
  const int64_t col_count = a.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(out, r, c,
                        load_f32_backend(a, r, c) + load_f32_backend(b, r, c));
    }
  }
}

void CpuBackend::add_inplace(TensorView &a, const TensorView &b) {
  const int64_t row_count = a.shape().r;
  const int64_t col_count = a.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(a, r, c,
                        load_f32_backend(a, r, c) + load_f32_backend(b, r, c));
    }
  }
}

void CpuBackend::add_bias_rowwise(const TensorView &x,
                                  const TensorView &bias_1xC,
                                  TensorView &out) {
  const int64_t row_count = x.shape().r;
  const int64_t col_count = x.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(out, r, c,
                        load_f32_backend(x, r, c) +
                            load_f32_backend(bias_1xC, 0, c));
    }
  }
}

void CpuBackend::mul_scalar(const TensorView &x, float s, TensorView &out) {
  const int64_t row_count = x.shape().r;
  const int64_t col_count = x.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(out, r, c, load_f32_backend(x, r, c) * s);
    }
  }
}

void CpuBackend::relu(const TensorView &x, TensorView &out) {
  const int64_t row_count = x.shape().r;
  const int64_t col_count = x.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      const float value = load_f32_backend(x, r, c);
      store_f32_backend(out, r, c, value > 0.0f ? value : 0.0f);
    }
  }
}

void CpuBackend::matmul(const TensorView &a, const TensorView &b,
                        TensorView &out) {
  const int64_t row_count = a.shape().r;
  const int64_t inner_dim = a.shape().c;
  const int64_t col_count = b.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < inner_dim; ++k) {
        acc += load_f32_backend(a, r, k) * load_f32_backend(b, k, c);
      }
      store_f32_backend(out, r, c, acc);
    }
  }
}

void CpuBackend::matmul_left_transposed(const TensorView &a, const TensorView &b,
                                        TensorView &out) {
  const int64_t row_count = a.shape().c;
  const int64_t inner_dim = a.shape().r;
  const int64_t col_count = b.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < inner_dim; ++k) {
        acc += load_f32_backend(a, k, r) * load_f32_backend(b, k, c);
      }
      store_f32_backend(out, r, c, acc);
    }
  }
}

void CpuBackend::matmul_right_transposed(const TensorView &a, const TensorView &b,
                                         TensorView &out) {
  const int64_t row_count = a.shape().r;
  const int64_t inner_dim = a.shape().c;
  const int64_t col_count = b.shape().r;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < inner_dim; ++k) {
        acc += load_f32_backend(a, r, k) * load_f32_backend(b, c, k);
      }
      store_f32_backend(out, r, c, acc);
    }
  }
}

void CpuBackend::transpose(const TensorView &x, TensorView &out) {
  const int64_t row_count = x.shape().r;
  const int64_t col_count = x.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(out, c, r, load_f32_backend(x, r, c));
    }
  }
}

void CpuBackend::layernorm_forward(const TensorView &x,
                                   const TensorView &gamma_1xC,
                                   const TensorView &beta_1xC,
                                   TensorView &out) {
  const int64_t token_rows = x.shape().r;
  const int64_t model_dim = x.shape().c;
  const float eps = 1e-5f;
  for (int64_t r = 0; r < token_rows; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      mean += load_f32_backend(x, r, c);
    }
    mean /= static_cast<double>(model_dim);

    double var = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      const double d = static_cast<double>(load_f32_backend(x, r, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(model_dim);
    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

    for (int64_t c = 0; c < model_dim; ++c) {
      const float xn =
          (load_f32_backend(x, r, c) - static_cast<float>(mean)) * inv_std;
      const float y = xn * load_f32_backend(gamma_1xC, 0, c) +
                      load_f32_backend(beta_1xC, 0, c);
      store_f32_backend(out, r, c, y);
    }
  }
}

void CpuBackend::layernorm_backward(const TensorView &x,
                                    const TensorView &gamma_1xC,
                                    const TensorView &dout, TensorView &dx,
                                    TensorView &dgamma_1xC,
                                    TensorView &dbeta_1xC) {
  const int64_t token_rows = x.shape().r;
  const int64_t model_dim = x.shape().c;
  const float eps = 1e-5f;
  for (int64_t c = 0; c < model_dim; ++c) {
    store_f32_backend(dgamma_1xC, 0, c, 0.0f);
    store_f32_backend(dbeta_1xC, 0, c, 0.0f);
  }
  for (int64_t r = 0; r < token_rows; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      mean += load_f32_backend(x, r, c);
    }
    mean /= static_cast<double>(model_dim);

    double var = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      const double d = static_cast<double>(load_f32_backend(x, r, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(model_dim);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    double sum_dxhat = 0.0;
    double sum_dxhat_xhat = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      const double xhat =
          (static_cast<double>(load_f32_backend(x, r, c)) - mean) * inv_std;
      const double g = static_cast<double>(load_f32_backend(gamma_1xC, 0, c));
      const double dyi = static_cast<double>(load_f32_backend(dout, r, c));
      const double dxhat = dyi * g;
      sum_dxhat += dxhat;
      sum_dxhat_xhat += dxhat * xhat;
      store_f32_backend(dgamma_1xC, 0, c,
                        load_f32_backend(dgamma_1xC, 0, c) +
                            static_cast<float>(dyi * xhat));
      store_f32_backend(dbeta_1xC, 0, c,
                        load_f32_backend(dbeta_1xC, 0, c) +
                            static_cast<float>(dyi));
    }

    for (int64_t c = 0; c < model_dim; ++c) {
      const double xhat =
          (static_cast<double>(load_f32_backend(x, r, c)) - mean) * inv_std;
      const double g = static_cast<double>(load_f32_backend(gamma_1xC, 0, c));
      const double dyi = static_cast<double>(load_f32_backend(dout, r, c));
      const double dxhat = dyi * g;
      const double n = static_cast<double>(model_dim);
      const double dxi =
          (inv_std / n) * (n * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
      store_f32_backend(dx, r, c, static_cast<float>(dxi));
    }
  }
}

void CpuBackend::embedding_lookup(const TensorView &table, const TensorView &ids,
                                  TensorView &out) {
  const int64_t vocab_size = table.shape().r;
  const int64_t model_dim = table.shape().c;
  const int64_t token_rows = out.shape().r;
  for (int64_t t = 0; t < token_rows; ++t) {
    const int64_t idx = load_index_backend(ids, t, 0);
    if (idx < 0 || idx >= vocab_size) {
      throw std::runtime_error("CpuBackend::embedding_lookup: embedding id out of range");
    }
    const uint8_t *src = reinterpret_cast<const uint8_t *>(table.data()) +
                         idx * table.stride_r_bytes();
    uint8_t *dst = reinterpret_cast<uint8_t *>(out.data()) +
                   t * out.stride_r_bytes();
    std::memcpy(dst, src, static_cast<size_t>(model_dim) * sizeof(float));
  }
}

void CpuBackend::cross_entropy_mean(const TensorView &logits,
                                    const TensorView &targets,
                                    TensorView &out_loss) {
  const int64_t token_rows = logits.shape().r;
  const int64_t vocab_size = logits.shape().c;
  double sum = 0.0;
  for (int64_t t = 0; t < token_rows; ++t) {
    const int64_t target = load_index_backend(targets, t, 0);
    if (target < 0 || target >= vocab_size) {
      throw std::runtime_error("CpuBackend::cross_entropy_mean: target out of range");
    }
    float max_logit = load_f32_backend(logits, t, 0);
    for (int64_t c = 1; c < vocab_size; ++c) {
      max_logit = std::max(max_logit, load_f32_backend(logits, t, c));
    }
    double lse = 0.0;
    for (int64_t c = 0; c < vocab_size; ++c) {
      lse += std::exp(static_cast<double>(load_f32_backend(logits, t, c) - max_logit));
    }
    const double log_denom = static_cast<double>(max_logit) + std::log(lse);
    sum += log_denom - static_cast<double>(load_f32_backend(logits, t, target));
  }
  store_f32_backend(out_loss, 0, 0,
                    static_cast<float>(sum / static_cast<double>(token_rows)));
}

float CpuBackend::read_scalar_f32(const TensorView &x) {
  return load_f32_backend(x, 0, 0);
}

void CpuBackend::backward_from_logits_targets(TensorView &logits,
                                              const TensorView &targets) {
  const int64_t token_rows = logits.shape().r;
  const int64_t vocab_size = logits.shape().c;
  const float inv_token_rows = 1.0f / static_cast<float>(token_rows);
  for (int64_t t = 0; t < token_rows; ++t) {
    const int64_t target = load_index_backend(targets, t, 0);
    if (target < 0 || target >= vocab_size) {
      throw std::runtime_error("CpuBackend::backward_from_logits_targets: target out of range");
    }
    float max_logit = load_f32_backend(logits, t, 0);
    for (int64_t c = 1; c < vocab_size; ++c) {
      max_logit = std::max(max_logit, load_f32_backend(logits, t, c));
    }
    double sum = 0.0;
    for (int64_t c = 0; c < vocab_size; ++c) {
      sum += std::exp(static_cast<double>(load_f32_backend(logits, t, c) - max_logit));
    }
    if (sum <= 0.0) {
      throw std::runtime_error("CpuBackend::backward_from_logits_targets: softmax sum <= 0");
    }
    for (int64_t c = 0; c < vocab_size; ++c) {
      const float p = static_cast<float>(
          std::exp(static_cast<double>(load_f32_backend(logits, t, c) - max_logit)) /
          sum);
      float gradient = p;
      if (c == target) {
        gradient -= 1.0f;
      }
      store_f32_backend(logits, t, c, gradient * inv_token_rows);
    }
  }
}

void CpuBackend::softmax_rows(const TensorView &x, TensorView &out) {
  const int64_t row_count = x.shape().r;
  const int64_t col_count = x.shape().c;
  for (int64_t r = 0; r < row_count; ++r) {
    float max_value = load_f32_backend(x, r, 0);
    for (int64_t c = 1; c < col_count; ++c) {
      max_value = std::max(max_value, load_f32_backend(x, r, c));
    }
    double sum = 0.0;
    for (int64_t c = 0; c < col_count; ++c) {
      const float exponent = std::exp(load_f32_backend(x, r, c) - max_value);
      store_f32_backend(out, r, c, exponent);
      sum += static_cast<double>(exponent);
    }
    if (sum <= 0.0) {
      throw std::runtime_error("CpuBackend::softmax_rows: softmax sum <= 0");
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int64_t c = 0; c < col_count; ++c) {
      store_f32_backend(out, r, c,
                        load_f32_backend(out, r, c) * inv_sum);
    }
  }
}

void CpuBackend::apply_causal_mask_inplace(TensorView &scores, float neg_inf) {
  const int64_t token_rows = scores.shape().r;
  for (int64_t i = 0; i < token_rows; ++i) {
    for (int64_t j = i + 1; j < token_rows; ++j) {
      store_f32_backend(scores, i, j, neg_inf);
    }
  }
}

bool CpuBackend::is_file2device_read_supported() const { return true; }

void CpuBackend::read_file2device(const std::string &path, void *dst,
                                  uint64_t size, uint64_t file_offset) {
  if (size == 0) {
    return;
  }
  if (dst == nullptr) {
    throw std::invalid_argument("CpuBackend::read_file2device: null destination");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("CpuBackend::read_file2device: failed to open file");
  }
  in.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
  if (!in) {
    throw std::runtime_error("CpuBackend::read_file2device: failed to seek file");
  }
  in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(size));
  if (in.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("CpuBackend::read_file2device: short read");
  }
}

void *CudaBackend::alloc(uint64_t bytes, uint32_t alignment) {
  (void)bytes;
  (void)alignment;
  throw std::runtime_error("CudaBackend::alloc: GPU backend is not implemented yet");
}

void CudaBackend::free(void *ptr) {
  (void)ptr;
  throw std::runtime_error("CudaBackend::free: GPU backend is not implemented yet");
}

void CudaBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  throw std::runtime_error(
      "CudaBackend::copy_host2device: GPU backend is not implemented yet");
}

void CudaBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  throw std::runtime_error(
      "CudaBackend::copy_device2host: GPU backend is not implemented yet");
}

void CudaBackend::copy(const TensorView &src, TensorView &dst) {
  (void)src; (void)dst;
  throw std::runtime_error("CudaBackend::copy: GPU backend is not implemented yet");
}
void CudaBackend::fill(TensorView &t, float v) {
  (void)t; (void)v;
  throw std::runtime_error("CudaBackend::fill: GPU backend is not implemented yet");
}
void CudaBackend::add(const TensorView &a, const TensorView &b, TensorView &out) {
  (void)a; (void)b; (void)out;
  throw std::runtime_error("CudaBackend::add: GPU backend is not implemented yet");
}
void CudaBackend::add_inplace(TensorView &a, const TensorView &b) {
  (void)a; (void)b;
  throw std::runtime_error("CudaBackend::add_inplace: GPU backend is not implemented yet");
}
void CudaBackend::add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                                   TensorView &out) {
  (void)x; (void)bias_1xC; (void)out;
  throw std::runtime_error("CudaBackend::add_bias_rowwise: GPU backend is not implemented yet");
}
void CudaBackend::mul_scalar(const TensorView &x, float s, TensorView &out) {
  (void)x; (void)s; (void)out;
  throw std::runtime_error("CudaBackend::mul_scalar: GPU backend is not implemented yet");
}
void CudaBackend::relu(const TensorView &x, TensorView &out) {
  (void)x; (void)out;
  throw std::runtime_error("CudaBackend::relu: GPU backend is not implemented yet");
}
void CudaBackend::matmul(const TensorView &a, const TensorView &b, TensorView &out) {
  (void)a; (void)b; (void)out;
  throw std::runtime_error("CudaBackend::matmul: GPU backend is not implemented yet");
}
void CudaBackend::matmul_left_transposed(const TensorView &a, const TensorView &b,
                                         TensorView &out) {
  (void)a; (void)b; (void)out;
  throw std::runtime_error("CudaBackend::matmul_left_transposed: GPU backend is not implemented yet");
}
void CudaBackend::matmul_right_transposed(const TensorView &a, const TensorView &b,
                                          TensorView &out) {
  (void)a; (void)b; (void)out;
  throw std::runtime_error("CudaBackend::matmul_right_transposed: GPU backend is not implemented yet");
}
void CudaBackend::transpose(const TensorView &x, TensorView &out) {
  (void)x; (void)out;
  throw std::runtime_error("CudaBackend::transpose: GPU backend is not implemented yet");
}

void CudaBackend::layernorm_forward(const TensorView &x,
                                    const TensorView &gamma_1xC,
                                    const TensorView &beta_1xC,
                                    TensorView &out) {
  (void)x;
  (void)gamma_1xC;
  (void)beta_1xC;
  (void)out;
  throw std::runtime_error(
      "CudaBackend::layernorm_forward: GPU backend is not implemented yet");
}

void CudaBackend::layernorm_backward(const TensorView &x,
                                     const TensorView &gamma_1xC,
                                     const TensorView &dout, TensorView &dx,
                                     TensorView &dgamma_1xC,
                                     TensorView &dbeta_1xC) {
  (void)x;
  (void)gamma_1xC;
  (void)dout;
  (void)dx;
  (void)dgamma_1xC;
  (void)dbeta_1xC;
  throw std::runtime_error(
      "CudaBackend::layernorm_backward: GPU backend is not implemented yet");
}

void CudaBackend::embedding_lookup(const TensorView &table, const TensorView &ids,
                                   TensorView &out) {
  (void)table; (void)ids; (void)out;
  throw std::runtime_error("CudaBackend::embedding_lookup: GPU backend is not implemented yet");
}
void CudaBackend::cross_entropy_mean(const TensorView &logits,
                                     const TensorView &targets,
                                     TensorView &out_loss) {
  (void)logits; (void)targets; (void)out_loss;
  throw std::runtime_error("CudaBackend::cross_entropy_mean: GPU backend is not implemented yet");
}
float CudaBackend::read_scalar_f32(const TensorView &x) {
  (void)x;
  throw std::runtime_error("CudaBackend::read_scalar_f32: GPU backend is not implemented yet");
}
void CudaBackend::backward_from_logits_targets(TensorView &logits,
                                               const TensorView &targets) {
  (void)logits; (void)targets;
  throw std::runtime_error(
      "CudaBackend::backward_from_logits_targets: GPU backend is not implemented yet");
}
void CudaBackend::softmax_rows(const TensorView &x, TensorView &out) {
  (void)x; (void)out;
  throw std::runtime_error("CudaBackend::softmax_rows: GPU backend is not implemented yet");
}
void CudaBackend::apply_causal_mask_inplace(TensorView &scores, float neg_inf) {
  (void)scores; (void)neg_inf;
  throw std::runtime_error(
      "CudaBackend::apply_causal_mask_inplace: GPU backend is not implemented yet");
}

bool CudaBackend::is_file2device_read_supported() const { return false; }

void CudaBackend::read_file2device(const std::string &path, void *dst,
                                   uint64_t size, uint64_t file_offset) {
  (void)path;
  (void)dst;
  (void)size;
  (void)file_offset;
  throw std::runtime_error(
      "CudaBackend::read_file2device: GPU backend is not implemented yet");
}

std::unique_ptr<DeviceBackend> make_device_backend(Device device) {
  switch (device) {
  case Device::CPU:
    return std::make_unique<CpuBackend>();
  case Device::GPU:
    return std::make_unique<CudaBackend>();
  default:
    throw std::runtime_error("make_device_backend: unsupported device");
  }
}

std::unique_ptr<DeviceBackend> make_device_backend(const Config &cfg) {
  if (!cfg.backend.library.empty()) {
    return std::make_unique<DynamicLibraryBackend>(cfg.backend.library, cfg.device);
  }
  return make_device_backend(cfg.device);
}
