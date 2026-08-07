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

void require_backend(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

struct AllocationHeader {
  void *base = nullptr;
};

struct CpuMemOperations {
  static float load_f32(const TensorView &t, int64_t r, int64_t c) {
    const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
    const uint8_t *p = base + r * t.stride_bytes(0) + c * t.stride_bytes(1);
    float out;
    std::memcpy(&out, p, sizeof(float));
    return out;
  }

  static void store_f32(const TensorView &t, int64_t r, int64_t c, float v) {
    uint8_t *base = reinterpret_cast<uint8_t *>(t.data());
    uint8_t *p = base + r * t.stride_bytes(0) + c * t.stride_bytes(1);
    std::memcpy(p, &v, sizeof(float));
  }

  static int32_t load_i32(const TensorView &t, int64_t r, int64_t c) {
    const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
    const uint8_t *p = base + r * t.stride_bytes(0) + c * t.stride_bytes(1);
    int32_t out;
    std::memcpy(&out, p, sizeof(int32_t));
    return out;
  }

  static int64_t load_index(const TensorView &t, int64_t r, int64_t c) {
    if (t.dtype() == DType::I32) {
      return static_cast<int64_t>(load_i32(t, r, c));
    }
    if (t.dtype() == DType::F32) {
      return static_cast<int64_t>(load_f32(t, r, c));
    }
    throw std::runtime_error("DeviceBackend: unsupported index dtype");
  }

  static int64_t logical_offset_bytes_for_linear_index(const TensorView &t,
                                                       uint64_t linear_index,
                                                       size_t axis_count) {
    int64_t offset = 0;
    for (size_t axis = axis_count; axis > 0; --axis) {
      const int64_t dim = t.dim(axis - 1);
      if (dim <= 0) {
        throw std::runtime_error("DeviceBackend: invalid non-positive dimension");
      }
      const uint64_t coord = linear_index % static_cast<uint64_t>(dim);
      linear_index /= static_cast<uint64_t>(dim);
      offset += static_cast<int64_t>(coord) * t.stride_bytes(axis - 1);
    }
    return offset;
  }

  static int64_t load_index_linear(const TensorView &t, uint64_t linear_index) {
    const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
    const uint8_t *p =
        base + logical_offset_bytes_for_linear_index(t, linear_index, t.rank());
    if (t.dtype() == DType::I32) {
      int32_t out;
      std::memcpy(&out, p, sizeof(int32_t));
      return static_cast<int64_t>(out);
    }
    if (t.dtype() == DType::F32) {
      float out;
      std::memcpy(&out, p, sizeof(float));
      return static_cast<int64_t>(out);
    }
    throw std::runtime_error("DeviceBackend: unsupported linear index dtype");
  }

  static uint64_t logical_prefix_count(const TensorView &t, size_t suffix_rank) {
    if (t.rank() < suffix_rank) {
      throw std::runtime_error("DeviceBackend: tensor rank smaller than suffix rank");
    }
    uint64_t count = 1;
    for (size_t axis = 0; axis + suffix_rank < t.rank(); ++axis) {
      count *= static_cast<uint64_t>(t.dim(axis));
    }
    return count;
  }

  static float load_f32_prefix_last1(const TensorView &t,
                                     uint64_t prefix_index, int64_t c) {
    const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
    const int64_t prefix_offset =
        logical_offset_bytes_for_linear_index(t, prefix_index, t.rank() - 1);
    const uint8_t *p = base + prefix_offset + c * t.stride_bytes(t.rank() - 1);
    float out;
    std::memcpy(&out, p, sizeof(float));
    return out;
  }

  static void store_f32_prefix_last1(const TensorView &t,
                                     uint64_t prefix_index, int64_t c,
                                     float v) {
    uint8_t *base = reinterpret_cast<uint8_t *>(t.data());
    const int64_t prefix_offset =
        logical_offset_bytes_for_linear_index(t, prefix_index, t.rank() - 1);
    uint8_t *p = base + prefix_offset + c * t.stride_bytes(t.rank() - 1);
    std::memcpy(p, &v, sizeof(float));
  }

  static float load_f32_prefix_last2(const TensorView &t,
                                     uint64_t prefix_index, int64_t r,
                                     int64_t c) {
    const uint8_t *base = reinterpret_cast<const uint8_t *>(t.data());
    const int64_t prefix_offset =
        logical_offset_bytes_for_linear_index(t, prefix_index, t.rank() - 2);
    const uint8_t *p = base + prefix_offset +
                       r * t.stride_bytes(t.rank() - 2) +
                       c * t.stride_bytes(t.rank() - 1);
    float out;
    std::memcpy(&out, p, sizeof(float));
    return out;
  }

  static void store_f32_prefix_last2(const TensorView &t,
                                     uint64_t prefix_index, int64_t r,
                                     int64_t c, float v) {
    uint8_t *base = reinterpret_cast<uint8_t *>(t.data());
    const int64_t prefix_offset =
        logical_offset_bytes_for_linear_index(t, prefix_index, t.rank() - 2);
    uint8_t *p = base + prefix_offset +
                 r * t.stride_bytes(t.rank() - 2) +
                 c * t.stride_bytes(t.rank() - 1);
    std::memcpy(p, &v, sizeof(float));
  }
};

BackendTensorView to_backend_tensor_view(const TensorView &view) {
  BackendTensorView abi{};
  abi.device = static_cast<uint32_t>(view.device());
  abi.dtype = static_cast<uint32_t>(view.dtype());
  abi.data = view.data();
  abi.rank = static_cast<uint32_t>(view.rank());
  abi.rows = view.shape().dim(0);
  abi.cols = view.shape().dim(1);
  abi.stride_r_bytes = view.rank() >= 2 ? view.stride_bytes(view.rank() - 2) : 0;
  abi.stride_c_bytes = view.rank() >= 1 ? view.stride_bytes(view.rank() - 1) : 0;
  for (size_t i = 0; i < view.rank() && i < kBackendTensorMaxRank; ++i) {
    abi.dims[i] = view.dim(i);
    abi.strides_bytes[i] = view.stride_bytes(i);
  }
  return abi;
}

void require_f32_cpu_contig(const TensorView &t, const char *who) {
  require_backend(t.device() == Device::CPU,
                  std::string(who) + " requires CPU tensor");
  require_backend(t.dtype() == DType::F32,
                  std::string(who) + " requires f32 tensor");
  require_backend(t.is_contiguous(),
                  std::string(who) + " requires contiguous tensor");
}

class DynamicLibraryBackend final : public DeviceBackend {
public:
  explicit DynamicLibraryBackend(const std::string &library_path) {
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
    std::string first_error;
    for (Device candidate : {Device::GPU, Device::CPU}) {
      try {
        instance_ = api_->create(static_cast<uint32_t>(candidate));
        if (instance_ != nullptr) {
          device_ = static_cast<Device>(api_->device(instance_));
          break;
        }
      } catch (const std::exception &e) {
        if (first_error.empty()) {
          first_error = e.what();
        }
      }
    }
    if (instance_ == nullptr) {
      dlclose(library_handle_);
      library_handle_ = nullptr;
      std::string msg = "DynamicLibraryBackend: backend create failed";
      if (!first_error.empty()) {
        msg += ": " + first_error;
      }
      throw std::runtime_error(msg);
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

  Device device() const override { return device_; }

  void free(void *ptr) override { api_->free(instance_, ptr); }

  DeviceMemoryInfo memory_info() const override {
    if (api_->memory_info == nullptr) {
      return {};
    }
    const BackendMemoryInfo info = api_->memory_info(instance_);
    return DeviceMemoryInfo{info.available != 0u, info.free_bytes,
                            info.total_bytes};
  }

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

  float sum_squares_f32(const TensorView &x) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    return api_->sum_squares_f32(instance_, &x_view);
  }

  void relu(const TensorView &x, TensorView &out) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView out_view = to_backend_tensor_view(out);
    api_->relu(instance_, &x_view, &out_view);
  }

  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) override {
    const BackendTensorView preact_view = to_backend_tensor_view(preact);
    const BackendTensorView dout_view = to_backend_tensor_view(dout);
    const BackendTensorView dx_view = to_backend_tensor_view(dx);
    api_->relu_backward(instance_, &preact_view, &dout_view, &dx_view);
  }

  void row_sum(const TensorView &x, TensorView &out_1xC) override {
    const BackendTensorView x_view = to_backend_tensor_view(x);
    const BackendTensorView out_view = to_backend_tensor_view(out_1xC);
    api_->row_sum(instance_, &x_view, &out_view);
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

  void accumulate_embedding_grads(const TensorView &ids,
                                  const TensorView &d_cur, TensorView &d_tok,
                                  TensorView &d_pos) override {
    const BackendTensorView ids_view = to_backend_tensor_view(ids);
    const BackendTensorView d_cur_view = to_backend_tensor_view(d_cur);
    const BackendTensorView d_tok_view = to_backend_tensor_view(d_tok);
    const BackendTensorView d_pos_view = to_backend_tensor_view(d_pos);
    api_->accumulate_embedding_grads(instance_, &ids_view, &d_cur_view,
                                     &d_tok_view, &d_pos_view);
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    const BackendTensorView logits_view = to_backend_tensor_view(logits);
    const BackendTensorView targets_view = to_backend_tensor_view(targets);
    const BackendTensorView out_loss_view = to_backend_tensor_view(out_loss);
    api_->cross_entropy_mean(instance_, &logits_view, &targets_view,
                             &out_loss_view);
  }

  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) override {
    const BackendTensorView logits_view = to_backend_tensor_view(logits);
    const BackendTensorView targets_view = to_backend_tensor_view(targets);
    const BackendTensorView out_loss_view = to_backend_tensor_view(out_loss);
    api_->cross_entropy_mean_backward_inplace(instance_, &logits_view,
                                              &targets_view, &out_loss_view);
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

  void softmax_backward_rows(const TensorView &softmax,
                             const TensorView &dout,
                             TensorView &dx) override {
    const BackendTensorView softmax_view = to_backend_tensor_view(softmax);
    const BackendTensorView dout_view = to_backend_tensor_view(dout);
    const BackendTensorView dx_view = to_backend_tensor_view(dx);
    api_->softmax_backward_rows(instance_, &softmax_view, &dout_view,
                                &dx_view);
  }

  void apply_causal_mask_inplace(TensorView &scores, float neg_inf) override {
    const BackendTensorView scores_view = to_backend_tensor_view(scores);
    api_->apply_causal_mask_inplace(instance_, &scores_view, neg_inf);
  }

  void adamw_step(TensorView &params, const TensorView &grads, TensorView &m,
                  TensorView &v, uint64_t step, float learning_rate,
                  float beta1, float beta2, float weight_decay,
                  bool apply_weight_decay) override {
    const BackendTensorView params_view = to_backend_tensor_view(params);
    const BackendTensorView grads_view = to_backend_tensor_view(grads);
    const BackendTensorView m_view = to_backend_tensor_view(m);
    const BackendTensorView v_view = to_backend_tensor_view(v);
    api_->adamw_step(instance_, &params_view, &grads_view, &m_view, &v_view,
                     step, learning_rate, beta1, beta2, weight_decay,
                     apply_weight_decay ? 1u : 0u);
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
  Device device_ = Device::CPU;
};
}

Device DefaultCpuBackend::device() const { return Device::CPU; }

void *DefaultCpuBackend::alloc(uint64_t bytes, uint32_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument("DefaultCpuBackend::alloc: invalid alignment");
  }

  const uint64_t extra =
      static_cast<uint64_t>(alignment - 1) + sizeof(AllocationHeader);
  void *base = std::malloc(static_cast<size_t>(bytes + extra));
  if (base == nullptr) {
    throw std::runtime_error("DefaultCpuBackend::alloc failed");
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

void DefaultCpuBackend::free(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  auto *header = reinterpret_cast<AllocationHeader *>(
      reinterpret_cast<std::uintptr_t>(ptr) - sizeof(AllocationHeader));
  std::free(header->base);
}

DeviceMemoryInfo DefaultCpuBackend::memory_info() const { return {}; }

void DefaultCpuBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("DefaultCpuBackend::copy_host2device: null pointer");
  }
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void DefaultCpuBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw std::invalid_argument("DefaultCpuBackend::copy_device2host: null pointer");
  }
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void DefaultCpuBackend::copy(const TensorView &src, TensorView &dst) {
  if (src.is_contiguous() && dst.is_contiguous() && src.dtype() == DType::F32 &&
      dst.dtype() == DType::F32) {
    const uint64_t n = src.numel();
    if (n != dst.numel()) {
      throw std::runtime_error("DefaultCpuBackend::copy: numel mismatch");
    }
    std::memcpy(dst.data(), src.data(), static_cast<size_t>(src.bytes()));
    return;
  }
  const int64_t row_count = src.shape().dim(0);
  const int64_t col_count = src.shape().dim(1);
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32(dst, r, c, CpuMemOperations::load_f32(src, r, c));
    }
  }
}

void DefaultCpuBackend::fill(TensorView &t, float v) {
  if (t.is_contiguous() && t.dtype() == DType::F32) {
    float *p = reinterpret_cast<float *>(t.data());
    const uint64_t n = t.numel();
    for (uint64_t i = 0; i < n; ++i) {
      p[i] = v;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(t, 1);
  const int64_t col_count = t.dim(t.rank() - 1);
  for (uint64_t p = 0; p < prefix_count; ++p) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32_prefix_last1(t, p, c, v);
    }
  }
}

void DefaultCpuBackend::add(const TensorView &a, const TensorView &b, TensorView &out) {
  if (a.is_contiguous() && b.is_contiguous() && out.is_contiguous() &&
      a.dtype() == DType::F32 && b.dtype() == DType::F32 &&
      out.dtype() == DType::F32) {
    const float *ap = reinterpret_cast<const float *>(a.data());
    const float *bp = reinterpret_cast<const float *>(b.data());
    float *op = reinterpret_cast<float *>(out.data());
    const uint64_t total = a.numel();
    if (total == b.numel() && total == out.numel()) {
      for (uint64_t i = 0; i < total; ++i) {
        op[i] = ap[i] + bp[i];
      }
      return;
    }
    const int64_t last_dim = a.rank() == 0 ? 1 : a.dim(a.rank() - 1);
    if (a.rank() == 3 && b.rank() == 2 && out.rank() == 3 &&
        a.dim(1) == b.dim(0) && a.dim(2) == b.dim(1) &&
        out.dim(0) == a.dim(0) && out.dim(1) == a.dim(1) &&
        out.dim(2) == a.dim(2) && last_dim > 0 &&
        b.numel() == static_cast<uint64_t>(a.dim(1) * a.dim(2)) &&
        total == out.numel()) {
      const uint64_t plane =
          static_cast<uint64_t>(a.dim(1)) * static_cast<uint64_t>(a.dim(2));
      for (uint64_t i = 0; i < total; ++i) {
        op[i] = ap[i] + bp[i % plane];
      }
      return;
    }
    throw std::runtime_error("DefaultCpuBackend::add: numel or broadcast shape mismatch");
  }
  if (a.rank() == 3 && b.rank() == 2 && out.rank() == 3 &&
      a.dim(1) == b.dim(0) && a.dim(2) == b.dim(1) &&
      out.dim(0) == a.dim(0) && out.dim(1) == a.dim(1) &&
      out.dim(2) == a.dim(2)) {
    const int64_t seq_len = a.dim(1);
    const int64_t col_count = a.dim(2);
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 1);
    for (uint64_t p = 0; p < prefix_count; ++p) {
      const int64_t semantic_row =
          static_cast<int64_t>(p % static_cast<uint64_t>(seq_len));
      for (int64_t c = 0; c < col_count; ++c) {
        const float av = CpuMemOperations::load_f32_prefix_last1(a, p, c);
        const float bv = CpuMemOperations::load_f32(b, semantic_row, c);
        CpuMemOperations::store_f32_prefix_last1(out, p, c, av + bv);
      }
    }
    return;
  }
  const uint64_t an = a.numel();
  if (an != b.numel() || an != out.numel()) {
    throw std::runtime_error("DefaultCpuBackend::add: numel mismatch");
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 1);
  const int64_t col_count = a.dim(a.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32_prefix_last1(
          out, r, c, CpuMemOperations::load_f32_prefix_last1(a, r, c) +
                         CpuMemOperations::load_f32_prefix_last1(b, r, c));
    }
  }
}

void DefaultCpuBackend::add_inplace(TensorView &a, const TensorView &b) {
  if (a.is_contiguous() && b.is_contiguous() && a.dtype() == DType::F32 &&
      b.dtype() == DType::F32) {
    const uint64_t n = a.numel();
    if (n != b.numel()) {
      throw std::runtime_error("DefaultCpuBackend::add_inplace: numel mismatch");
    }
    float *ap = reinterpret_cast<float *>(a.data());
    const float *bp = reinterpret_cast<const float *>(b.data());
    for (uint64_t i = 0; i < n; ++i) {
      ap[i] += bp[i];
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 1);
  const int64_t col_count = a.dim(a.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32_prefix_last1(
          a, r, c, CpuMemOperations::load_f32_prefix_last1(a, r, c) +
                       CpuMemOperations::load_f32_prefix_last1(b, r, c));
    }
  }
}

void DefaultCpuBackend::add_bias_rowwise(const TensorView &x,
                                  const TensorView &bias_1xC,
                                  TensorView &out) {
  if (x.is_contiguous() && bias_1xC.is_contiguous() && out.is_contiguous() &&
      x.dtype() == DType::F32 && bias_1xC.dtype() == DType::F32 &&
      out.dtype() == DType::F32) {
    const int64_t col_count = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
    const uint64_t total = x.numel();
    if (col_count <= 0 ||
        total % static_cast<uint64_t>(col_count) != 0 ||
        bias_1xC.numel() != static_cast<uint64_t>(col_count) ||
        out.numel() != total) {
      throw std::runtime_error("DefaultCpuBackend::add_bias_rowwise: semantic shape mismatch");
    }
    const float *xp = reinterpret_cast<const float *>(x.data());
    const float *bp = reinterpret_cast<const float *>(bias_1xC.data());
    float *op = reinterpret_cast<float *>(out.data());
    for (uint64_t i = 0; i < total; ++i) {
      op[i] = xp[i] + bp[i % static_cast<uint64_t>(col_count)];
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t col_count = x.dim(x.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32_prefix_last1(
          out, r, c,
          CpuMemOperations::load_f32_prefix_last1(x, r, c) +
              CpuMemOperations::load_f32(bias_1xC, 0, c));
    }
  }
}

void DefaultCpuBackend::mul_scalar(const TensorView &x, float s, TensorView &out) {
  if (x.is_contiguous() && out.is_contiguous() && x.dtype() == DType::F32 &&
      out.dtype() == DType::F32) {
    const uint64_t n = x.numel();
    if (n != out.numel()) {
      throw std::runtime_error("DefaultCpuBackend::mul_scalar: numel mismatch");
    }
    const float *xp = reinterpret_cast<const float *>(x.data());
    float *op = reinterpret_cast<float *>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
      op[i] = xp[i] * s;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t col_count = x.dim(x.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32_prefix_last1(
          out, r, c, CpuMemOperations::load_f32_prefix_last1(x, r, c) * s);
    }
  }
}

float DefaultCpuBackend::sum_squares_f32(const TensorView &x) {
  require_f32_cpu_contig(x, "DefaultCpuBackend::sum_squares_f32(x)");
  const float *p = reinterpret_cast<const float *>(x.data());
  const uint64_t n = x.numel();
  double sum_sq = 0.0;
  for (uint64_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(p[i]);
    sum_sq += v * v;
  }
  return static_cast<float>(sum_sq);
}

void DefaultCpuBackend::relu(const TensorView &x, TensorView &out) {
  if (x.is_contiguous() && out.is_contiguous() && x.dtype() == DType::F32 &&
      out.dtype() == DType::F32) {
    const uint64_t n = x.numel();
    if (n != out.numel()) {
      throw std::runtime_error("DefaultCpuBackend::relu: numel mismatch");
    }
    const float *xp = reinterpret_cast<const float *>(x.data());
    float *op = reinterpret_cast<float *>(out.data());
    for (uint64_t i = 0; i < n; ++i) {
      op[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t col_count = x.dim(x.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      const float value = CpuMemOperations::load_f32_prefix_last1(x, r, c);
      CpuMemOperations::store_f32_prefix_last1(out, r, c,
                                     value > 0.0f ? value : 0.0f);
    }
  }
}

void DefaultCpuBackend::relu_backward(const TensorView &preact, const TensorView &dout,
                               TensorView &dx) {
  if (preact.is_contiguous() && dout.is_contiguous() && dx.is_contiguous() &&
      preact.dtype() == DType::F32 && dout.dtype() == DType::F32 &&
      dx.dtype() == DType::F32) {
    const uint64_t n = preact.numel();
    if (n != dout.numel() || n != dx.numel()) {
      throw std::runtime_error("DefaultCpuBackend::relu_backward: numel mismatch");
    }
    const float *pp = reinterpret_cast<const float *>(preact.data());
    const float *dp = reinterpret_cast<const float *>(dout.data());
    float *xp = reinterpret_cast<float *>(dx.data());
    for (uint64_t i = 0; i < n; ++i) {
      xp[i] = pp[i] > 0.0f ? dp[i] : 0.0f;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(preact, 1);
  const int64_t col_count = preact.dim(preact.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      const float g = CpuMemOperations::load_f32_prefix_last1(preact, r, c) > 0.0f
                          ? CpuMemOperations::load_f32_prefix_last1(dout, r, c)
                          : 0.0f;
      CpuMemOperations::store_f32_prefix_last1(dx, r, c, g);
    }
  }
}

void DefaultCpuBackend::row_sum(const TensorView &x, TensorView &out_1xC) {
  if (x.is_contiguous() && out_1xC.is_contiguous() && x.dtype() == DType::F32 &&
      out_1xC.dtype() == DType::F32) {
    const int64_t col_count = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
    const uint64_t total = x.numel();
    if (col_count <= 0 ||
        total % static_cast<uint64_t>(col_count) != 0 ||
        out_1xC.numel() != static_cast<uint64_t>(col_count)) {
      throw std::runtime_error("DefaultCpuBackend::row_sum: semantic shape mismatch");
    }
    const uint64_t prefix_count = total / static_cast<uint64_t>(col_count);
    const float *xp = reinterpret_cast<const float *>(x.data());
    float *op = reinterpret_cast<float *>(out_1xC.data());
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < prefix_count; ++p) {
        acc += xp[p * static_cast<uint64_t>(col_count) +
                  static_cast<uint64_t>(c)];
      }
      op[c] = acc;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t col_count = x.dim(x.rank() - 1);
  for (int64_t c = 0; c < col_count; ++c) {
    float acc = 0.0f;
    for (uint64_t r = 0; r < prefix_count; ++r) {
      acc += CpuMemOperations::load_f32_prefix_last1(x, r, c);
    }
    CpuMemOperations::store_f32(out_1xC, 0, c, acc);
  }
}

void DefaultCpuBackend::matmul(const TensorView &a, const TensorView &b,
                        TensorView &out) {
  if (a.rank() >= 3 && out.rank() == a.rank() &&
      (b.rank() == a.rank() || b.rank() == 2)) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
    const int64_t m = a.dim(a.rank() - 2);
    const int64_t k = a.dim(a.rank() - 1);
    const int64_t n = b.dim(b.rank() - 1);
    for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
      for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < n; ++c) {
          float acc = 0.0f;
          for (int64_t kk = 0; kk < k; ++kk) {
            acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, r, kk) *
                   (b.rank() == 2 ? CpuMemOperations::load_f32(b, kk, c)
                                  : CpuMemOperations::load_f32_prefix_last2(b, prefix, kk, c));
          }
          CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
        }
      }
    }
    return;
  }
  const int64_t row_count = a.shape().dim(0);
  const int64_t inner_dim = a.shape().dim(1);
  const int64_t col_count = b.shape().dim(1);
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < inner_dim; ++k) {
        acc += CpuMemOperations::load_f32(a, r, k) * CpuMemOperations::load_f32(b, k, c);
      }
      CpuMemOperations::store_f32(out, r, c, acc);
    }
  }
}

void DefaultCpuBackend::matmul_left_transposed(const TensorView &a, const TensorView &b,
                                        TensorView &out) {
  if (a.rank() >= 3 && b.rank() == a.rank()) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
    const int64_t shared_dim = a.dim(a.rank() - 2);
    const int64_t out_rows = a.dim(a.rank() - 1);
    const int64_t out_cols = b.dim(b.rank() - 1);
    for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
      for (int64_t r = 0; r < out_rows; ++r) {
        for (int64_t c = 0; c < out_cols; ++c) {
          float acc = 0.0f;
          for (int64_t k = 0; k < shared_dim; ++k) {
            acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, k, r) *
                   CpuMemOperations::load_f32_prefix_last2(b, prefix, k, c);
          }
          if (out.rank() == 2) {
            CpuMemOperations::store_f32(out, r, c,
                              CpuMemOperations::load_f32(out, r, c) + acc);
          } else {
            CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
          }
        }
      }
    }
    return;
  }
  const int64_t row_count = a.shape().dim(1);
  const int64_t inner_dim = a.shape().dim(0);
  const int64_t col_count = b.shape().dim(1);
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < inner_dim; ++k) {
        acc += CpuMemOperations::load_f32(a, k, r) * CpuMemOperations::load_f32(b, k, c);
      }
      CpuMemOperations::store_f32(out, r, c, acc);
    }
  }
}

void DefaultCpuBackend::matmul_right_transposed(const TensorView &a, const TensorView &b,
                                         TensorView &out) {
  if (a.rank() >= 3 && out.rank() == a.rank() &&
      (b.rank() == a.rank() || b.rank() == 2)) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
    const int64_t m = a.dim(a.rank() - 2);
    const int64_t k = a.dim(a.rank() - 1);
    const int64_t n = b.rank() == 2 ? b.dim(0) : b.dim(b.rank() - 2);
    for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
      for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < n; ++c) {
          float acc = 0.0f;
          for (int64_t kk = 0; kk < k; ++kk) {
            acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, r, kk) *
                   (b.rank() == 2 ? CpuMemOperations::load_f32(b, c, kk)
                                  : CpuMemOperations::load_f32_prefix_last2(b, prefix, c, kk));
          }
          CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
        }
      }
    }
    return;
  }
  const int64_t row_count = a.shape().dim(0);
  const int64_t inner_dim = a.shape().dim(1);
  const int64_t col_count = b.shape().dim(0);
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      float acc = 0.0f;
      for (int64_t k = 0; k < inner_dim; ++k) {
        acc += CpuMemOperations::load_f32(a, r, k) * CpuMemOperations::load_f32(b, c, k);
      }
      CpuMemOperations::store_f32(out, r, c, acc);
    }
  }
}

void DefaultCpuBackend::transpose(const TensorView &x, TensorView &out) {
  const int64_t row_count = x.shape().dim(0);
  const int64_t col_count = x.shape().dim(1);
  for (int64_t r = 0; r < row_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32(out, c, r, CpuMemOperations::load_f32(x, r, c));
    }
  }
}

void DefaultCpuBackend::layernorm_forward(const TensorView &x,
                                   const TensorView &gamma_1xC,
                                   const TensorView &beta_1xC,
                                   TensorView &out) {
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t model_dim = x.dim(x.rank() - 1);
  const float eps = 1e-5f;
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    double mean = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      mean += CpuMemOperations::load_f32_prefix_last1(x, prefix, c);
    }
    mean /= static_cast<double>(model_dim);

    double var = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      const double d =
          static_cast<double>(CpuMemOperations::load_f32_prefix_last1(x, prefix, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(model_dim);
    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

    for (int64_t c = 0; c < model_dim; ++c) {
      const float xn = (CpuMemOperations::load_f32_prefix_last1(x, prefix, c) -
                        static_cast<float>(mean)) *
                       inv_std;
      const float y = xn * CpuMemOperations::load_f32(gamma_1xC, 0, c) +
                      CpuMemOperations::load_f32(beta_1xC, 0, c);
      CpuMemOperations::store_f32_prefix_last1(out, prefix, c, y);
    }
  }
}

void DefaultCpuBackend::layernorm_backward(const TensorView &x,
                                    const TensorView &gamma_1xC,
                                    const TensorView &dout, TensorView &dx,
                                    TensorView &dgamma_1xC,
                                    TensorView &dbeta_1xC) {
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t model_dim = x.dim(x.rank() - 1);
  const float eps = 1e-5f;
  for (int64_t c = 0; c < model_dim; ++c) {
    CpuMemOperations::store_f32(dgamma_1xC, 0, c, 0.0f);
    CpuMemOperations::store_f32(dbeta_1xC, 0, c, 0.0f);
  }
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    double mean = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      mean += CpuMemOperations::load_f32_prefix_last1(x, prefix, c);
    }
    mean /= static_cast<double>(model_dim);

    double var = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      const double d =
          static_cast<double>(CpuMemOperations::load_f32_prefix_last1(x, prefix, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(model_dim);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    double sum_dxhat = 0.0;
    double sum_dxhat_xhat = 0.0;
    for (int64_t c = 0; c < model_dim; ++c) {
      const double xhat =
          (static_cast<double>(CpuMemOperations::load_f32_prefix_last1(x, prefix, c)) - mean) *
          inv_std;
      const double g = static_cast<double>(CpuMemOperations::load_f32(gamma_1xC, 0, c));
      const double dyi =
          static_cast<double>(CpuMemOperations::load_f32_prefix_last1(dout, prefix, c));
      const double dxhat = dyi * g;
      sum_dxhat += dxhat;
      sum_dxhat_xhat += dxhat * xhat;
      CpuMemOperations::store_f32(dgamma_1xC, 0, c,
                        CpuMemOperations::load_f32(dgamma_1xC, 0, c) +
                            static_cast<float>(dyi * xhat));
      CpuMemOperations::store_f32(dbeta_1xC, 0, c,
                        CpuMemOperations::load_f32(dbeta_1xC, 0, c) +
                            static_cast<float>(dyi));
    }

    for (int64_t c = 0; c < model_dim; ++c) {
      const double xhat =
          (static_cast<double>(CpuMemOperations::load_f32_prefix_last1(x, prefix, c)) - mean) *
          inv_std;
      const double g = static_cast<double>(CpuMemOperations::load_f32(gamma_1xC, 0, c));
      const double dyi =
          static_cast<double>(CpuMemOperations::load_f32_prefix_last1(dout, prefix, c));
      const double dxhat = dyi * g;
      const double n = static_cast<double>(model_dim);
      const double dxi =
          (inv_std / n) * (n * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
      CpuMemOperations::store_f32_prefix_last1(dx, prefix, c, static_cast<float>(dxi));
    }
  }
}

void DefaultCpuBackend::embedding_lookup(const TensorView &table, const TensorView &ids,
                                  TensorView &out) {
  const int64_t vocab_size = table.dim(0);
  const int64_t model_dim = table.dim(1);
  const uint64_t token_count = ids.numel();
  const size_t out_prefix_rank = out.rank() - 1;
  const size_t out_last_axis = out.rank() - 1;
  for (uint64_t t = 0; t < token_count; ++t) {
    const int64_t idx = CpuMemOperations::load_index_linear(ids, t);
    if (idx < 0 || idx >= vocab_size) {
      throw std::runtime_error("DefaultCpuBackend::embedding_lookup: embedding id out of range");
    }
    uint8_t *dst =
        reinterpret_cast<uint8_t *>(out.data()) +
        CpuMemOperations::logical_offset_bytes_for_linear_index(out, t,
                                                               out_prefix_rank);
    for (int64_t d = 0; d < model_dim; ++d) {
      const float v = CpuMemOperations::load_f32(table, idx, d);
      std::memcpy(dst + d * out.stride_bytes(out_last_axis), &v, sizeof(float));
    }
  }
}

void DefaultCpuBackend::accumulate_embedding_grads(const TensorView &ids,
                                            const TensorView &d_cur,
                                            TensorView &d_tok,
                                            TensorView &d_pos) {
  const uint64_t token_count = ids.numel();
  const int64_t model_dim = d_cur.dim(d_cur.rank() - 1);
  const int64_t vocab_size = d_tok.dim(0);
  const int64_t seq_len = ids.dim(ids.rank() - 1);
  for (uint64_t t = 0; t < token_count; ++t) {
    const int64_t idx = CpuMemOperations::load_index_linear(ids, t);
    if (idx < 0 || idx >= vocab_size) {
      throw std::runtime_error(
          "DefaultCpuBackend::accumulate_embedding_grads: token id out of range");
    }
    const int64_t seq_pos = static_cast<int64_t>(t % static_cast<uint64_t>(seq_len));
    for (int64_t d = 0; d < model_dim; ++d) {
      const float g = CpuMemOperations::load_f32_prefix_last1(d_cur, t, d);
      CpuMemOperations::store_f32(d_tok, idx, d, CpuMemOperations::load_f32(d_tok, idx, d) + g);
      CpuMemOperations::store_f32(d_pos, seq_pos, d, CpuMemOperations::load_f32(d_pos, seq_pos, d) + g);
    }
  }
}

void DefaultCpuBackend::cross_entropy_mean(const TensorView &logits,
                                    const TensorView &targets,
                                    TensorView &out_loss) {
  const uint64_t token_count = targets.numel();
  const int64_t vocab_size = logits.dim(logits.rank() - 1);
  double sum = 0.0;
  for (uint64_t t = 0; t < token_count; ++t) {
    const int64_t target = CpuMemOperations::load_index_linear(targets, t);
    if (target < 0 || target >= vocab_size) {
      throw std::runtime_error("DefaultCpuBackend::cross_entropy_mean: target out of range");
    }
    float max_logit = CpuMemOperations::load_f32_prefix_last1(logits, t, 0);
    for (int64_t c = 1; c < vocab_size; ++c) {
      max_logit = std::max(max_logit, CpuMemOperations::load_f32_prefix_last1(logits, t, c));
    }
    double lse = 0.0;
    for (int64_t c = 0; c < vocab_size; ++c) {
      lse += std::exp(static_cast<double>(
          CpuMemOperations::load_f32_prefix_last1(logits, t, c) - max_logit));
    }
    const double log_denom = static_cast<double>(max_logit) + std::log(lse);
    sum += log_denom -
           static_cast<double>(CpuMemOperations::load_f32_prefix_last1(logits, t, target));
  }
  CpuMemOperations::store_f32(out_loss, 0, 0,
                    static_cast<float>(sum / static_cast<double>(token_count)));
}

void DefaultCpuBackend::cross_entropy_mean_backward_inplace(TensorView &logits,
                                                     const TensorView &targets,
                                                     TensorView &out_loss) {
  cross_entropy_mean(logits, targets, out_loss);
  backward_from_logits_targets(logits, targets);
}

float DefaultCpuBackend::read_scalar_f32(const TensorView &x) {
  return CpuMemOperations::load_f32(x, 0, 0);
}

void DefaultCpuBackend::backward_from_logits_targets(TensorView &logits,
                                              const TensorView &targets) {
  const uint64_t token_count = targets.numel();
  const int64_t vocab_size = logits.dim(logits.rank() - 1);
  const float inv_token_rows = 1.0f / static_cast<float>(token_count);
  for (uint64_t t = 0; t < token_count; ++t) {
    const int64_t target = CpuMemOperations::load_index_linear(targets, t);
    if (target < 0 || target >= vocab_size) {
      throw std::runtime_error("DefaultCpuBackend::backward_from_logits_targets: target out of range");
    }
    float max_logit = CpuMemOperations::load_f32_prefix_last1(logits, t, 0);
    for (int64_t c = 1; c < vocab_size; ++c) {
      max_logit = std::max(max_logit, CpuMemOperations::load_f32_prefix_last1(logits, t, c));
    }
    double sum = 0.0;
    for (int64_t c = 0; c < vocab_size; ++c) {
      sum += std::exp(static_cast<double>(
          CpuMemOperations::load_f32_prefix_last1(logits, t, c) - max_logit));
    }
    if (sum <= 0.0) {
      throw std::runtime_error("DefaultCpuBackend::backward_from_logits_targets: softmax sum <= 0");
    }
    for (int64_t c = 0; c < vocab_size; ++c) {
      const float p = static_cast<float>(
          std::exp(static_cast<double>(
                       CpuMemOperations::load_f32_prefix_last1(logits, t, c) - max_logit)) /
          sum);
      float gradient = p;
      if (c == target) {
        gradient -= 1.0f;
      }
      CpuMemOperations::store_f32_prefix_last1(logits, t, c, gradient * inv_token_rows);
    }
  }
}

void DefaultCpuBackend::softmax_rows(const TensorView &x, TensorView &out) {
  if (x.rank() >= 2 && out.rank() == x.rank()) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
    const int64_t col_count = x.dim(x.rank() - 1);
    for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
      float max_value = CpuMemOperations::load_f32_prefix_last1(x, prefix, 0);
      for (int64_t c = 1; c < col_count; ++c) {
        max_value =
            std::max(max_value, CpuMemOperations::load_f32_prefix_last1(x, prefix, c));
      }
      double sum = 0.0;
      for (int64_t c = 0; c < col_count; ++c) {
        const float exponent = std::exp(
            CpuMemOperations::load_f32_prefix_last1(x, prefix, c) - max_value);
        CpuMemOperations::store_f32_prefix_last1(out, prefix, c, exponent);
        sum += static_cast<double>(exponent);
      }
      if (sum <= 0.0) {
        throw std::runtime_error("DefaultCpuBackend::softmax_rows: softmax sum <= 0");
      }
      const float inv_sum = static_cast<float>(1.0 / sum);
      for (int64_t c = 0; c < col_count; ++c) {
        CpuMemOperations::store_f32_prefix_last1(
            out, prefix, c,
            CpuMemOperations::load_f32_prefix_last1(out, prefix, c) * inv_sum);
      }
    }
    return;
  }
  const int64_t row_count = x.shape().dim(0);
  const int64_t col_count = x.shape().dim(1);
  for (int64_t r = 0; r < row_count; ++r) {
    float max_value = CpuMemOperations::load_f32(x, r, 0);
    for (int64_t c = 1; c < col_count; ++c) {
      max_value = std::max(max_value, CpuMemOperations::load_f32(x, r, c));
    }
    double sum = 0.0;
    for (int64_t c = 0; c < col_count; ++c) {
      const float exponent = std::exp(CpuMemOperations::load_f32(x, r, c) - max_value);
      CpuMemOperations::store_f32(out, r, c, exponent);
      sum += static_cast<double>(exponent);
    }
    if (sum <= 0.0) {
      throw std::runtime_error("DefaultCpuBackend::softmax_rows: softmax sum <= 0");
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int64_t c = 0; c < col_count; ++c) {
      CpuMemOperations::store_f32(out, r, c,
                        CpuMemOperations::load_f32(out, r, c) * inv_sum);
    }
  }
}

void DefaultCpuBackend::softmax_backward_rows(const TensorView &softmax,
                                       const TensorView &dout,
                                       TensorView &dx) {
  if (softmax.rank() >= 2 && dx.rank() == softmax.rank()) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(softmax, 1);
    const int64_t col_count = softmax.dim(softmax.rank() - 1);
    for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
      float dot = 0.0f;
      for (int64_t c = 0; c < col_count; ++c) {
        dot += CpuMemOperations::load_f32_prefix_last1(softmax, prefix, c) *
               CpuMemOperations::load_f32_prefix_last1(dout, prefix, c);
      }
      for (int64_t c = 0; c < col_count; ++c) {
        const float s = CpuMemOperations::load_f32_prefix_last1(softmax, prefix, c);
        const float g = s * (CpuMemOperations::load_f32_prefix_last1(dout, prefix, c) - dot);
        CpuMemOperations::store_f32_prefix_last1(dx, prefix, c, g);
      }
    }
    return;
  }
  const int64_t row_count = softmax.shape().dim(0);
  const int64_t col_count = softmax.shape().dim(1);
  for (int64_t r = 0; r < row_count; ++r) {
    float dot = 0.0f;
    for (int64_t c = 0; c < col_count; ++c) {
      dot += CpuMemOperations::load_f32(softmax, r, c) * CpuMemOperations::load_f32(dout, r, c);
    }
    for (int64_t c = 0; c < col_count; ++c) {
      const float s = CpuMemOperations::load_f32(softmax, r, c);
      const float g = s * (CpuMemOperations::load_f32(dout, r, c) - dot);
      CpuMemOperations::store_f32(dx, r, c, g);
    }
  }
}

void DefaultCpuBackend::apply_causal_mask_inplace(TensorView &scores, float neg_inf) {
  if (scores.rank() >= 2) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(scores, 2);
    const int64_t token_rows = scores.dim(scores.rank() - 2);
    const int64_t token_cols = scores.dim(scores.rank() - 1);
    require_backend(token_rows == token_cols,
                    "DefaultCpuBackend::apply_causal_mask_inplace: scores must end with [T,T]");
    for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
      for (int64_t i = 0; i < token_rows; ++i) {
        for (int64_t j = i + 1; j < token_cols; ++j) {
          CpuMemOperations::store_f32_prefix_last2(scores, prefix, i, j, neg_inf);
        }
      }
    }
    return;
  }
  const int64_t token_rows = scores.shape().dim(0);
  for (int64_t i = 0; i < token_rows; ++i) {
    for (int64_t j = i + 1; j < token_rows; ++j) {
      CpuMemOperations::store_f32(scores, i, j, neg_inf);
    }
  }
}

void DefaultCpuBackend::adamw_step(TensorView &params, const TensorView &grads,
                            TensorView &m, TensorView &v, uint64_t step,
                            float learning_rate, float beta1, float beta2,
                            float weight_decay, bool apply_weight_decay) {
  require_backend(step >= 1, "DefaultCpuBackend::adamw_step: step must be >= 1");
  require_f32_cpu_contig(params, "DefaultCpuBackend::adamw_step(params)");
  require_f32_cpu_contig(grads, "DefaultCpuBackend::adamw_step(grads)");
  require_f32_cpu_contig(m, "DefaultCpuBackend::adamw_step(m)");
  require_f32_cpu_contig(v, "DefaultCpuBackend::adamw_step(v)");

  require_backend(params.shape().dim(0) == grads.shape().dim(0) &&
                      params.shape().dim(1) == grads.shape().dim(1),
                  "DefaultCpuBackend::adamw_step: params/grads shape mismatch");
  require_backend(params.shape().dim(0) == m.shape().dim(0) &&
                      params.shape().dim(1) == m.shape().dim(1),
                  "DefaultCpuBackend::adamw_step: params/m shape mismatch");
  require_backend(params.shape().dim(0) == v.shape().dim(0) &&
                      params.shape().dim(1) == v.shape().dim(1),
                  "DefaultCpuBackend::adamw_step: params/v shape mismatch");
  require_backend(learning_rate > 0.0f,
                  "DefaultCpuBackend::adamw_step: learning_rate must be > 0");
  require_backend(beta1 >= 0.0f && beta1 < 1.0f,
                  "DefaultCpuBackend::adamw_step: beta1 must be in [0,1)");
  require_backend(beta2 >= 0.0f && beta2 < 1.0f,
                  "DefaultCpuBackend::adamw_step: beta2 must be in [0,1)");

  const int64_t n = params.shape().dim(0) * params.shape().dim(1);
  const double t = static_cast<double>(step);
  const float b1_corr =
      1.0f - static_cast<float>(std::pow(static_cast<double>(beta1), t));
  const float b2_corr =
      1.0f - static_cast<float>(std::pow(static_cast<double>(beta2), t));
  require_backend(b1_corr > 0.0f && b2_corr > 0.0f,
                  "DefaultCpuBackend::adamw_step: invalid bias correction");

  float *p = params.f32();
  const float *g = grads.f32();
  float *m1 = m.f32();
  float *m2 = v.f32();
  constexpr float kEps = 1e-8f;
  for (int64_t i = 0; i < n; ++i) {
    const float grad = g[i];
    m1[i] = beta1 * m1[i] + (1.0f - beta1) * grad;
    m2[i] = beta2 * m2[i] + (1.0f - beta2) * grad * grad;

    const float mhat = m1[i] / b1_corr;
    const float vhat = m2[i] / b2_corr;
    const float adam = mhat / (std::sqrt(vhat) + kEps);
    const float decay = apply_weight_decay ? (weight_decay * p[i]) : 0.0f;
    p[i] -= learning_rate * (adam + decay);
  }
}

bool DefaultCpuBackend::is_file2device_read_supported() const { return true; }

void DefaultCpuBackend::read_file2device(const std::string &path, void *dst,
                                  uint64_t size, uint64_t file_offset) {
  if (size == 0) {
    return;
  }
  if (dst == nullptr) {
    throw std::invalid_argument("DefaultCpuBackend::read_file2device: null destination");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("DefaultCpuBackend::read_file2device: failed to open file");
  }
  in.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
  if (!in) {
    throw std::runtime_error("DefaultCpuBackend::read_file2device: failed to seek file");
  }
  in.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(size));
  if (in.gcount() != static_cast<std::streamsize>(size)) {
    throw std::runtime_error("DefaultCpuBackend::read_file2device: short read");
  }
}

std::unique_ptr<DeviceBackend> DeviceBackend::create_instance(const Config &cfg) {
  if (!cfg.backend.library.empty()) {
    return std::make_unique<DynamicLibraryBackend>(cfg.backend.library);
  }
  return std::make_unique<DefaultCpuBackend>();
}
