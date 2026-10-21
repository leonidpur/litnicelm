#include "device_backend.hpp"
#include "backend_plugin_api.hpp"
#include "backend_events.hpp"

#include <litnicelm_utils/bf16.hpp>
#include <litnicelm_utils/checks.hpp>
#include <litnicelm_utils/log.hpp>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
bool is_power_of_two(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

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

  // Capability: token ids and targets are I32.
  static void check_index_dtype(const TensorView &t, std::string_view who) {
    checker().check(CheckKind::Capability, t.dtype() == DType::I32, who,
                    "index tensor must be i32");
  }

  static int64_t logical_offset_bytes_for_linear_index(const TensorView &t,
                                                       uint64_t linear_index,
                                                       size_t axis_count) {
    int64_t offset = 0;
    for (size_t axis = axis_count; axis > 0; --axis) {
      const int64_t dim = t.dim(axis - 1);
      checker().check(CheckKind::Layout, dim > 0,
                      "DeviceBackend", "invalid non-positive dimension");
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
    int32_t out;
    std::memcpy(&out, p, sizeof(int32_t));
    return static_cast<int64_t>(out);
  }

  static uint64_t logical_prefix_count(const TensorView &t, size_t suffix_rank) {
    checker().check(CheckKind::Layout, t.rank() >= suffix_rank,
                    "DeviceBackend", "tensor rank smaller than suffix rank");
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


class DynamicLibraryBackend final : public DeviceBackend {
public:
  explicit DynamicLibraryBackend(const std::string &library_path) {
    library_handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_handle_ == nullptr) {
      throw std::runtime_error("DynamicLibraryBackend: failed to load " +
                               library_path + ": " + dlerror());
    }
    logger().log(backend_events::kBackendLoaded, library_path);

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

  uint32_t precision_variants() const override {
    return api_->precision_variants(instance_);
  }

  void copy_host2device(void *dst, const void *src, uint64_t bytes) override {
    api_->copy_host2device(instance_, dst, src, bytes);
  }

  void copy_device2host(void *dst, const void *src, uint64_t bytes) override {
    api_->copy_device2host(instance_, dst, src, bytes);
  }

  void copy(const TensorView &src, TensorView &dst) override {
    api_->copy(instance_, &src, &dst);
  }

  void cast(const TensorView &src, TensorView &dst) override {
    api_->cast(instance_, &src, &dst);
  }

  void fill(TensorView &t, float v) override {
    api_->fill(instance_, &t, v);
  }

  void add(const TensorView &a, const TensorView &b, TensorView &out) override {
    api_->add(instance_, &a, &b, &out);
  }

  void add_inplace(TensorView &a, const TensorView &b) override {
    api_->add_inplace(instance_, &a, &b);
  }

  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override {
    api_->add_bias_rowwise(instance_, &x, &bias_1xC, &out);
  }

  void add_bias_relu_rowwise(const TensorView &x, const TensorView &bias_1xC,
                             TensorView &out) override {
    api_->add_bias_relu_rowwise(instance_, &x, &bias_1xC, &out);
  }

  void add_bias_relu_rowwise_inplace(TensorView &x,
                                     const TensorView &bias_1xC) override {
    api_->add_bias_relu_rowwise_inplace(instance_, &x, &bias_1xC);
  }

  void mul_scalar(const TensorView &x, float s, TensorView &out) override {
    api_->mul_scalar(instance_, &x, s, &out);
  }

  float sum_squares_f32(const TensorView &x) override {
    return api_->sum_squares_f32(instance_, &x);
  }

  void relu(const TensorView &x, TensorView &out) override {
    api_->relu(instance_, &x, &out);
  }

  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) override {
    api_->relu_backward(instance_, &preact, &dout, &dx);
  }

  void relu_backward_inplace(const TensorView &preact,
                             TensorView &dout_dx) override {
    api_->relu_backward_inplace(instance_, &preact, &dout_dx);
  }

  void row_sum(const TensorView &x, TensorView &out_1xC) override {
    api_->row_sum(instance_, &x, &out_1xC);
  }

  void gemm(const TensorView &a, const TensorView &b, TensorView &out) override {
    api_->gemm(instance_, &a, &b, &out);
  }

  void gemm_ranked_matrix_rhs(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    api_->gemm_ranked_matrix_rhs(instance_, &a, &b, &out);
  }

  void gemm_ranked_matrix_rhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) override {
    api_->gemm_ranked_matrix_rhs_t(instance_, &a, &b, &out);
  }

  void gemm_ranked_reduce_lhs_t(const TensorView &a, const TensorView &b,
                                TensorView &out) override {
    api_->gemm_ranked_reduce_lhs_t(instance_, &a, &b, &out);
  }

  void gemm_batched(const TensorView &a, const TensorView &b,
                    TensorView &out) override {
    api_->gemm_batched(instance_, &a, &b, &out);
  }

  void gemm_batched_lhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) override {
    api_->gemm_batched_lhs_t(instance_, &a, &b, &out);
  }

  void gemm_batched_rhs_t(const TensorView &a, const TensorView &b,
                          TensorView &out) override {
    api_->gemm_batched_rhs_t(instance_, &a, &b, &out);
  }

  void transpose(const TensorView &x, TensorView &out) override {
    api_->transpose(instance_, &x, &out);
  }

  void layernorm_forward(const TensorView &x, const TensorView &gamma_1xC,
                         const TensorView &beta_1xC, TensorView &out) override {
    api_->layernorm_forward(instance_, &x, &gamma_1xC, &beta_1xC,
                            &out);
  }

  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) override {
    api_->layernorm_backward(instance_, &x, &gamma_1xC, &dout,
                             &dx, &dgamma_1xC, &dbeta_1xC);
  }

  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) override {
    api_->embedding_lookup(instance_, &table, &ids, &out);
  }

  void accumulate_embedding_grads(const TensorView &ids,
                                  const TensorView &d_cur,
                                  TensorView &d_tok) override {
    api_->accumulate_embedding_grads(instance_, &ids, &d_cur,
                                     &d_tok);
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    api_->cross_entropy_mean(instance_, &logits, &targets,
                             &out_loss);
  }

  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) override {
    api_->cross_entropy_mean_backward_inplace(instance_, &logits,
                                              &targets, &out_loss);
  }

  float read_scalar_f32(const TensorView &x) override {
    return api_->read_scalar_f32(instance_, &x);
  }

  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override {
    api_->backward_from_logits_targets(instance_, &logits, &targets);
  }

  void softmax_rows(const TensorView &x, TensorView &out) override {
    api_->softmax_rows(instance_, &x, &out);
  }

  void softmax_backward_rows(const TensorView &softmax,
                             const TensorView &dout,
                             TensorView &dx) override {
    api_->softmax_backward_rows(instance_, &softmax, &dout,
                                &dx);
  }

  void scaled_causal_softmax_rows(const TensorView &scores, float scale,
                                  TensorView &out) override {
    api_->scaled_causal_softmax_rows(instance_, &scores, scale, &out);
  }

  bool supports_exec_context_iteration() const override {
    return api_->supports_exec_context_iteration != nullptr &&
           api_->supports_exec_context_iteration(instance_) != 0;
  }

  void start_exec_context_iteration() override {
    if (api_->start_exec_context_iteration == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context iteration is not supported");
    }
    api_->start_exec_context_iteration(instance_);
  }

  void finish_exec_context_iteration() override {
    if (api_->finish_exec_context_iteration == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context iteration is not supported");
    }
    api_->finish_exec_context_iteration(instance_);
  }

  void start_exec_context_group() override {
    if (api_->start_exec_context_group == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context group is not supported");
    }
    api_->start_exec_context_group(instance_);
  }

  void finish_exec_context_group() override {
    if (api_->finish_exec_context_group == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context group is not supported");
    }
    api_->finish_exec_context_group(instance_);
  }

  void gemm_batched_exec_context(const TensorView &a, const TensorView &b,
                                 TensorView &out) override {
    if (api_->gemm_batched_exec_context == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context batched gemm is not supported");
    }
    api_->gemm_batched_exec_context(instance_, &a, &b, &out);
  }

  void gemm_batched_rhs_t_exec_context(const TensorView &a,
                                       const TensorView &b,
                                       TensorView &out) override {
    if (api_->gemm_batched_rhs_t_exec_context == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context rhs-transposed batched gemm is not supported");
    }
    api_->gemm_batched_rhs_t_exec_context(instance_, &a, &b,
                                          &out);
  }

  void gemm_batched_lhs_t_exec_context(const TensorView &a,
                                       const TensorView &b,
                                       TensorView &out) override {
    if (api_->gemm_batched_lhs_t_exec_context == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context lhs-transposed batched gemm is not supported");
    }
    api_->gemm_batched_lhs_t_exec_context(instance_, &a, &b,
                                          &out);
  }

  void scaled_causal_softmax_rows_exec_context(
      const TensorView &scores, float scale, TensorView &out) override {
    if (api_->scaled_causal_softmax_rows_exec_context == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context causal softmax is not supported");
    }
    api_->scaled_causal_softmax_rows_exec_context(instance_, &scores,
                                                  scale, &out);
  }

  void softmax_backward_causal_rows_exec_context(const TensorView &softmax,
                                                 const TensorView &dout,
                                                 TensorView &dx) override {
    if (api_->softmax_backward_causal_rows_exec_context == nullptr) {
      throw std::runtime_error(
          "DynamicLibraryBackend: exec context causal softmax backward is not supported");
    }
    api_->softmax_backward_causal_rows_exec_context(
        instance_, &softmax, &dout, &dx);
  }

  void softmax_backward_causal_rows(const TensorView &softmax,
                                    const TensorView &dout,
                                    TensorView &dx) override {
    api_->softmax_backward_causal_rows(instance_, &softmax, &dout,
                                       &dx);
  }

  void apply_causal_mask_inplace(TensorView &scores, float neg_inf) override {
    api_->apply_causal_mask_inplace(instance_, &scores, neg_inf);
  }

  void rotary_embedding_inplace(TensorView &x, int64_t n_heads, float base,
                                bool inverse) override {
    api_->rotary_embedding_inplace(instance_, &x, n_heads, base,
                                   inverse ? 1u : 0u);
  }

  void adamw_step(TensorView &params, const TensorView &grads, TensorView &m,
                  TensorView &v, uint64_t step, float learning_rate,
                  float beta1, float beta2, float weight_decay,
                  bool apply_weight_decay) override {
    api_->adamw_step(instance_, &params, &grads, &m, &v,
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

uint32_t DefaultCpuBackend::precision_variants() const {
  return kPrecisionVariantF32;
}

void DefaultCpuBackend::copy_host2device(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  checker().check(CheckKind::Layout,
                  dst != nullptr && src != nullptr,
                  "DefaultCpuBackend copy_host2device", "null pointer");
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void DefaultCpuBackend::copy_device2host(void *dst, const void *src, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  checker().check(CheckKind::Layout,
                  dst != nullptr && src != nullptr,
                  "DefaultCpuBackend copy_device2host", "null pointer");
  std::memcpy(dst, src, static_cast<size_t>(bytes));
}

void DefaultCpuBackend::copy(const TensorView &src, TensorView &dst) {
  if (src.is_contiguous() && dst.is_contiguous() && src.dtype() == DType::F32 &&
      dst.dtype() == DType::F32) {
    const uint64_t n = src.numel();
    checker().check(CheckKind::Contract, n == dst.numel(),
                    "DefaultCpuBackend copy", "numel mismatch");
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

void DefaultCpuBackend::cast(const TensorView &src, TensorView &dst) {
  const char *who = "DefaultCpuBackend cast";
  checker().check_same_shape(src, dst, who);
  const bool to_bf16 = src.dtype() == DType::F32 && dst.dtype() == DType::BF16;
  const bool to_f32 = src.dtype() == DType::BF16 && dst.dtype() == DType::F32;
  checker().check(CheckKind::Capability, to_bf16 || to_f32, who,
                  "supports F32 -> BF16 and BF16 -> F32");
  if (src.rank() == 0) {
    return;
  }
  const size_t last = src.rank() - 1;
  const uint64_t rows = CpuMemOperations::logical_prefix_count(src, 1);
  const int64_t cols = src.dim(last);
  const auto *src_base = static_cast<const uint8_t *>(src.data());
  auto *dst_base = static_cast<uint8_t *>(dst.data());
  for (uint64_t r = 0; r < rows; ++r) {
    const uint8_t *s =
        src_base +
        CpuMemOperations::logical_offset_bytes_for_linear_index(src, r, last);
    uint8_t *d =
        dst_base +
        CpuMemOperations::logical_offset_bytes_for_linear_index(dst, r, last);
    for (int64_t c = 0; c < cols; ++c) {
      const uint8_t *from = s + c * src.stride_bytes(last);
      uint8_t *to = d + c * dst.stride_bytes(last);
      if (to_bf16) {
        float value;
        std::memcpy(&value, from, sizeof(value));
        const uint16_t bits = f32_to_bf16_bits(value);
        std::memcpy(to, &bits, sizeof(bits));
      } else {
        uint16_t bits;
        std::memcpy(&bits, from, sizeof(bits));
        const float value = bf16_bits_to_f32(bits);
        std::memcpy(to, &value, sizeof(value));
      }
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
  checker().check(CheckKind::Contract,
                  an == b.numel() && an == out.numel(),
                  "DefaultCpuBackend add", "numel mismatch");
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
    checker().check(CheckKind::Contract, n == b.numel(),
                    "DefaultCpuBackend add_inplace", "numel mismatch");
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
    checker().check(CheckKind::Contract,
                    col_count > 0 &&
                        total % static_cast<uint64_t>(col_count) == 0 &&
                        bias_1xC.numel() == static_cast<uint64_t>(col_count) &&
                        out.numel() == total,
                    "DefaultCpuBackend add_bias_rowwise",
                    "semantic shape mismatch");
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

void DefaultCpuBackend::add_bias_relu_rowwise(const TensorView &x,
                                              const TensorView &bias_1xC,
                                              TensorView &out) {
  if (x.is_contiguous() && bias_1xC.is_contiguous() && out.is_contiguous() &&
      x.dtype() == DType::F32 && bias_1xC.dtype() == DType::F32 &&
      out.dtype() == DType::F32) {
    const int64_t col_count = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
    const uint64_t total = x.numel();
    checker().check(CheckKind::Contract,
                    col_count > 0 &&
                        total % static_cast<uint64_t>(col_count) == 0 &&
                        bias_1xC.numel() == static_cast<uint64_t>(col_count) &&
                        out.numel() == total,
                    "DefaultCpuBackend add_bias_relu_rowwise",
                    "semantic shape mismatch");
    const float *xp = reinterpret_cast<const float *>(x.data());
    const float *bp = reinterpret_cast<const float *>(bias_1xC.data());
    float *op = reinterpret_cast<float *>(out.data());
    for (uint64_t i = 0; i < total; ++i) {
      const float value = xp[i] + bp[i % static_cast<uint64_t>(col_count)];
      op[i] = value > 0.0f ? value : 0.0f;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t col_count = x.dim(x.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      const float value = CpuMemOperations::load_f32_prefix_last1(x, r, c) +
                          CpuMemOperations::load_f32(bias_1xC, 0, c);
      CpuMemOperations::store_f32_prefix_last1(
          out, r, c, value > 0.0f ? value : 0.0f);
    }
  }
}

void DefaultCpuBackend::add_bias_relu_rowwise_inplace(
    TensorView &x, const TensorView &bias_1xC) {
  if (x.is_contiguous() && bias_1xC.is_contiguous() &&
      x.dtype() == DType::F32 && bias_1xC.dtype() == DType::F32) {
    const int64_t col_count = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
    const uint64_t total = x.numel();
    checker().check(CheckKind::Contract,
                    col_count > 0 &&
                        total % static_cast<uint64_t>(col_count) == 0 &&
                        bias_1xC.numel() == static_cast<uint64_t>(col_count),
                    "DefaultCpuBackend add_bias_relu_rowwise_inplace",
                    "semantic shape mismatch");
    float *xp = reinterpret_cast<float *>(x.data());
    const float *bp = reinterpret_cast<const float *>(bias_1xC.data());
    for (uint64_t i = 0; i < total; ++i) {
      const float value = xp[i] + bp[i % static_cast<uint64_t>(col_count)];
      xp[i] = value > 0.0f ? value : 0.0f;
    }
    return;
  }
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(x, 1);
  const int64_t col_count = x.dim(x.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      const float value = CpuMemOperations::load_f32_prefix_last1(x, r, c) +
                          CpuMemOperations::load_f32(bias_1xC, 0, c);
      CpuMemOperations::store_f32_prefix_last1(
          x, r, c, value > 0.0f ? value : 0.0f);
    }
  }
}

void DefaultCpuBackend::mul_scalar(const TensorView &x, float s, TensorView &out) {
  if (x.is_contiguous() && out.is_contiguous() && x.dtype() == DType::F32 &&
      out.dtype() == DType::F32) {
    const uint64_t n = x.numel();
    checker().check(CheckKind::Contract, n == out.numel(),
                    "DefaultCpuBackend mul_scalar", "numel mismatch");
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
  checker().check_supported(x, Device::CPU, DType::F32, true,
                            "DefaultCpuBackend sum_squares_f32 x");
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
    checker().check(CheckKind::Contract, n == out.numel(),
                    "DefaultCpuBackend relu", "numel mismatch");
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
    checker().check(CheckKind::Contract,
                    n == dout.numel() && n == dx.numel(),
                    "DefaultCpuBackend relu_backward", "numel mismatch");
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

void DefaultCpuBackend::relu_backward_inplace(const TensorView &preact,
                                              TensorView &dout_dx) {
  if (preact.is_contiguous() && dout_dx.is_contiguous() &&
      preact.dtype() == DType::F32 && dout_dx.dtype() == DType::F32) {
    const uint64_t n = preact.numel();
    checker().check(CheckKind::Contract, n == dout_dx.numel(),
                    "DefaultCpuBackend relu_backward_inplace",
                    "numel mismatch");
    const float *pp = reinterpret_cast<const float *>(preact.data());
    float *dp = reinterpret_cast<float *>(dout_dx.data());
    for (uint64_t i = 0; i < n; ++i) {
      dp[i] = pp[i] > 0.0f ? dp[i] : 0.0f;
    }
    return;
  }
  const uint64_t prefix_count =
      CpuMemOperations::logical_prefix_count(preact, 1);
  const int64_t col_count = preact.dim(preact.rank() - 1);
  for (uint64_t r = 0; r < prefix_count; ++r) {
    for (int64_t c = 0; c < col_count; ++c) {
      const float g =
          CpuMemOperations::load_f32_prefix_last1(preact, r, c) > 0.0f
              ? CpuMemOperations::load_f32_prefix_last1(dout_dx, r, c)
              : 0.0f;
      CpuMemOperations::store_f32_prefix_last1(dout_dx, r, c, g);
    }
  }
}

void DefaultCpuBackend::row_sum(const TensorView &x, TensorView &out_1xC) {
  if (x.is_contiguous() && out_1xC.is_contiguous() && x.dtype() == DType::F32 &&
      out_1xC.dtype() == DType::F32) {
    const int64_t col_count = x.rank() == 0 ? 1 : x.dim(x.rank() - 1);
    const uint64_t total = x.numel();
    checker().check(CheckKind::Contract,
                    col_count > 0 &&
                        total % static_cast<uint64_t>(col_count) == 0 &&
                        out_1xC.numel() == static_cast<uint64_t>(col_count),
                    "DefaultCpuBackend row_sum", "semantic shape mismatch");
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

void DefaultCpuBackend::gemm(const TensorView &a, const TensorView &b,
                             TensorView &out) {
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

void DefaultCpuBackend::gemm_ranked_matrix_rhs(const TensorView &a,
                                               const TensorView &b,
                                               TensorView &out) {
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
  const int64_t m = a.dim(a.rank() - 2);
  const int64_t k = a.dim(a.rank() - 1);
  const int64_t n = b.dim(1);
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    for (int64_t r = 0; r < m; ++r) {
      for (int64_t c = 0; c < n; ++c) {
        float acc = 0.0f;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, r, kk) *
                 CpuMemOperations::load_f32(b, kk, c);
        }
        CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
      }
    }
  }
}

void DefaultCpuBackend::gemm_ranked_matrix_rhs_t(const TensorView &a,
                                                 const TensorView &b,
                                                 TensorView &out) {
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
  const int64_t m = a.dim(a.rank() - 2);
  const int64_t k = a.dim(a.rank() - 1);
  const int64_t n = b.dim(0);
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    for (int64_t r = 0; r < m; ++r) {
      for (int64_t c = 0; c < n; ++c) {
        float acc = 0.0f;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, r, kk) *
                 CpuMemOperations::load_f32(b, c, kk);
        }
        CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
      }
    }
  }
}

void DefaultCpuBackend::gemm_ranked_reduce_lhs_t(const TensorView &a,
                                                 const TensorView &b,
                                                 TensorView &out) {
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
  const int64_t shared_dim = a.dim(a.rank() - 2);
  const int64_t out_rows = a.dim(a.rank() - 1);
  const int64_t out_cols = b.dim(b.rank() - 1);
  for (int64_t r = 0; r < out_rows; ++r) {
    for (int64_t c = 0; c < out_cols; ++c) {
      float acc = 0.0f;
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        for (int64_t k = 0; k < shared_dim; ++k) {
          acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, k, r) *
                 CpuMemOperations::load_f32_prefix_last2(b, prefix, k, c);
        }
      }
      CpuMemOperations::store_f32(out, r, c, acc);
    }
  }
}

void DefaultCpuBackend::gemm_batched(const TensorView &a, const TensorView &b,
                                     TensorView &out) {
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
                 CpuMemOperations::load_f32_prefix_last2(b, prefix, kk, c);
        }
        CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
      }
    }
  }
}

void DefaultCpuBackend::gemm_batched_lhs_t(const TensorView &a,
                                           const TensorView &b,
                                           TensorView &out) {
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
        CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
      }
    }
  }
}

void DefaultCpuBackend::gemm_batched_rhs_t(const TensorView &a,
                                           const TensorView &b,
                                           TensorView &out) {
  const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(a, 2);
  const int64_t m = a.dim(a.rank() - 2);
  const int64_t k = a.dim(a.rank() - 1);
  const int64_t n = b.dim(b.rank() - 2);
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    for (int64_t r = 0; r < m; ++r) {
      for (int64_t c = 0; c < n; ++c) {
        float acc = 0.0f;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += CpuMemOperations::load_f32_prefix_last2(a, prefix, r, kk) *
                 CpuMemOperations::load_f32_prefix_last2(b, prefix, c, kk);
        }
        CpuMemOperations::store_f32_prefix_last2(out, prefix, r, c, acc);
      }
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
  CpuMemOperations::check_index_dtype(ids, "DefaultCpuBackend::embedding_lookup");
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
                                            TensorView &d_tok) {
  CpuMemOperations::check_index_dtype(ids, "DefaultCpuBackend::accumulate_embedding_grads");
  const uint64_t token_count = ids.numel();
  const int64_t model_dim = d_cur.dim(d_cur.rank() - 1);
  const int64_t vocab_size = d_tok.dim(0);
  for (uint64_t t = 0; t < token_count; ++t) {
    const int64_t idx = CpuMemOperations::load_index_linear(ids, t);
    if (idx < 0 || idx >= vocab_size) {
      throw std::runtime_error(
          "DefaultCpuBackend::accumulate_embedding_grads: token id out of range");
    }
    for (int64_t d = 0; d < model_dim; ++d) {
      const float g = CpuMemOperations::load_f32_prefix_last1(d_cur, t, d);
      CpuMemOperations::store_f32(d_tok, idx, d, CpuMemOperations::load_f32(d_tok, idx, d) + g);
    }
  }
}

void DefaultCpuBackend::cross_entropy_mean(const TensorView &logits,
                                    const TensorView &targets,
                                    TensorView &out_loss) {
  CpuMemOperations::check_index_dtype(targets, "DefaultCpuBackend::cross_entropy_mean");
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
  CpuMemOperations::check_index_dtype(targets, "DefaultCpuBackend::backward_from_logits_targets");
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

void DefaultCpuBackend::scaled_causal_softmax_rows(const TensorView &scores,
                                                   float scale,
                                                   TensorView &out) {
  const uint64_t prefix_count =
      CpuMemOperations::logical_prefix_count(scores, 1);
  const int64_t col_count = scores.dim(scores.rank() - 1);
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    const int64_t local_row = static_cast<int64_t>(prefix % col_count);
    float max_value = -std::numeric_limits<float>::max();
    for (int64_t c = 0; c <= local_row; ++c) {
      max_value = std::max(
          max_value,
          CpuMemOperations::load_f32_prefix_last1(scores, prefix, c) * scale);
    }

    double sum = 0.0;
    for (int64_t c = 0; c < col_count; ++c) {
      float value = 0.0f;
      if (c <= local_row) {
        value = std::exp(
            CpuMemOperations::load_f32_prefix_last1(scores, prefix, c) *
                scale -
            max_value);
        sum += static_cast<double>(value);
      }
      CpuMemOperations::store_f32_prefix_last1(out, prefix, c, value);
    }
    if (sum <= 0.0) {
      throw std::runtime_error(
          "DefaultCpuBackend::scaled_causal_softmax_rows: softmax sum <= 0");
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int64_t c = 0; c <= local_row; ++c) {
      CpuMemOperations::store_f32_prefix_last1(
          out, prefix, c,
          CpuMemOperations::load_f32_prefix_last1(out, prefix, c) * inv_sum);
    }
  }
}

bool DefaultCpuBackend::supports_exec_context_iteration() const {
  return false;
}

void DefaultCpuBackend::start_exec_context_iteration() {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context iteration is not supported");
}

void DefaultCpuBackend::finish_exec_context_iteration() {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context iteration is not supported");
}

void DefaultCpuBackend::start_exec_context_group() {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context group is not supported");
}

void DefaultCpuBackend::finish_exec_context_group() {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context group is not supported");
}

void DefaultCpuBackend::gemm_batched_exec_context(const TensorView &,
                                                  const TensorView &,
                                                  TensorView &) {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context batched gemm is not supported");
}

void DefaultCpuBackend::gemm_batched_rhs_t_exec_context(
    const TensorView &, const TensorView &, TensorView &) {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context rhs-transposed batched gemm is not supported");
}

void DefaultCpuBackend::gemm_batched_lhs_t_exec_context(
    const TensorView &, const TensorView &, TensorView &) {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context lhs-transposed batched gemm is not supported");
}

void DefaultCpuBackend::scaled_causal_softmax_rows_exec_context(
    const TensorView &, float, TensorView &) {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context causal softmax is not supported");
}

void DefaultCpuBackend::softmax_backward_causal_rows_exec_context(
    const TensorView &, const TensorView &, TensorView &) {
  throw std::runtime_error(
      "DefaultCpuBackend: exec context causal softmax backward is not supported");
}

void DefaultCpuBackend::softmax_backward_causal_rows(const TensorView &softmax,
                                                     const TensorView &dout,
                                                     TensorView &dx) {
  const uint64_t prefix_count =
      CpuMemOperations::logical_prefix_count(softmax, 1);
  const int64_t col_count = softmax.dim(softmax.rank() - 1);
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    const int64_t local_row = static_cast<int64_t>(prefix % col_count);
    float dot = 0.0f;
    for (int64_t c = 0; c <= local_row; ++c) {
      dot += CpuMemOperations::load_f32_prefix_last1(softmax, prefix, c) *
             CpuMemOperations::load_f32_prefix_last1(dout, prefix, c);
    }
    for (int64_t c = 0; c < col_count; ++c) {
      float g = 0.0f;
      if (c <= local_row) {
        const float s =
            CpuMemOperations::load_f32_prefix_last1(softmax, prefix, c);
        g = s * (CpuMemOperations::load_f32_prefix_last1(dout, prefix, c) -
                 dot);
      }
      CpuMemOperations::store_f32_prefix_last1(dx, prefix, c, g);
    }
  }
}

void DefaultCpuBackend::apply_causal_mask_inplace(TensorView &scores, float neg_inf) {
  if (scores.rank() >= 2) {
    const uint64_t prefix_count = CpuMemOperations::logical_prefix_count(scores, 2);
    const int64_t token_rows = scores.dim(scores.rank() - 2);
    const int64_t token_cols = scores.dim(scores.rank() - 1);
    checker().check(CheckKind::Contract, token_rows == token_cols,
                    "DefaultCpuBackend apply_causal_mask_inplace",
                    "scores must end with [T,T]");
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

void DefaultCpuBackend::rotary_embedding_inplace(TensorView &x,
                                                 int64_t n_heads, float base,
                                                 bool inverse) {
  const Checker &checks = checker();
  const char *who = "DefaultCpuBackend rotary_embedding_inplace";
  checks.check(CheckKind::Contract, x.rank() >= 2, who,
               "x must be [..., S, D]");
  const int64_t seq_len = x.dim(x.rank() - 2);
  const int64_t model_dim = x.dim(x.rank() - 1);
  checks.check(CheckKind::Contract, n_heads > 0 && model_dim % n_heads == 0,
               who, "D must divide by n_heads");
  const int64_t head_dim = model_dim / n_heads;
  checks.check(CheckKind::Contract, head_dim % 2 == 0, who,
               "head_dim must be even");
  const int64_t half = head_dim / 2;
  std::vector<float> inv_freq(static_cast<size_t>(half));
  for (int64_t i = 0; i < half; ++i) {
    inv_freq[static_cast<size_t>(i)] =
        std::pow(base, -2.0f * static_cast<float>(i) /
                           static_cast<float>(head_dim));
  }
  const float sign = inverse ? -1.0f : 1.0f;
  const uint64_t row_count = CpuMemOperations::logical_prefix_count(x, 1);
  for (uint64_t row = 0; row < row_count; ++row) {
    const float pos = static_cast<float>(
        static_cast<int64_t>(row % static_cast<uint64_t>(seq_len)));
    for (int64_t h = 0; h < n_heads; ++h) {
      for (int64_t i = 0; i < half; ++i) {
        const float angle = pos * inv_freq[static_cast<size_t>(i)];
        const float c = std::cos(angle);
        const float s = sign * std::sin(angle);
        const int64_t c1 = h * head_dim + i;
        const int64_t c2 = c1 + half;
        const float x1 = CpuMemOperations::load_f32_prefix_last1(x, row, c1);
        const float x2 = CpuMemOperations::load_f32_prefix_last1(x, row, c2);
        CpuMemOperations::store_f32_prefix_last1(x, row, c1, x1 * c - x2 * s);
        CpuMemOperations::store_f32_prefix_last1(x, row, c2, x1 * s + x2 * c);
      }
    }
  }
}

void DefaultCpuBackend::adamw_step(TensorView &params, const TensorView &grads,
                            TensorView &m, TensorView &v, uint64_t step,
                            float learning_rate, float beta1, float beta2,
                            float weight_decay, bool apply_weight_decay) {
  const Checker &checks = checker();
  const char *who = "DefaultCpuBackend adamw_step";
  checks.check(CheckKind::Contract, step >= 1, who, "step must be >= 1");
  checks.check_supported(params, Device::CPU, DType::F32, true,
                         "DefaultCpuBackend adamw_step params");
  checks.check_supported(grads, Device::CPU, DType::F32, true,
                         "DefaultCpuBackend adamw_step grads");
  checks.check_supported(m, Device::CPU, DType::F32, true,
                         "DefaultCpuBackend adamw_step m");
  checks.check_supported(v, Device::CPU, DType::F32, true,
                         "DefaultCpuBackend adamw_step v");

  checks.check_same_shape(params, grads, "DefaultCpuBackend adamw_step grads");
  checks.check_same_shape(params, m, "DefaultCpuBackend adamw_step m");
  checks.check_same_shape(params, v, "DefaultCpuBackend adamw_step v");
  checks.check_positive(learning_rate, who, "learning_rate");
  checks.check_range(beta1, 0.0, 1.0, Interval::ClosedOpen, who, "beta1");
  checks.check_range(beta2, 0.0, 1.0, Interval::ClosedOpen, who, "beta2");

  const int64_t n = params.shape().dim(0) * params.shape().dim(1);
  const double t = static_cast<double>(step);
  const float b1_corr =
      1.0f - static_cast<float>(std::pow(static_cast<double>(beta1), t));
  const float b2_corr =
      1.0f - static_cast<float>(std::pow(static_cast<double>(beta2), t));
  checks.check(CheckKind::Numeric, b1_corr > 0.0f && b2_corr > 0.0f, who,
               "invalid bias correction");

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
  checker().check(CheckKind::Layout, dst != nullptr,
                  "DefaultCpuBackend read_file2device", "null destination");

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
