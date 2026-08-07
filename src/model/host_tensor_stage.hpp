#pragma once

#include "backend/device_backend.hpp"
#include "tensor.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace host_tensor_stage {

inline void require_row_dense(const TensorView &t, const char *who) {
  const int64_t elem = static_cast<int64_t>(dtype_size(t.dtype()));
  if (elem <= 0) {
    throw std::runtime_error(std::string(who) + ": invalid dtype");
  }
  if (t.rank() == 0) {
    return;
  }
  if (t.stride_bytes(t.rank() - 1) != elem) {
    throw std::runtime_error(std::string(who) +
                             ": requires row-dense tensor view");
  }
}

inline uint64_t logical_prefix_count_last1(const TensorView &t) {
  if (t.rank() == 0) {
    return 1;
  }
  uint64_t count = 1;
  for (size_t axis = 0; axis + 1 < t.rank(); ++axis) {
    count *= static_cast<uint64_t>(t.dim(axis));
  }
  return count;
}

inline int64_t prefix_last1_byte_offset(const TensorView &t, uint64_t prefix) {
  if (t.rank() <= 1) {
    return static_cast<int64_t>(prefix) *
           (t.rank() == 0 ? 0 : t.stride_bytes(0));
  }

  int64_t offset = 0;
  for (size_t axis = t.rank() - 1; axis-- > 0;) {
    const uint64_t dim = static_cast<uint64_t>(t.dim(axis));
    const uint64_t idx = prefix % dim;
    prefix /= dim;
    offset += static_cast<int64_t>(idx) * t.stride_bytes(axis);
  }
  return offset;
}

inline uint64_t dense_row_bytes(const TensorView &t, const char *who) {
  const uint64_t elem = dtype_size(t.dtype());
  if (elem == 0) {
    throw std::runtime_error(std::string(who) + ": invalid dtype size");
  }
  if (t.rank() == 0) {
    return elem;
  }
  return static_cast<uint64_t>(t.dim(t.rank() - 1)) * elem;
}

inline void require_copy_compatible(const TensorView &src, const TensorView &dst,
                                    const char *who) {
  if (src.dtype() != dst.dtype()) {
    throw std::runtime_error(std::string(who) + ": dtype mismatch");
  }
  if (src.numel() != dst.numel()) {
    throw std::runtime_error(std::string(who) + ": shape mismatch");
  }
  require_row_dense(src, who);
  require_row_dense(dst, who);
}

inline void copy_tensor_cpu_to_cpu(const TensorView &src, TensorView &dst,
                                   const char *who) {
  require_copy_compatible(src, dst, who);
  if (src.device() != Device::CPU || dst.device() != Device::CPU) {
    throw std::runtime_error(std::string(who) +
                             ": copy_tensor_cpu_to_cpu requires CPU tensors");
  }
  if (src.numel() == 0) {
    return;
  }
  if (src.is_contiguous() && dst.is_contiguous()) {
    std::memcpy(dst.data(), src.data(), static_cast<size_t>(src.bytes()));
    return;
  }

  const size_t row_bytes = static_cast<size_t>(dense_row_bytes(src, who));
  const uint64_t prefix_count = logical_prefix_count_last1(src);
  auto *dst_base = reinterpret_cast<uint8_t *>(dst.data());
  const auto *src_base = reinterpret_cast<const uint8_t *>(src.data());
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    std::memcpy(dst_base + prefix_last1_byte_offset(dst, prefix),
                src_base + prefix_last1_byte_offset(src, prefix), row_bytes);
  }
}

inline void copy_to_cpu(DeviceBackend &backend, const TensorView &src,
                        TensorView &dst_cpu, const char *who) {
  require_copy_compatible(src, dst_cpu, who);
  if (dst_cpu.device() != Device::CPU) {
    throw std::runtime_error(std::string(who) +
                             ": destination must be CPU");
  }
  if (src.numel() == 0) {
    return;
  }
  if (src.device() == Device::CPU) {
    copy_tensor_cpu_to_cpu(src, dst_cpu, who);
    return;
  }

  const uint64_t row_bytes = dense_row_bytes(src, who);
  const uint64_t prefix_count = logical_prefix_count_last1(src);
  auto *dst_base = reinterpret_cast<uint8_t *>(dst_cpu.data());
  const auto *src_base = reinterpret_cast<const uint8_t *>(src.data());
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    backend.copy_device2host(dst_base + prefix_last1_byte_offset(dst_cpu, prefix),
                             src_base + prefix_last1_byte_offset(src, prefix),
                             row_bytes);
  }
}

inline void copy_from_cpu(DeviceBackend &backend, const TensorView &src_cpu,
                          TensorView &dst, const char *who) {
  require_copy_compatible(src_cpu, dst, who);
  if (src_cpu.device() != Device::CPU) {
    throw std::runtime_error(std::string(who) + ": source must be CPU");
  }
  if (src_cpu.numel() == 0) {
    return;
  }
  if (dst.device() == Device::CPU) {
    copy_tensor_cpu_to_cpu(src_cpu, dst, who);
    return;
  }

  const uint64_t row_bytes = dense_row_bytes(src_cpu, who);
  const uint64_t prefix_count = logical_prefix_count_last1(src_cpu);
  const auto *src_base = reinterpret_cast<const uint8_t *>(src_cpu.data());
  auto *dst_base = reinterpret_cast<uint8_t *>(dst.data());
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    backend.copy_host2device(dst_base + prefix_last1_byte_offset(dst, prefix),
                             src_base + prefix_last1_byte_offset(src_cpu, prefix),
                             row_bytes);
  }
}

inline Tensor stage_to_cpu(DeviceBackend &backend, const TensorView &src,
                           const char *who) {
  Tensor staged = Tensor::make_cpu(src.dtype(), src.shape());
  TensorView staged_view = staged.view();
  copy_to_cpu(backend, src, staged_view, who);
  return staged;
}

} // namespace host_tensor_stage
