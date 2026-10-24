#pragma once

#include "backend/device_backend.hpp"
#include "tensor.hpp"

#include <litnicelm_utils/bf16.hpp>
#include <litnicelm_utils/checks.hpp>
#include <litnicelm_utils/view_geometry.hpp>

#include <cstring>

namespace host_tensor_stage {

inline void check_row_dense(const TensorView &t, const char *who) {
  const int64_t elem = static_cast<int64_t>(dtype_size(t.dtype()));
  checker().check(CheckKind::Layout, elem > 0, who, "invalid dtype");
  if (t.rank() == 0) {
    return;
  }
  checker().check(CheckKind::Layout, t.stride_bytes(t.rank() - 1) == elem,
                  who, "requires row-dense tensor view");
}

// Rows run over the leading axes; a rank-0 view is a single row.
inline size_t row_suffix_rank(const TensorView &t) { return t.rank() == 0 ? 0 : 1; }

inline uint64_t dense_row_bytes(const TensorView &t, const char *who) {
  const uint64_t elem = dtype_size(t.dtype());
  checker().check(CheckKind::Layout, elem != 0, who, "invalid dtype size");
  if (t.rank() == 0) {
    return elem;
  }
  return static_cast<uint64_t>(t.dim(t.rank() - 1)) * elem;
}

inline void check_copy_compatible(const TensorView &src, const TensorView &dst,
                                  const char *who) {
  checker().check(CheckKind::Contract, src.dtype() == dst.dtype(), who,
                  "dtype mismatch");
  checker().check(CheckKind::Contract, src.numel() == dst.numel(), who,
                  "shape mismatch");
  check_row_dense(src, who);
  check_row_dense(dst, who);
}

// Host-to-host copy for copy_to_cpu / copy_from_cpu, which have already
// checked the operands (compatible, both on the CPU, not empty).
inline void copy_rows_cpu_to_cpu(const TensorView &src, TensorView &dst,
                                 const char *who) {
  if (src.is_contiguous() && dst.is_contiguous()) {
    std::memcpy(dst.data(), src.data(), static_cast<size_t>(src.bytes()));
    return;
  }

  const size_t row_bytes = static_cast<size_t>(dense_row_bytes(src, who));
  const uint64_t prefix_count = view_geometry::prefix_count(src, row_suffix_rank(src));
  auto *dst_base = reinterpret_cast<uint8_t *>(dst.data());
  const auto *src_base = reinterpret_cast<const uint8_t *>(src.data());
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    std::memcpy(dst_base + view_geometry::prefix_byte_offset(dst, prefix, row_suffix_rank(dst)),
                src_base + view_geometry::prefix_byte_offset(src, prefix, row_suffix_rank(src)), row_bytes);
  }
}

inline void copy_to_cpu(DeviceBackend &backend, const TensorView &src,
                        TensorView &dst_cpu, const char *who) {
  check_copy_compatible(src, dst_cpu, who);
  checker().check(CheckKind::Capability, dst_cpu.device() == Device::CPU, who,
                  "destination must be CPU");
  if (src.numel() == 0) {
    return;
  }
  if (src.device() == Device::CPU) {
    copy_rows_cpu_to_cpu(src, dst_cpu, who);
    return;
  }

  const uint64_t row_bytes = dense_row_bytes(src, who);
  const uint64_t prefix_count = view_geometry::prefix_count(src, row_suffix_rank(src));
  auto *dst_base = reinterpret_cast<uint8_t *>(dst_cpu.data());
  const auto *src_base = reinterpret_cast<const uint8_t *>(src.data());
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    backend.copy_device2host(dst_base + view_geometry::prefix_byte_offset(dst_cpu, prefix, row_suffix_rank(dst_cpu)),
                             src_base + view_geometry::prefix_byte_offset(src, prefix, row_suffix_rank(src)),
                             row_bytes);
  }
}

inline void copy_from_cpu(DeviceBackend &backend, const TensorView &src_cpu,
                          TensorView &dst, const char *who) {
  check_copy_compatible(src_cpu, dst, who);
  checker().check(CheckKind::Capability, src_cpu.device() == Device::CPU, who,
                  "source must be CPU");
  if (src_cpu.numel() == 0) {
    return;
  }
  if (dst.device() == Device::CPU) {
    copy_rows_cpu_to_cpu(src_cpu, dst, who);
    return;
  }

  const uint64_t row_bytes = dense_row_bytes(src_cpu, who);
  const uint64_t prefix_count = view_geometry::prefix_count(src_cpu, row_suffix_rank(src_cpu));
  const auto *src_base = reinterpret_cast<const uint8_t *>(src_cpu.data());
  auto *dst_base = reinterpret_cast<uint8_t *>(dst.data());
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    backend.copy_host2device(dst_base + view_geometry::prefix_byte_offset(dst, prefix, row_suffix_rank(dst)),
                             src_base + view_geometry::prefix_byte_offset(src_cpu, prefix, row_suffix_rank(src_cpu)),
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

// A host F32 copy of an F32 or BF16 tensor, for host code that reads the
// values as float: BF16 elements are widened exactly on the way.
inline Tensor stage_to_cpu_f32(DeviceBackend &backend, const TensorView &src,
                               const char *who) {
  checker().check(CheckKind::Capability,
                  src.dtype() == DType::F32 || src.dtype() == DType::BF16, who,
                  "stage_to_cpu_f32 reads F32 or BF16 tensors");
  if (src.dtype() == DType::F32) {
    return stage_to_cpu(backend, src, who);
  }
  const Tensor raw = stage_to_cpu(backend, src, who);
  Tensor widened = Tensor::make_cpu(DType::F32, src.shape());
  const auto *bits = static_cast<const uint16_t *>(raw.view().data());
  auto *values = static_cast<float *>(widened.view().data());
  for (uint64_t i = 0; i < src.numel(); ++i) {
    values[i] = bf16_bits_to_f32(bits[i]);
  }
  return widened;
}

} // namespace host_tensor_stage
