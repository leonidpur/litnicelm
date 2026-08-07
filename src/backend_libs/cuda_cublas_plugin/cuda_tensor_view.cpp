#include "cuda_tensor_view.hpp"

#include <array>
#include <stdexcept>
#include <string>

namespace cuda_cublas_plugin {

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
  validate_backend_tensor_view(abi, "cuda cublas plugin");
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

KernelTensorView to_kernel_tensor_view(const TensorView &view) {
  KernelTensorView out{};
  out.dtype = view.dtype();
  out.data = view.data();
  out.rank = static_cast<int32_t>(view.rank());
  for (size_t i = 0; i < view.rank() && i < kMaxTensorRank; ++i) {
    out.dims[i] = view.dim(i);
    out.strides_bytes[i] = view.stride_bytes(i);
  }
  out.rows = tensor_rows(view);
  out.cols = tensor_cols(view);
  out.stride_r_bytes = view.rank() >= 2 ? view.stride_bytes(view.rank() - 2) : 0;
  out.stride_c_bytes = view.rank() >= 1 ? view.stride_bytes(view.rank() - 1) : 0;
  return out;
}

KernelTensorView to_kernel_index_vector_view(const TensorView &view) {
  KernelTensorView out{};
  out.dtype = view.dtype();
  out.data = view.data();
  out.rank = 2;
  out.dims[0] = static_cast<int64_t>(view.numel());
  out.dims[1] = 1;
  out.rows = out.dims[0];
  out.cols = 1;
  out.stride_r_bytes = view.rank() >= 1 ? view.stride_bytes(view.rank() - 1) : 0;
  out.stride_c_bytes = out.stride_r_bytes;
  out.strides_bytes[0] = out.stride_r_bytes;
  out.strides_bytes[1] = out.stride_c_bytes;
  return out;
}

bool is_power_of_two(uint32_t x) { return x != 0 && (x & (x - 1)) == 0; }

int64_t tensor_cols(const TensorView &view) {
  return view.rank() == 0 ? 0 : view.dim(view.rank() - 1);
}

int64_t tensor_rows(const TensorView &view) {
  if (view.rank() == 0) {
    return 0;
  }
  if (view.rank() == 1) {
    return 1;
  }
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < view.rank(); ++i) {
    rows *= view.dim(i);
  }
  return rows;
}

uint64_t row_bytes(const TensorView &view) {
  return static_cast<uint64_t>(tensor_cols(view)) * dtype_size(view.dtype());
}

uint64_t span_bytes(const TensorView &view) {
  if (tensor_rows(view) <= 0 || tensor_cols(view) <= 0) {
    return 0;
  }
  const uint64_t elem_size = dtype_size(view.dtype());
  return static_cast<uint64_t>(tensor_rows(view) - 1) *
             static_cast<uint64_t>(view.stride_bytes(view.rank() - 2)) +
         static_cast<uint64_t>(tensor_cols(view) - 1) *
             static_cast<uint64_t>(view.stride_bytes(view.rank() - 1)) +
         elem_size;
}

bool has_storage(const TensorView &view) {
  return tensor_rows(view) == 0 || tensor_cols(view) == 0 || view.data() != nullptr;
}

bool is_cuda_row_major(const TensorView &view) {
  const int64_t elem_size = static_cast<int64_t>(dtype_size(view.dtype()));
  return view.device() == Device::GPU && has_storage(view) &&
         view.rank() >= 2 &&
         view.stride_bytes(view.rank() - 1) == elem_size &&
         view.stride_bytes(view.rank() - 2) >= tensor_cols(view) * elem_size;
}

bool is_cuda_f32_row_major(const TensorView &view) {
  return view.dtype() == DType::F32 && is_cuda_row_major(view);
}

bool is_cuda_f32_contiguous_row_major(const TensorView &view) {
  return is_cuda_f32_row_major(view) && view.is_contiguous_row_major();
}

uint64_t logical_prefix_count(const TensorView &view, size_t suffix_rank) {
  if (view.rank() < suffix_rank) {
    throw std::runtime_error(
        "cuda_cublas_plugin: suffix rank exceeds tensor rank");
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
        "cuda_cublas_plugin: suffix rank exceeds tensor rank");
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

float *prefix_matrix_ptr(TensorView &view, uint64_t prefix) {
  auto *base = reinterpret_cast<uint8_t *>(view.data());
  return reinterpret_cast<float *>(
      base + prefix_matrix_byte_offset(view, prefix, 2));
}

const float *prefix_matrix_ptr(const TensorView &view, uint64_t prefix) {
  auto *base = reinterpret_cast<const uint8_t *>(view.data());
  return reinterpret_cast<const float *>(
      base + prefix_matrix_byte_offset(view, prefix, 2));
}

void require_cuda_row_major(const TensorView &view, const char *what) {
  if (!is_cuda_row_major(view)) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " requires a row-major GPU tensor");
  }
}

void require_cuda_f32_row_major(const TensorView &view, const char *what) {
  if (!is_cuda_f32_row_major(view)) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " requires a row-major GPU f32 tensor");
  }
}

int leading_dim_f32(const TensorView &view) {
  return static_cast<int>(view.stride_bytes(view.rank() - 2) / sizeof(float));
}

void copy_tensor_2d(const TensorView &src, const TensorView &dst,
                    cudaMemcpyKind kind, const char *what) {
  if (tensor_rows(src) != tensor_rows(dst) || tensor_cols(src) != tensor_cols(dst) ||
      src.dtype() != dst.dtype()) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " shape or dtype mismatch");
  }
  if (tensor_rows(src) == 0 || tensor_cols(src) == 0) {
    return;
  }
  check_cuda(cudaMemcpy2D(dst.data(), static_cast<size_t>(dst.stride_bytes(dst.rank() - 2)),
                          src.data(), static_cast<size_t>(src.stride_bytes(src.rank() - 2)),
                          static_cast<size_t>(row_bytes(src)),
                          static_cast<size_t>(tensor_rows(src)), kind),
             what);
}

HostTensorStage::HostTensorStage(const TensorView &like)
    : storage(static_cast<size_t>(span_bytes(like))),
      view(Device::CPU, like.dtype(), storage.empty() ? nullptr : storage.data(),
           like.shape(),
           like.rank() >= 1 ? like.stride_bytes(like.rank() - 1) : 0,
           like.rank() >= 2 ? like.stride_bytes(like.rank() - 2) : 0) {}

} // namespace cuda_cublas_plugin
