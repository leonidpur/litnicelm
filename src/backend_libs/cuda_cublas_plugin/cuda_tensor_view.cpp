#include "cuda_tensor_view.hpp"

#include <litnicelm_utils/view_geometry.hpp>
#include <litnicelm_utils/checks.hpp>

#include <array>
#include <string>

namespace cuda_cublas_plugin {

const TensorView &to_tensor_view(const TensorViewStruct &abi) {
  return as_tensor_view(abi, "cuda cublas plugin");
}

namespace {

// The byte distance between consecutive rows, rows running over all leading
// dims, if that distance is the same everywhere: the stride of the innermost
// leading axis longer than 1, with every outer such axis spanning exactly
// the rows inside it. Size-1 axes do not matter. Returns false otherwise.
bool uniform_row_stride(const TensorView &view, int64_t &stride) {
  stride = 0;
  int64_t span = -1;  // bytes covered by the rows of the axes seen so far
  for (size_t axis = view.rank() >= 2 ? view.rank() - 1 : 0; axis-- > 0;) {
    if (view.dim(axis) == 1) {
      continue;
    }
    if (span < 0) {
      stride = view.stride_bytes(axis);
    } else if (view.stride_bytes(axis) != span) {
      return false;
    }
    span = view.stride_bytes(axis) * view.dim(axis);
  }
  return true;
}

// The view's elements lie back to back in row-major order. Size-1 axes may
// have any stride, as when an [S] row of ids is sliced to its first n.
bool dense_ignoring_unit_axes(const TensorView &view) {
  int64_t expected = static_cast<int64_t>(dtype_size(view.dtype()));
  for (size_t axis = view.rank(); axis-- > 0;) {
    if (view.dim(axis) == 1) {
      continue;
    }
    if (view.stride_bytes(axis) != expected) {
      return false;
    }
    expected *= view.dim(axis);
  }
  return true;
}

} // namespace

KernelTensorView to_kernel_tensor_view(const TensorView &view) {
  int64_t row_stride = 0;
  checker().check(CheckKind::Capability, uniform_row_stride(view, row_stride),
                  "cuda_cublas_plugin",
                  "kernels need evenly spaced rows; this view's leading axes "
                  "are not");
  return {view.data(), tensor_rows(view), tensor_cols(view), row_stride};
}

KernelTensorView to_kernel_index_vector_view(const TensorView &view) {
  // Read as one column of numel ids, one per row.
  checker().check(CheckKind::Capability, dense_ignoring_unit_axes(view),
                  "cuda_cublas_plugin", "id tensors must be contiguous");
  return {view.data(), static_cast<int64_t>(view.numel()), 1,
          static_cast<int64_t>(dtype_size(view.dtype()))};
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

void *prefix_matrix_ptr(TensorView &view, uint64_t prefix) {
  return static_cast<uint8_t *>(view.data()) +
         view_geometry::prefix_byte_offset(view, prefix, 2);
}

const void *prefix_matrix_ptr(const TensorView &view, uint64_t prefix) {
  return static_cast<const uint8_t *>(view.data()) +
         view_geometry::prefix_byte_offset(view, prefix, 2);
}

PrecisionVariant precision_variant_of(const TensorView &activation) {
  return activation.dtype() == DType::BF16 ? PrecisionVariant::Bf16Act : PrecisionVariant::F32;
}

DType activation_dtype(PrecisionVariant v) {
  return v == PrecisionVariant::Bf16Act ? DType::BF16 : DType::F32;
}

void check_cuda_row_major(const TensorView &view, const char *what) {
  checker().check(CheckKind::Capability, is_cuda_row_major(view),
                  "cuda_cublas_plugin", [&] {
                    return std::string(what) +
                           " requires a row-major GPU tensor";
                  });
}

void check_cuda_row_major(const TensorView &view,
                          std::initializer_list<DType> dtypes,
                          const char *what) {
  check_cuda_row_major(view, what);
  bool supported = false;
  for (DType dtype : dtypes) {
    supported = supported || view.dtype() == dtype;
  }
  checker().check(CheckKind::Capability, supported, "cuda_cublas_plugin", [&] {
    std::string allowed;
    for (DType dtype : dtypes) {
      allowed += (allowed.empty() ? "" : ", ") + std::string(dtype_name(dtype));
    }
    return std::string(what) + " requires dtype " + allowed + ", got " +
           dtype_name(view.dtype());
  });
}

int leading_dim(const TensorView &view) {
  return static_cast<int>(view.stride_bytes(view.rank() - 2) /
                          static_cast<int64_t>(dtype_size(view.dtype())));
}

void copy_tensor_2d(const TensorView &src, const TensorView &dst,
                    cudaMemcpyKind kind, const char *what) {
  checker().check(CheckKind::Contract,
                  tensor_rows(src) == tensor_rows(dst) &&
                      tensor_cols(src) == tensor_cols(dst) &&
                      src.dtype() == dst.dtype(),
                  "cuda_cublas_plugin", [&] {
                    return std::string(what) + " shape or dtype mismatch";
                  });
  if (tensor_rows(src) == 0 || tensor_cols(src) == 0) {
    return;
  }
  check_cuda(cudaMemcpy2D(dst.data(), static_cast<size_t>(dst.stride_bytes(dst.rank() - 2)),
                          src.data(), static_cast<size_t>(src.stride_bytes(src.rank() - 2)),
                          static_cast<size_t>(row_bytes(src)),
                          static_cast<size_t>(tensor_rows(src)), kind),
             what);
}

} // namespace cuda_cublas_plugin
