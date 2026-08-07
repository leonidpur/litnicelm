#include "../../model/backend/backend_plugin_api.hpp"
#include "../../model/backend/device_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kTileDim = 16;
constexpr float kLayerNormEps = 1e-5f;

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

struct KernelTensorView {
  DType dtype;
  void *data;
  int32_t rank;
  int64_t dims[kMaxTensorRank];
  int64_t strides_bytes[kMaxTensorRank];
  int64_t rows;
  int64_t cols;
  int64_t stride_r_bytes;
  int64_t stride_c_bytes;
};

int64_t tensor_cols(const TensorView &view);
int64_t tensor_rows(const TensorView &view);

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

const char *cublas_status_string(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  default:
    return "CUBLAS_STATUS_UNKNOWN";
  }
}

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " failed: " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char *what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " failed: " + cublas_status_string(status));
  }
}

void check_kernel_launch(const char *what) {
  check_cuda(cudaPeekAtLastError(), what);
}

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

struct HostTensorStage {
  std::vector<uint8_t> storage;
  TensorView view;

  explicit HostTensorStage(const TensorView &like)
      : storage(static_cast<size_t>(span_bytes(like))),
        view(Device::CPU, like.dtype(),
             storage.empty() ? nullptr : storage.data(), like.shape(),
             like.rank() >= 1 ? like.stride_bytes(like.rank() - 1) : 0,
             like.rank() >= 2 ? like.stride_bytes(like.rank() - 2) : 0) {}
};

__device__ inline int64_t kernel_prefix_offset_bytes(const KernelTensorView &t,
                                                     int64_t prefix,
                                                     int32_t prefix_rank) {
  int64_t offset = 0;
  for (int32_t axis = prefix_rank - 1; axis >= 0; --axis) {
    const int64_t dim = t.dims[axis];
    const int64_t coord = dim > 0 ? prefix % dim : 0;
    prefix = dim > 0 ? prefix / dim : 0;
    offset += coord * t.strides_bytes[axis];
  }
  return offset;
}

__device__ inline const char *tensor_ptr(const KernelTensorView &t, int64_t r,
                                         int64_t c) {
  if (t.rank >= 2) {
    const int32_t last_axis = t.rank - 1;
    const int64_t prefix_offset =
        kernel_prefix_offset_bytes(t, r, last_axis);
    return reinterpret_cast<const char *>(t.data) + prefix_offset +
           c * t.strides_bytes[last_axis];
  }
  return reinterpret_cast<const char *>(t.data) + r * t.stride_r_bytes +
         c * t.stride_c_bytes;
}

__device__ inline char *tensor_ptr_mut(const KernelTensorView &t, int64_t r,
                                       int64_t c) {
  if (t.rank >= 2) {
    const int32_t last_axis = t.rank - 1;
    const int64_t prefix_offset =
        kernel_prefix_offset_bytes(t, r, last_axis);
    return reinterpret_cast<char *>(t.data) + prefix_offset +
           c * t.strides_bytes[last_axis];
  }
  return reinterpret_cast<char *>(t.data) + r * t.stride_r_bytes +
         c * t.stride_c_bytes;
}

__device__ inline float load_f32(const KernelTensorView &t, int64_t r,
                                 int64_t c) {
  return *reinterpret_cast<const float *>(tensor_ptr(t, r, c));
}

__device__ inline void store_f32(const KernelTensorView &t, int64_t r, int64_t c,
                                 float value) {
  *reinterpret_cast<float *>(tensor_ptr_mut(t, r, c)) = value;
}

__device__ inline int32_t load_i32(const KernelTensorView &t, int64_t r,
                                   int64_t c) {
  return *reinterpret_cast<const int32_t *>(tensor_ptr(t, r, c));
}

__device__ inline int64_t load_index(const KernelTensorView &t, int64_t r,
                                     int64_t c) {
  if (t.dtype == DType::I32) {
    return static_cast<int64_t>(load_i32(t, r, c));
  }
  return static_cast<int64_t>(load_f32(t, r, c));
}

__global__ void fill_kernel(KernelTensorView t, float value) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = t.rows * t.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / t.cols;
  const int64_t c = idx % t.cols;
  store_f32(t, r, c, value);
}

__global__ void add_kernel(KernelTensorView a, KernelTensorView b,
                           KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  store_f32(out, r, c, load_f32(a, r, c) + load_f32(b, r, c));
}

__global__ void add_batch_seq_plus_pos_kernel(KernelTensorView a,
                                              KernelTensorView b,
                                              KernelTensorView out,
                                              int64_t seq_len) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const int64_t pos = seq_len > 0 ? r % seq_len : r;
  store_f32(out, r, c, load_f32(a, r, c) + load_f32(b, pos, c));
}

__global__ void add_inplace_kernel(KernelTensorView a, KernelTensorView b) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = a.rows * a.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / a.cols;
  const int64_t c = idx % a.cols;
  store_f32(a, r, c, load_f32(a, r, c) + load_f32(b, r, c));
}

__global__ void add_bias_rowwise_kernel(KernelTensorView x, KernelTensorView bias,
                                        KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  store_f32(out, r, c, load_f32(x, r, c) + load_f32(bias, 0, c));
}

__global__ void mul_scalar_kernel(KernelTensorView x, float scale,
                                  KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  store_f32(out, r, c, load_f32(x, r, c) * scale);
}

__global__ void sum_squares_f32_kernel(KernelTensorView x, float *out_sum_sq) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = x.rows * x.cols;

  float local_sum = 0.0f;
  for (int64_t i = idx; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / x.cols;
    const int64_t c = i % x.cols;
    const float v = load_f32(x, r, c);
    local_sum += v * v;
  }

  extern __shared__ float scratch[];
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(out_sum_sq, scratch[0]);
  }
}

__global__ void relu_kernel(KernelTensorView x, KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const float value = load_f32(x, r, c);
  store_f32(out, r, c, value > 0.0f ? value : 0.0f);
}

__global__ void relu_backward_kernel(KernelTensorView preact,
                                     KernelTensorView dout,
                                     KernelTensorView dx) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = dx.rows * dx.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / dx.cols;
  const int64_t c = idx % dx.cols;
  const float grad = load_f32(preact, r, c) > 0.0f ? load_f32(dout, r, c) : 0.0f;
  store_f32(dx, r, c, grad);
}

__global__ void row_sum_kernel(KernelTensorView x, KernelTensorView out_1xC) {
  const int64_t col = blockIdx.x;
  if (col >= x.cols) {
    return;
  }

  extern __shared__ float row_sum_scratch[];
  float local_sum = 0.0f;
  for (int64_t row = threadIdx.x; row < x.rows; row += blockDim.x) {
    local_sum += load_f32(x, row, col);
  }
  row_sum_scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      row_sum_scratch[threadIdx.x] += row_sum_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    store_f32(out_1xC, 0, col, row_sum_scratch[0]);
  }
}

__global__ void transpose_kernel(KernelTensorView x, KernelTensorView out) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= x.rows || c >= x.cols) {
    return;
  }
  store_f32(out, c, r, load_f32(x, r, c));
}

__global__ void matmul_right_transposed_kernel(KernelTensorView a,
                                               KernelTensorView b,
                                               KernelTensorView out) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= out.rows || c >= out.cols) {
    return;
  }
  float sum = 0.0f;
  for (int64_t k = 0; k < a.cols; ++k) {
    sum += load_f32(a, r, k) * load_f32(b, c, k);
  }
  store_f32(out, r, c, sum);
}

__global__ void embedding_lookup_kernel(KernelTensorView table,
                                        KernelTensorView ids,
                                        KernelTensorView out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = out.rows * out.cols;
  if (idx >= total) {
    return;
  }
  const int64_t r = idx / out.cols;
  const int64_t c = idx % out.cols;
  const int64_t token = load_index(ids, r, 0);
  if (token < 0 || token >= table.rows) {
    store_f32(out, r, c, 0.0f);
    return;
  }
  store_f32(out, r, c, load_f32(table, token, c));
}

__global__ void accumulate_embedding_grads_kernel(KernelTensorView ids,
                                                  KernelTensorView d_cur,
                                                  KernelTensorView d_tok,
                                                  KernelTensorView d_pos,
                                                  int64_t seq_len) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = d_cur.rows * d_cur.cols;
  if (idx >= total) {
    return;
  }

  const int64_t t = idx / d_cur.cols;
  const int64_t d = idx % d_cur.cols;
  const int64_t token = load_index(ids, t, 0);
  if (token < 0 || token >= d_tok.rows) {
    return;
  }

  const float g = load_f32(d_cur, t, d);
  const int64_t seq_pos = seq_len > 0 ? t % seq_len : t;
  atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(d_tok, token, d)), g);
  atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(d_pos, seq_pos, d)), g);
}

__global__ void apply_causal_mask_inplace_kernel(KernelTensorView scores,
                                                 float neg_inf) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t r = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= scores.rows || c >= scores.cols) {
    return;
  }
  const int64_t local_r = r % scores.cols;
  if (c > local_r) {
    store_f32(scores, r, c, neg_inf);
  }
}

__global__ void softmax_rows_kernel(KernelTensorView x, KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }

  extern __shared__ float scratch[];

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_max = fmaxf(local_max, load_f32(x, row, c));
  }
  scratch[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = scratch[0];

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float exponent = expf(load_f32(x, row, c) - row_max);
    store_f32(out, row, c, exponent);
    local_sum += exponent;
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / scratch[0];

  for (int64_t c = threadIdx.x; c < out.cols; c += blockDim.x) {
    store_f32(out, row, c, load_f32(out, row, c) * inv_sum);
  }
}

__global__ void softmax_backward_rows_kernel(KernelTensorView softmax,
                                             KernelTensorView dout,
                                             KernelTensorView dx) {
  const int64_t row = blockIdx.x;
  if (row >= softmax.rows) {
    return;
  }

  extern __shared__ float softmax_bw_scratch[];

  float local_dot = 0.0f;
  for (int64_t c = threadIdx.x; c < softmax.cols; c += blockDim.x) {
    local_dot += load_f32(softmax, row, c) * load_f32(dout, row, c);
  }
  softmax_bw_scratch[threadIdx.x] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      softmax_bw_scratch[threadIdx.x] += softmax_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float dot = softmax_bw_scratch[0];

  for (int64_t c = threadIdx.x; c < dx.cols; c += blockDim.x) {
    const float s = load_f32(softmax, row, c);
    const float g = s * (load_f32(dout, row, c) - dot);
    store_f32(dx, row, c, g);
  }
}

__global__ void cross_entropy_mean_backward_inplace_kernel(
    KernelTensorView logits, KernelTensorView targets, KernelTensorView out_loss) {
  const int64_t row = blockIdx.x;
  if (row >= logits.rows) {
    return;
  }

  extern __shared__ float scratch[];

  __shared__ int64_t target;
  if (threadIdx.x == 0) {
    target = load_index(targets, row, 0);
  }
  __syncthreads();
  if (target < 0 || target >= logits.cols) {
    for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
      store_f32(logits, row, c, NAN);
    }
    if (threadIdx.x == 0) {
      atomicAdd(reinterpret_cast<float *>(out_loss.data), NAN);
    }
    return;
  }

  float local_max = -FLT_MAX;
  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    const float v = load_f32(logits, row, c);
    local_max = fmaxf(local_max, v);
  }
  scratch[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = scratch[0];
  if (!isfinite(row_max)) {
    for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
      store_f32(logits, row, c, NAN);
    }
    if (threadIdx.x == 0) {
      atomicAdd(reinterpret_cast<float *>(out_loss.data), NAN);
    }
    return;
  }

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    local_sum += expf(load_f32(logits, row, c) - row_max);
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float sum = scratch[0];
  if (!(sum > 0.0f) || !isfinite(sum)) {
    for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
      store_f32(logits, row, c, NAN);
    }
    if (threadIdx.x == 0) {
      atomicAdd(reinterpret_cast<float *>(out_loss.data), NAN);
    }
    return;
  }
  const float inv_sum = 1.0f / sum;
  const float inv_token_rows = 1.0f / static_cast<float>(logits.rows);

  if (threadIdx.x == 0) {
    const float target_logit = load_f32(logits, row, target);
    const float row_loss =
        (logf(sum) + row_max - target_logit) * inv_token_rows;
    atomicAdd(reinterpret_cast<float *>(out_loss.data), row_loss);
  }

  for (int64_t c = threadIdx.x; c < logits.cols; c += blockDim.x) {
    const float p = expf(load_f32(logits, row, c) - row_max) * inv_sum;
    float gradient = p;
    if (c == target) {
      gradient -= 1.0f;
    }
    store_f32(logits, row, c, gradient * inv_token_rows);
  }
}

__global__ void adamw_step_kernel(KernelTensorView params, KernelTensorView grads,
                                  KernelTensorView m, KernelTensorView v,
                                  float learning_rate, float beta1, float beta2,
                                  float weight_decay, float inv_b1_corr,
                                  float inv_b2_corr, uint32_t apply_weight_decay) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = params.rows * params.cols;
  if (idx >= total) {
    return;
  }

  const int64_t r = idx / params.cols;
  const int64_t c = idx % params.cols;
  constexpr float kEps = 1e-8f;

  const float grad = load_f32(grads, r, c);
  const float old_m = load_f32(m, r, c);
  const float old_v = load_f32(v, r, c);
  const float new_m = beta1 * old_m + (1.0f - beta1) * grad;
  const float new_v = beta2 * old_v + (1.0f - beta2) * grad * grad;
  const float mhat = new_m * inv_b1_corr;
  const float vhat = new_v * inv_b2_corr;
  const float param = load_f32(params, r, c);
  const float decay = apply_weight_decay != 0 ? (weight_decay * param) : 0.0f;
  const float adam = mhat / (sqrtf(vhat) + kEps);

  store_f32(m, r, c, new_m);
  store_f32(v, r, c, new_v);
  store_f32(params, r, c, param - learning_rate * (adam + decay));
}

__global__ void layernorm_backward_kernel(KernelTensorView x,
                                          KernelTensorView gamma,
                                          KernelTensorView dout,
                                          KernelTensorView dx,
                                          KernelTensorView dgamma,
                                          KernelTensorView dbeta) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }

  extern __shared__ double ln_bw_scratch[];

  double local_sum = 0.0;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += static_cast<double>(load_f32(x, row, c));
  }
  ln_bw_scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double mean = ln_bw_scratch[0] / static_cast<double>(x.cols);

  double local_var = 0.0;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const double delta = static_cast<double>(load_f32(x, row, c)) - mean;
    local_var += delta * delta;
  }
  ln_bw_scratch[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  constexpr double kEps = 1e-5;
  const double inv_std =
      rsqrt(static_cast<double>(ln_bw_scratch[0] / static_cast<double>(x.cols)) + kEps);

  double local_sum_dxhat = 0.0;
  double local_sum_dxhat_xhat = 0.0;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const double xhat =
        (static_cast<double>(load_f32(x, row, c)) - mean) * inv_std;
    const double g = static_cast<double>(load_f32(gamma, 0, c));
    const double dyi = static_cast<double>(load_f32(dout, row, c));
    const double dxhat = dyi * g;
    local_sum_dxhat += dxhat;
    local_sum_dxhat_xhat += dxhat * xhat;
    atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(dgamma, 0, c)),
              static_cast<float>(dyi * xhat));
    atomicAdd(reinterpret_cast<float *>(tensor_ptr_mut(dbeta, 0, c)),
              static_cast<float>(dyi));
  }
  ln_bw_scratch[threadIdx.x] = local_sum_dxhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double sum_dxhat = ln_bw_scratch[0];

  ln_bw_scratch[threadIdx.x] = local_sum_dxhat_xhat;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ln_bw_scratch[threadIdx.x] += ln_bw_scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double sum_dxhat_xhat = ln_bw_scratch[0];
  const double n = static_cast<double>(x.cols);

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const double xhat =
        (static_cast<double>(load_f32(x, row, c)) - mean) * inv_std;
    const double g = static_cast<double>(load_f32(gamma, 0, c));
    const double dyi = static_cast<double>(load_f32(dout, row, c));
    const double dxhat = dyi * g;
    const double dxi =
        (inv_std / n) * (n * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
    store_f32(dx, row, c, static_cast<float>(dxi));
  }
}

__global__ void layernorm_forward_kernel(KernelTensorView x,
                                         KernelTensorView gamma,
                                         KernelTensorView beta,
                                         KernelTensorView out) {
  const int64_t row = blockIdx.x;
  if (row >= x.rows) {
    return;
  }

  extern __shared__ float scratch[];

  float local_sum = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    local_sum += load_f32(x, row, c);
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(x.cols);

  float local_var = 0.0f;
  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float delta = load_f32(x, row, c) - mean;
    local_var += delta * delta;
  }
  scratch[threadIdx.x] = local_var;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_std =
      rsqrtf(scratch[0] / static_cast<float>(x.cols) + kLayerNormEps);

  for (int64_t c = threadIdx.x; c < x.cols; c += blockDim.x) {
    const float normalized = (load_f32(x, row, c) - mean) * inv_std;
    const float value =
        normalized * load_f32(gamma, 0, c) + load_f32(beta, 0, c);
    store_f32(out, row, c, value);
  }
}

class CuBlasPluginBackend final : public DeviceBackend {
public:
  CuBlasPluginBackend() {
    check_cublas(cublasCreate(&handle_), "cublasCreate");
    const cublasStatus_t math_status =
        cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    if (math_status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      check_cublas(math_status, "cublasSetMathMode");
    }
  }

  ~CuBlasPluginBackend() override {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  Device device() const override { return Device::GPU; }

  void *alloc(uint64_t bytes, uint32_t alignment) override {
    if (bytes == 0) {
      return nullptr;
    }
    if (!is_power_of_two(alignment)) {
      throw std::invalid_argument("cuda_cublas_plugin: invalid allocation alignment");
    }
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, static_cast<size_t>(bytes)), "cudaMalloc");
    return ptr;
  }

  void free(void *ptr) override {
    if (ptr == nullptr) {
      return;
    }
    check_cuda(cudaFree(ptr), "cudaFree");
  }

  DeviceMemoryInfo memory_info() const override {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      return {};
    }
    return DeviceMemoryInfo{true, static_cast<uint64_t>(free_bytes),
                            static_cast<uint64_t>(total_bytes)};
  }

  void copy_host2device(void *dst, const void *src, uint64_t bytes) override {
    if (bytes == 0) {
      return;
    }
    if (dst == nullptr || src == nullptr) {
      throw std::invalid_argument("cuda_cublas_plugin: copy_host2device null pointer");
    }
    check_cuda(cudaMemcpy(dst, src, static_cast<size_t>(bytes),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(host_to_device)");
  }

  void copy_device2host(void *dst, const void *src, uint64_t bytes) override {
    if (bytes == 0) {
      return;
    }
    if (dst == nullptr || src == nullptr) {
      throw std::invalid_argument("cuda_cublas_plugin: copy_device2host null pointer");
    }
    check_cuda(cudaMemcpy(dst, src, static_cast<size_t>(bytes),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(device_to_host)");
  }

  void copy(const TensorView &src, TensorView &dst) override {
    require_cuda_row_major(src, "copy(src)");
    require_cuda_row_major(dst, "copy(dst)");
    copy_tensor_2d(src, dst, cudaMemcpyDeviceToDevice, "cudaMemcpy2D(copy)");
  }

  void fill(TensorView &t, float v) override {
    require_cuda_f32_row_major(t, "fill");
    if (tensor_rows(t) == 0 || tensor_cols(t) == 0) {
      return;
    }
    const int64_t total = tensor_rows(t) * tensor_cols(t);
    fill_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                            kThreadsPerBlock),
                  kThreadsPerBlock>>>(to_kernel_tensor_view(t), v);
    check_kernel_launch("fill_kernel");
  }

  void add(const TensorView &a, const TensorView &b, TensorView &out) override {
    require_cuda_f32_row_major(a, "add(a)");
    require_cuda_f32_row_major(b, "add(b)");
    require_cuda_f32_row_major(out, "add(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    const int64_t total = tensor_rows(out) * tensor_cols(out);
    if (a.rank() == 3 && b.rank() == 2 && out.rank() == 3 &&
        a.dim(1) == b.dim(0) && a.dim(2) == b.dim(1) &&
        out.dim(0) == a.dim(0) && out.dim(1) == a.dim(1) &&
        out.dim(2) == a.dim(2)) {
      add_batch_seq_plus_pos_kernel<<<
          static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                    kThreadsPerBlock),
          kThreadsPerBlock>>>(to_kernel_tensor_view(a),
                              to_kernel_tensor_view(b),
                              to_kernel_tensor_view(out), a.dim(1));
      check_kernel_launch("add_batch_seq_plus_pos_kernel");
      return;
    }
    add_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                           kThreadsPerBlock),
                 kThreadsPerBlock>>>(to_kernel_tensor_view(a),
                                     to_kernel_tensor_view(b),
                                     to_kernel_tensor_view(out));
    check_kernel_launch("add_kernel");
  }

  void add_inplace(TensorView &a, const TensorView &b) override {
    require_cuda_f32_row_major(a, "add_inplace(a)");
    require_cuda_f32_row_major(b, "add_inplace(b)");
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0) {
      return;
    }
    const int64_t total = tensor_rows(a) * tensor_cols(a);
    add_inplace_kernel<<<
        static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock),
        kThreadsPerBlock>>>(to_kernel_tensor_view(a), to_kernel_tensor_view(b));
    check_kernel_launch("add_inplace_kernel");
  }

  void add_bias_rowwise(const TensorView &x, const TensorView &bias_1xC,
                        TensorView &out) override {
    require_cuda_f32_row_major(x, "add_bias_rowwise(x)");
    require_cuda_f32_row_major(bias_1xC, "add_bias_rowwise(bias)");
    require_cuda_f32_row_major(out, "add_bias_rowwise(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    const int64_t total = tensor_rows(out) * tensor_cols(out);
    add_bias_rowwise_kernel<<<
        static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock),
        kThreadsPerBlock>>>(to_kernel_tensor_view(x),
                            to_kernel_tensor_view(bias_1xC),
                            to_kernel_tensor_view(out));
    check_kernel_launch("add_bias_rowwise_kernel");
  }

  void mul_scalar(const TensorView &x, float s, TensorView &out) override {
    require_cuda_f32_row_major(x, "mul_scalar(x)");
    require_cuda_f32_row_major(out, "mul_scalar(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    const int64_t total = tensor_rows(out) * tensor_cols(out);
    mul_scalar_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                                  kThreadsPerBlock),
                        kThreadsPerBlock>>>(to_kernel_tensor_view(x), s,
                                            to_kernel_tensor_view(out));
    check_kernel_launch("mul_scalar_kernel");
  }

  float sum_squares_f32(const TensorView &x) override {
    require_cuda_f32_row_major(x, "sum_squares_f32(x)");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return 0.0f;
    }
    float *device_sum_sq = nullptr;
    check_cuda(cudaMalloc(&device_sum_sq, sizeof(float)),
               "cudaMalloc(sum_squares_f32)");
    check_cuda(cudaMemset(device_sum_sq, 0, sizeof(float)),
               "cudaMemset(sum_squares_f32)");
    const int64_t total = tensor_rows(x) * tensor_cols(x);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<int64_t>((total + kThreadsPerBlock - 1) / kThreadsPerBlock,
                          1024));
    sum_squares_f32_kernel<<<blocks, kThreadsPerBlock,
                             static_cast<size_t>(kThreadsPerBlock *
                                                 sizeof(float))>>>(
        to_kernel_tensor_view(x), device_sum_sq);
    check_kernel_launch("sum_squares_f32_kernel");
    float host_sum_sq = 0.0f;
    copy_device2host(&host_sum_sq, device_sum_sq, sizeof(float));
    check_cuda(cudaFree(device_sum_sq), "cudaFree(sum_squares_f32)");
    return host_sum_sq;
  }

  void relu(const TensorView &x, TensorView &out) override {
    require_cuda_f32_row_major(x, "relu(x)");
    require_cuda_f32_row_major(out, "relu(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    const int64_t total = tensor_rows(out) * tensor_cols(out);
    relu_kernel<<<static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                            kThreadsPerBlock),
                  kThreadsPerBlock>>>(to_kernel_tensor_view(x),
                                      to_kernel_tensor_view(out));
    check_kernel_launch("relu_kernel");
  }

  void relu_backward(const TensorView &preact, const TensorView &dout,
                     TensorView &dx) override {
    require_cuda_f32_row_major(preact, "relu_backward(preact)");
    require_cuda_f32_row_major(dout, "relu_backward(dout)");
    require_cuda_f32_row_major(dx, "relu_backward(dx)");
    if (tensor_rows(dx) == 0 || tensor_cols(dx) == 0) {
      return;
    }
    const int64_t total = tensor_rows(dx) * tensor_cols(dx);
    relu_backward_kernel<<<
        static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock),
        kThreadsPerBlock>>>(to_kernel_tensor_view(preact),
                            to_kernel_tensor_view(dout),
                            to_kernel_tensor_view(dx));
    check_kernel_launch("relu_backward_kernel");
  }

  void row_sum(const TensorView &x, TensorView &out_1xC) override {
    require_cuda_f32_row_major(x, "row_sum(x)");
    require_cuda_f32_row_major(out_1xC, "row_sum(out)");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return;
    }
    row_sum_kernel<<<static_cast<unsigned int>(tensor_cols(x)), kThreadsPerBlock,
                     static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(out_1xC));
    check_kernel_launch("row_sum_kernel");
  }

  void matmul(const TensorView &a, const TensorView &b, TensorView &out) override {
    require_cuda_f32_row_major(a, "matmul(a)");
    require_cuda_f32_row_major(b, "matmul(b)");
    require_cuda_f32_row_major(out, "matmul(out)");
    if (a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank() &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const int m = static_cast<int>(logical_prefix_count(a, 2) *
                                     static_cast<uint64_t>(a.dim(a.rank() - 2)));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(1));
      const float alpha = 1.0f;
      const float beta = 0.0f;
      check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                               &alpha, b.f32(), n, a.f32(), k, &beta,
                               out.f32(), n),
                   "cublasSgemm(matmul ranked x matrix)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank() &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const int batch_count = static_cast<int>(logical_prefix_count(a, 2));
      const int m = static_cast<int>(a.dim(a.rank() - 2));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const long long int stride_a = static_cast<long long>(m) * k;
      const long long int stride_b = static_cast<long long>(k) * n;
      const long long int stride_out = static_cast<long long>(m) * n;
      const float alpha = 1.0f;
      const float beta = 0.0f;
      check_cublas(cublasSgemmStridedBatched(
                       handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                       b.f32(), n, stride_b, a.f32(), k, stride_a, &beta,
                       out.f32(), n, stride_out, batch_count),
                   "cublasSgemmStridedBatched(matmul ranked)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank()) {
      if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
        return;
      }
      const uint64_t prefix_count = logical_prefix_count(a, 2);
      const int m = static_cast<int>(a.dim(a.rank() - 2));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const int lda = leading_dim_f32(a);
      const int ldb = leading_dim_f32(b);
      const int ldc = leading_dim_f32(out);
      const float alpha = 1.0f;
      const float beta = 0.0f;
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                 &alpha, prefix_matrix_ptr(b, prefix), ldb,
                                 prefix_matrix_ptr(a, prefix), lda, &beta,
                                 prefix_matrix_ptr(out, prefix), ldc),
                     "cublasSgemm(matmul ranked non-contiguous)");
      }
      return;
    }
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
      return;
    }
    const int m = static_cast<int>(tensor_rows(a));
    const int k = static_cast<int>(tensor_cols(a));
    const int n = static_cast<int>(tensor_cols(b));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A * B is the same memory layout as column-major C^T = B^T * A^T.
    check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             b.f32(), leading_dim_f32(b), a.f32(),
                             leading_dim_f32(a), &beta, out.f32(),
                             leading_dim_f32(out)),
                 "cublasSgemm(matmul)");
  }

  void matmul_left_transposed(const TensorView &a, const TensorView &b,
                              TensorView &out) override {
    require_cuda_f32_row_major(a, "matmul_left_transposed(a)");
    require_cuda_f32_row_major(b, "matmul_left_transposed(b)");
    require_cuda_f32_row_major(out, "matmul_left_transposed(out)");
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == 2 &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const int m = static_cast<int>(a.dim(a.rank() - 1));
      const int k = static_cast<int>(logical_prefix_count(a, 2) *
                                     static_cast<uint64_t>(a.dim(a.rank() - 2)));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const float alpha = 1.0f;
      const float beta = 0.0f;
      check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                               &alpha, b.f32(), n, a.f32(), m, &beta,
                               out.f32(), n),
                   "cublasSgemm(matmul_left_transposed ranked reduce)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank() &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const int batch_count = static_cast<int>(logical_prefix_count(a, 2));
      const int m = static_cast<int>(a.dim(a.rank() - 1));
      const int k = static_cast<int>(a.dim(a.rank() - 2));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const long long int stride_a = static_cast<long long>(k) * m;
      const long long int stride_b = static_cast<long long>(k) * n;
      const long long int stride_out = static_cast<long long>(m) * n;
      const float alpha = 1.0f;
      const float beta = 0.0f;
      check_cublas(cublasSgemmStridedBatched(
                       handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha,
                       b.f32(), n, stride_b, a.f32(), m, stride_a, &beta,
                       out.f32(), n, stride_out, batch_count),
                   "cublasSgemmStridedBatched(matmul_left_transposed ranked)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank()) {
      if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
        return;
      }
      const uint64_t prefix_count = logical_prefix_count(a, 2);
      const int m = static_cast<int>(a.dim(a.rank() - 1));
      const int k = static_cast<int>(a.dim(a.rank() - 2));
      const int n = static_cast<int>(b.dim(b.rank() - 1));
      const int lda = leading_dim_f32(a);
      const int ldb = leading_dim_f32(b);
      const int ldc = leading_dim_f32(out);
      const float alpha = 1.0f;
      const float beta = 0.0f;
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                                 &alpha, prefix_matrix_ptr(b, prefix), ldb,
                                 prefix_matrix_ptr(a, prefix), lda, &beta,
                                 prefix_matrix_ptr(out, prefix), ldc),
                     "cublasSgemm(matmul_left_transposed ranked non-contiguous)");
      }
      return;
    }
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
      return;
    }
    const int m = static_cast<int>(tensor_cols(a));
    const int k = static_cast<int>(tensor_rows(a));
    const int n = static_cast<int>(tensor_cols(b));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha,
                             b.f32(), leading_dim_f32(b), a.f32(),
                             leading_dim_f32(a), &beta, out.f32(),
                             leading_dim_f32(out)),
                 "cublasSgemm(matmul_left_transposed)");
  }

  void matmul_right_transposed(const TensorView &a, const TensorView &b,
                               TensorView &out) override {
    require_cuda_f32_row_major(a, "matmul_right_transposed(a)");
    require_cuda_f32_row_major(b, "matmul_right_transposed(b)");
    require_cuda_f32_row_major(out, "matmul_right_transposed(out)");
    if (a.rank() >= 3 && b.rank() == 2 && out.rank() == 2 &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const dim3 block(kTileDim, kTileDim);
      const dim3 grid(
          static_cast<unsigned int>((tensor_cols(out) + block.x - 1) / block.x),
          static_cast<unsigned int>((tensor_rows(out) + block.y - 1) / block.y));
      matmul_right_transposed_kernel<<<grid, block>>>(
          to_kernel_tensor_view(a), to_kernel_tensor_view(b),
          to_kernel_tensor_view(out));
      check_kernel_launch("matmul_right_transposed_kernel(ranked reduce)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank() &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const int m = static_cast<int>(logical_prefix_count(a, 2) *
                                     static_cast<uint64_t>(a.dim(a.rank() - 2)));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(0));
      const float alpha = 1.0f;
      const float beta = 0.0f;
      check_cublas(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                               &alpha, b.f32(), k, a.f32(), k, &beta,
                               out.f32(), n),
                   "cublasSgemm(matmul_right_transposed ranked x matrix)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank() &&
        is_cuda_f32_contiguous_row_major(a) &&
        is_cuda_f32_contiguous_row_major(b) &&
        is_cuda_f32_contiguous_row_major(out)) {
      const int batch_count = static_cast<int>(logical_prefix_count(a, 2));
      const int m = static_cast<int>(a.dim(a.rank() - 2));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(b.rank() - 2));
      const long long int stride_a = static_cast<long long>(m) * k;
      const long long int stride_b = static_cast<long long>(n) * k;
      const long long int stride_out = static_cast<long long>(m) * n;
      const float alpha = 1.0f;
      const float beta = 0.0f;
      check_cublas(cublasSgemmStridedBatched(
                       handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                       b.f32(), k, stride_b, a.f32(), k, stride_a, &beta,
                       out.f32(), n, stride_out, batch_count),
                   "cublasSgemmStridedBatched(matmul_right_transposed ranked)");
      return;
    }
    if (a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank()) {
      if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_rows(b) == 0) {
        return;
      }
      const uint64_t prefix_count = logical_prefix_count(a, 2);
      const int m = static_cast<int>(a.dim(a.rank() - 2));
      const int k = static_cast<int>(a.dim(a.rank() - 1));
      const int n = static_cast<int>(b.dim(b.rank() - 2));
      const int lda = leading_dim_f32(a);
      const int ldb = leading_dim_f32(b);
      const int ldc = leading_dim_f32(out);
      const float alpha = 1.0f;
      const float beta = 0.0f;
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        check_cublas(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                 &alpha, prefix_matrix_ptr(b, prefix), ldb,
                                 prefix_matrix_ptr(a, prefix), lda, &beta,
                                 prefix_matrix_ptr(out, prefix), ldc),
                     "cublasSgemm(matmul_right_transposed ranked non-contiguous)");
      }
      return;
    }
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_rows(b) == 0) {
      return;
    }
    const dim3 block(kTileDim, kTileDim);
    const dim3 grid(
        static_cast<unsigned int>((tensor_cols(out) + block.x - 1) / block.x),
        static_cast<unsigned int>((tensor_rows(out) + block.y - 1) / block.y));
    matmul_right_transposed_kernel<<<grid, block>>>(
        to_kernel_tensor_view(a), to_kernel_tensor_view(b),
        to_kernel_tensor_view(out));
    check_kernel_launch("matmul_right_transposed_kernel");
  }

  void transpose(const TensorView &x, TensorView &out) override {
    require_cuda_f32_row_major(x, "transpose(x)");
    require_cuda_f32_row_major(out, "transpose(out)");
    if (x.data() == out.data()) {
      throw std::runtime_error(
          "cuda_cublas_plugin: transpose requires distinct input and output buffers");
    }
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    const dim3 block(kTileDim, kTileDim);
    const dim3 grid(
        static_cast<unsigned int>((tensor_cols(x) + block.x - 1) / block.x),
        static_cast<unsigned int>((tensor_rows(x) + block.y - 1) / block.y));
    transpose_kernel<<<grid, block>>>(to_kernel_tensor_view(x),
                                      to_kernel_tensor_view(out));
    check_kernel_launch("transpose_kernel");
  }

  void layernorm_forward(const TensorView &x, const TensorView &gamma_1xC,
                         const TensorView &beta_1xC, TensorView &out) override {
    require_cuda_f32_row_major(x, "layernorm_forward(x)");
    require_cuda_f32_row_major(gamma_1xC, "layernorm_forward(gamma)");
    require_cuda_f32_row_major(beta_1xC, "layernorm_forward(beta)");
    require_cuda_f32_row_major(out, "layernorm_forward(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    layernorm_forward_kernel<<<static_cast<unsigned int>(tensor_rows(x)),
                               kThreadsPerBlock,
                               static_cast<size_t>(kThreadsPerBlock *
                                                   sizeof(float))>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(gamma_1xC),
        to_kernel_tensor_view(beta_1xC), to_kernel_tensor_view(out));
    check_kernel_launch("layernorm_forward_kernel");
  }

  void layernorm_backward(const TensorView &x, const TensorView &gamma_1xC,
                          const TensorView &dout, TensorView &dx,
                          TensorView &dgamma_1xC,
                          TensorView &dbeta_1xC) override {
    require_cuda_f32_row_major(x, "layernorm_backward(x)");
    require_cuda_f32_row_major(gamma_1xC, "layernorm_backward(gamma)");
    require_cuda_f32_row_major(dout, "layernorm_backward(dout)");
    require_cuda_f32_row_major(dx, "layernorm_backward(dx)");
    require_cuda_f32_row_major(dgamma_1xC, "layernorm_backward(dgamma)");
    require_cuda_f32_row_major(dbeta_1xC, "layernorm_backward(dbeta)");
    check_cuda(cudaMemset(dgamma_1xC.data(), 0,
                          static_cast<size_t>(row_bytes(dgamma_1xC))),
               "cudaMemset(layernorm_backward(dgamma))");
    check_cuda(cudaMemset(dbeta_1xC.data(), 0,
                          static_cast<size_t>(row_bytes(dbeta_1xC))),
               "cudaMemset(layernorm_backward(dbeta))");
    if (tensor_rows(x) == 0 || tensor_cols(x) == 0) {
      return;
    }
    layernorm_backward_kernel<<<
        static_cast<unsigned int>(tensor_rows(x)), kThreadsPerBlock,
        static_cast<size_t>(kThreadsPerBlock * sizeof(double))>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(gamma_1xC),
        to_kernel_tensor_view(dout), to_kernel_tensor_view(dx),
        to_kernel_tensor_view(dgamma_1xC), to_kernel_tensor_view(dbeta_1xC));
    check_kernel_launch("layernorm_backward_kernel");
  }

  void embedding_lookup(const TensorView &table, const TensorView &ids,
                        TensorView &out) override {
    require_cuda_f32_row_major(table, "embedding_lookup(table)");
    require_cuda_row_major(ids, "embedding_lookup(ids)");
    require_cuda_f32_row_major(out, "embedding_lookup(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    const int64_t total = tensor_rows(out) * tensor_cols(out);
    embedding_lookup_kernel<<<
        static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock),
        kThreadsPerBlock>>>(to_kernel_tensor_view(table),
                            to_kernel_index_vector_view(ids),
                            to_kernel_tensor_view(out));
    check_kernel_launch("embedding_lookup_kernel");
  }

  void accumulate_embedding_grads(const TensorView &ids,
                                  const TensorView &d_cur, TensorView &d_tok,
                                  TensorView &d_pos) override {
    require_cuda_row_major(ids, "accumulate_embedding_grads(ids)");
    require_cuda_f32_row_major(d_cur, "accumulate_embedding_grads(d_cur)");
    require_cuda_f32_row_major(d_tok, "accumulate_embedding_grads(d_tok)");
    require_cuda_f32_row_major(d_pos, "accumulate_embedding_grads(d_pos)");
    if (tensor_rows(d_cur) == 0 || tensor_cols(d_cur) == 0) {
      return;
    }
    const int64_t total = tensor_rows(d_cur) * tensor_cols(d_cur);
    accumulate_embedding_grads_kernel<<<
        static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock),
        kThreadsPerBlock>>>(to_kernel_index_vector_view(ids),
                            to_kernel_tensor_view(d_cur),
                            to_kernel_tensor_view(d_tok),
                            to_kernel_tensor_view(d_pos),
                            ids.dim(ids.rank() - 1));
    check_kernel_launch("accumulate_embedding_grads_kernel");
  }

  void cross_entropy_mean(const TensorView &logits, const TensorView &targets,
                          TensorView &out_loss) override {
    fallback_cross_entropy_mean(logits, targets, out_loss);
  }

  void cross_entropy_mean_backward_inplace(TensorView &logits,
                                           const TensorView &targets,
                                           TensorView &out_loss) override {
    require_cuda_f32_row_major(logits, "cross_entropy_mean_backward_inplace(logits)");
    require_cuda_row_major(targets, "cross_entropy_mean_backward_inplace(targets)");
    require_cuda_f32_row_major(out_loss, "cross_entropy_mean_backward_inplace(out_loss)");
    check_cuda(cudaMemset(out_loss.data(), 0, sizeof(float)),
               "cudaMemset(cross_entropy_mean_backward_inplace(out_loss))");
    cross_entropy_mean_backward_inplace_kernel<<<
        static_cast<unsigned int>(tensor_rows(logits)), kThreadsPerBlock,
        static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
        to_kernel_tensor_view(logits), to_kernel_index_vector_view(targets),
        to_kernel_tensor_view(out_loss));
    check_kernel_launch("cross_entropy_mean_backward_inplace_kernel");
  }

  float read_scalar_f32(const TensorView &x) override {
    require_cuda_f32_row_major(x, "read_scalar_f32");
    float value = 0.0f;
    copy_device2host(&value, x.data(), sizeof(float));
    return value;
  }

  void backward_from_logits_targets(TensorView &logits,
                                    const TensorView &targets) override {
    fallback_backward_from_logits_targets(logits, targets);
  }

  void softmax_rows(const TensorView &x, TensorView &out) override {
    require_cuda_f32_row_major(x, "softmax_rows(x)");
    require_cuda_f32_row_major(out, "softmax_rows(out)");
    if (tensor_rows(out) == 0 || tensor_cols(out) == 0) {
      return;
    }
    softmax_rows_kernel<<<static_cast<unsigned int>(tensor_rows(x)),
                          kThreadsPerBlock,
                          static_cast<size_t>(kThreadsPerBlock *
                                              sizeof(float))>>>(
        to_kernel_tensor_view(x), to_kernel_tensor_view(out));
    check_kernel_launch("softmax_rows_kernel");
  }

  void softmax_backward_rows(const TensorView &softmax, const TensorView &dout,
                             TensorView &dx) override {
    require_cuda_f32_row_major(softmax, "softmax_backward_rows(softmax)");
    require_cuda_f32_row_major(dout, "softmax_backward_rows(dout)");
    require_cuda_f32_row_major(dx, "softmax_backward_rows(dx)");
    if (tensor_rows(dx) == 0 || tensor_cols(dx) == 0) {
      return;
    }
    softmax_backward_rows_kernel<<<
        static_cast<unsigned int>(tensor_rows(softmax)), kThreadsPerBlock,
        static_cast<size_t>(kThreadsPerBlock * sizeof(float))>>>(
        to_kernel_tensor_view(softmax), to_kernel_tensor_view(dout),
        to_kernel_tensor_view(dx));
    check_kernel_launch("softmax_backward_rows_kernel");
  }

  void apply_causal_mask_inplace(TensorView &scores,
                                 float neg_inf = -1e9f) override {
    require_cuda_f32_row_major(scores, "apply_causal_mask_inplace");
    if (tensor_rows(scores) == 0 || tensor_cols(scores) == 0) {
      return;
    }
    const dim3 block(kTileDim, kTileDim);
    const dim3 grid(
        static_cast<unsigned int>((tensor_cols(scores) + block.x - 1) / block.x),
        static_cast<unsigned int>((tensor_rows(scores) + block.y - 1) / block.y));
    apply_causal_mask_inplace_kernel<<<grid, block>>>(
        to_kernel_tensor_view(scores), neg_inf);
    check_kernel_launch("apply_causal_mask_inplace_kernel");
  }

  void adamw_step(TensorView &params, const TensorView &grads, TensorView &m,
                  TensorView &v, uint64_t step, float learning_rate,
                  float beta1, float beta2, float weight_decay,
                  bool apply_weight_decay) override {
    require_cuda_f32_row_major(params, "adamw_step(params)");
    require_cuda_f32_row_major(grads, "adamw_step(grads)");
    require_cuda_f32_row_major(m, "adamw_step(m)");
    require_cuda_f32_row_major(v, "adamw_step(v)");
    if (step < 1) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step step must be >= 1");
    }
    if (tensor_rows(params) != tensor_rows(grads) || tensor_cols(params) != tensor_cols(grads)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step params/grads shape mismatch");
    }
    if (tensor_rows(params) != tensor_rows(m) || tensor_cols(params) != tensor_cols(m)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step params/m shape mismatch");
    }
    if (tensor_rows(params) != tensor_rows(v) || tensor_cols(params) != tensor_cols(v)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step params/v shape mismatch");
    }
    if (learning_rate <= 0.0f) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step learning_rate must be > 0");
    }
    if (!(beta1 >= 0.0f && beta1 < 1.0f)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step beta1 must be in [0,1)");
    }
    if (!(beta2 >= 0.0f && beta2 < 1.0f)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step beta2 must be in [0,1)");
    }
    if (tensor_rows(params) == 0 || tensor_cols(params) == 0) {
      return;
    }

    const double t = static_cast<double>(step);
    const float b1_corr =
        1.0f - static_cast<float>(std::pow(static_cast<double>(beta1), t));
    const float b2_corr =
        1.0f - static_cast<float>(std::pow(static_cast<double>(beta2), t));
    if (!(b1_corr > 0.0f && b2_corr > 0.0f)) {
      throw std::runtime_error("cuda_cublas_plugin: adamw_step invalid bias correction");
    }

    const int64_t total = tensor_rows(params) * tensor_cols(params);
    adamw_step_kernel<<<
        static_cast<unsigned int>((total + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock),
        kThreadsPerBlock>>>(
        to_kernel_tensor_view(params), to_kernel_tensor_view(grads),
        to_kernel_tensor_view(m), to_kernel_tensor_view(v), learning_rate,
        beta1, beta2, weight_decay, 1.0f / b1_corr, 1.0f / b2_corr,
        apply_weight_decay ? 1u : 0u);
    check_kernel_launch("adamw_step_kernel");
  }

  bool is_file2device_read_supported() const override { return true; }

  void read_file2device(const std::string &path, void *dst, uint64_t size,
                        uint64_t file_offset) override {
    if (size == 0) {
      return;
    }
    std::vector<uint8_t> host_buffer(static_cast<size_t>(size));
    cpu_backend_.read_file2device(path, host_buffer.data(), size, file_offset);
    copy_host2device(dst, host_buffer.data(), size);
  }

private:
  void copy_stage_from_device(const TensorView &src, HostTensorStage &dst) {
    copy_tensor_2d(src, dst.view, cudaMemcpyDeviceToHost,
                   "cudaMemcpy2D(device_to_host_stage)");
  }

  void copy_stage_to_device(const HostTensorStage &src, const TensorView &dst) {
    copy_tensor_2d(src.view, dst, cudaMemcpyHostToDevice,
                   "cudaMemcpy2D(host_stage_to_device)");
  }

  void fallback_cross_entropy_mean(const TensorView &logits,
                                   const TensorView &targets,
                                   TensorView &out_loss) {
    HostTensorStage logits_host(logits);
    HostTensorStage targets_host(targets);
    HostTensorStage out_loss_host(out_loss);

    copy_stage_from_device(logits, logits_host);
    copy_stage_from_device(targets, targets_host);

    cpu_backend_.cross_entropy_mean(logits_host.view, targets_host.view,
                                    out_loss_host.view);
    copy_stage_to_device(out_loss_host, out_loss);
  }

  void fallback_backward_from_logits_targets(TensorView &logits,
                                             const TensorView &targets) {
    HostTensorStage logits_host(logits);
    HostTensorStage targets_host(targets);

    copy_stage_from_device(logits, logits_host);
    copy_stage_from_device(targets, targets_host);

    cpu_backend_.backward_from_logits_targets(logits_host.view, targets_host.view);
    copy_stage_to_device(logits_host, logits);
  }

  cublasHandle_t handle_ = nullptr;
  DefaultCpuBackend cpu_backend_;
};

CuBlasPluginBackend &to_cublas_backend(void *backend) {
  if (backend == nullptr) {
    throw std::runtime_error("cuda_cublas_plugin: null backend instance");
  }
  return *reinterpret_cast<CuBlasPluginBackend *>(backend);
}

void *plugin_create(uint32_t device) {
  if (static_cast<Device>(device) != Device::GPU) {
    throw std::runtime_error("cuda_cublas_plugin: only gpu device is supported");
  }
  return new CuBlasPluginBackend();
}

void plugin_destroy(void *backend) {
  delete reinterpret_cast<CuBlasPluginBackend *>(backend);
}

uint32_t plugin_device(void *backend) {
  return static_cast<uint32_t>(to_cublas_backend(backend).device());
}

void *plugin_alloc(void *backend, uint64_t bytes, uint32_t alignment) {
  return to_cublas_backend(backend).alloc(bytes, alignment);
}

void plugin_free(void *backend, void *ptr) {
  to_cublas_backend(backend).free(ptr);
}

void plugin_copy_host2device(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  to_cublas_backend(backend).copy_host2device(dst, src, bytes);
}

void plugin_copy_device2host(void *backend, void *dst, const void *src,
                             uint64_t bytes) {
  to_cublas_backend(backend).copy_device2host(dst, src, bytes);
}

void plugin_copy(void *backend, const BackendTensorView *src,
                 const BackendTensorView *dst) {
  TensorView dst_view = to_tensor_view(*dst);
  to_cublas_backend(backend).copy(to_tensor_view(*src), dst_view);
}

void plugin_fill(void *backend, const BackendTensorView *t, float v) {
  TensorView t_view = to_tensor_view(*t);
  to_cublas_backend(backend).fill(t_view, v);
}

void plugin_add(void *backend, const BackendTensorView *a,
                const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).add(to_tensor_view(*a), to_tensor_view(*b),
                                 out_view);
}

void plugin_add_inplace(void *backend, const BackendTensorView *a,
                        const BackendTensorView *b) {
  TensorView a_view = to_tensor_view(*a);
  to_cublas_backend(backend).add_inplace(a_view, to_tensor_view(*b));
}

void plugin_add_bias_rowwise(void *backend, const BackendTensorView *x,
                             const BackendTensorView *bias_1xC,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).add_bias_rowwise(
      to_tensor_view(*x), to_tensor_view(*bias_1xC), out_view);
}

void plugin_mul_scalar(void *backend, const BackendTensorView *x, float s,
                       const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).mul_scalar(to_tensor_view(*x), s, out_view);
}

float plugin_sum_squares_f32(void *backend, const BackendTensorView *x) {
  return to_cublas_backend(backend).sum_squares_f32(to_tensor_view(*x));
}

void plugin_relu(void *backend, const BackendTensorView *x,
                 const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).relu(to_tensor_view(*x), out_view);
}

void plugin_relu_backward(void *backend, const BackendTensorView *preact,
                          const BackendTensorView *dout,
                          const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_cublas_backend(backend).relu_backward(to_tensor_view(*preact),
                                           to_tensor_view(*dout), dx_view);
}

void plugin_row_sum(void *backend, const BackendTensorView *x,
                    const BackendTensorView *out_1xC) {
  TensorView out_view = to_tensor_view(*out_1xC);
  to_cublas_backend(backend).row_sum(to_tensor_view(*x), out_view);
}

void plugin_matmul(void *backend, const BackendTensorView *a,
                   const BackendTensorView *b, const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).matmul(to_tensor_view(*a), to_tensor_view(*b),
                                    out_view);
}

void plugin_matmul_left_transposed(void *backend, const BackendTensorView *a,
                                   const BackendTensorView *b,
                                   const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).matmul_left_transposed(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_matmul_right_transposed(void *backend, const BackendTensorView *a,
                                    const BackendTensorView *b,
                                    const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).matmul_right_transposed(
      to_tensor_view(*a), to_tensor_view(*b), out_view);
}

void plugin_transpose(void *backend, const BackendTensorView *x,
                      const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).transpose(to_tensor_view(*x), out_view);
}

void plugin_layernorm_forward(void *backend, const BackendTensorView *x,
                              const BackendTensorView *gamma_1xC,
                              const BackendTensorView *beta_1xC,
                              const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).layernorm_forward(
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
  to_cublas_backend(backend).layernorm_backward(
      to_tensor_view(*x), to_tensor_view(*gamma_1xC), to_tensor_view(*dout),
      dx_view, dgamma_view, dbeta_view);
}

void plugin_embedding_lookup(void *backend, const BackendTensorView *table,
                             const BackendTensorView *ids,
                             const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).embedding_lookup(
      to_tensor_view(*table), to_tensor_view(*ids), out_view);
}

void plugin_accumulate_embedding_grads(void *backend,
                                       const BackendTensorView *ids,
                                       const BackendTensorView *d_cur,
                                       const BackendTensorView *d_tok,
                                       const BackendTensorView *d_pos) {
  TensorView d_tok_view = to_tensor_view(*d_tok);
  TensorView d_pos_view = to_tensor_view(*d_pos);
  to_cublas_backend(backend).accumulate_embedding_grads(
      to_tensor_view(*ids), to_tensor_view(*d_cur), d_tok_view, d_pos_view);
}

void plugin_cross_entropy_mean(void *backend, const BackendTensorView *logits,
                               const BackendTensorView *targets,
                               const BackendTensorView *out_loss) {
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_cublas_backend(backend).cross_entropy_mean(
      to_tensor_view(*logits), to_tensor_view(*targets), out_loss_view);
}

void plugin_cross_entropy_mean_backward_inplace(
    void *backend, const BackendTensorView *logits,
    const BackendTensorView *targets, const BackendTensorView *out_loss) {
  TensorView logits_view = to_tensor_view(*logits);
  TensorView out_loss_view = to_tensor_view(*out_loss);
  to_cublas_backend(backend).cross_entropy_mean_backward_inplace(
      logits_view, to_tensor_view(*targets), out_loss_view);
}

float plugin_read_scalar_f32(void *backend, const BackendTensorView *x) {
  return to_cublas_backend(backend).read_scalar_f32(to_tensor_view(*x));
}

void plugin_backward_from_logits_targets(void *backend,
                                         const BackendTensorView *logits,
                                         const BackendTensorView *targets) {
  TensorView logits_view = to_tensor_view(*logits);
  to_cublas_backend(backend).backward_from_logits_targets(
      logits_view, to_tensor_view(*targets));
}

void plugin_softmax_rows(void *backend, const BackendTensorView *x,
                         const BackendTensorView *out) {
  TensorView out_view = to_tensor_view(*out);
  to_cublas_backend(backend).softmax_rows(to_tensor_view(*x), out_view);
}

void plugin_softmax_backward_rows(void *backend,
                                  const BackendTensorView *softmax,
                                  const BackendTensorView *dout,
                                  const BackendTensorView *dx) {
  TensorView dx_view = to_tensor_view(*dx);
  to_cublas_backend(backend).softmax_backward_rows(to_tensor_view(*softmax),
                                                   to_tensor_view(*dout),
                                                   dx_view);
}

void plugin_apply_causal_mask_inplace(void *backend,
                                      const BackendTensorView *scores,
                                      float neg_inf) {
  TensorView scores_view = to_tensor_view(*scores);
  to_cublas_backend(backend).apply_causal_mask_inplace(scores_view, neg_inf);
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
  to_cublas_backend(backend).adamw_step(params_view, to_tensor_view(*grads),
                                        m_view, v_view, step, learning_rate,
                                        beta1, beta2, weight_decay,
                                        apply_weight_decay != 0);
}

uint32_t plugin_is_file2device_read_supported(void *backend) {
  return to_cublas_backend(backend).is_file2device_read_supported() ? 1u : 0u;
}

void plugin_read_file2device(void *backend, const char *path, void *dst,
                             uint64_t size, uint64_t file_offset) {
  to_cublas_backend(backend).read_file2device(path, dst, size, file_offset);
}

BackendMemoryInfo plugin_memory_info(void *backend) {
  const DeviceMemoryInfo info = to_cublas_backend(backend).memory_info();
  return BackendMemoryInfo{info.available ? 1u : 0u, info.free_bytes,
                           info.total_bytes};
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
    &plugin_mul_scalar,
    &plugin_sum_squares_f32,
    &plugin_relu,
    &plugin_relu_backward,
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
    &plugin_apply_causal_mask_inplace,
    &plugin_adamw_step,
    &plugin_is_file2device_read_supported,
    &plugin_read_file2device,
    &plugin_memory_info,
};

} // namespace

extern "C" const BackendApiV1 *litnice_backend_get_api() { return &kBackendApi; }
