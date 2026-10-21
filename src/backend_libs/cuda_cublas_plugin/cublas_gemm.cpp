#include "cublas_gemm.hpp"

#include <litnicelm_utils/checks.hpp>

#include <cstdint>
#include <string>

namespace cuda_cublas_plugin {
namespace {

constexpr bool fallback_to_prefix_loop = false;

struct GemmShape {
  int out_rows;
  int shared_dim;
  int out_cols;
};

struct GemmStrides {
  long long int a;
  long long int b;
  long long int out;
};

// A matrix as cuBLAS sees it: data, element type, row stride and batch
// stride in elements.
struct CublasInputMatrix {
  const void *data;
  cudaDataType_t type;
  int leading_dim;
  long long int stride;
};

struct CublasOutputMatrix {
  void *data;
  cudaDataType_t type;
  int leading_dim;
  long long int stride;
};

// The cuBLAS element type of a view: F32 or BF16.
cudaDataType_t cuda_data_type(const TensorView &view) {
  checker().check(CheckKind::Capability,
                  view.dtype() == DType::F32 || view.dtype() == DType::BF16,
                  "cuda_cublas_plugin", [&] {
                    return std::string("GEMM does not take dtype ") +
                           dtype_name(view.dtype());
                  });
  return view.dtype() == DType::BF16 ? CUDA_R_16BF : CUDA_R_32F;
}

// The whole view as one matrix with the given row stride in elements.
CublasInputMatrix input_matrix(const TensorView &view, int leading_dim) {
  return {view.data(), cuda_data_type(view), leading_dim, 0};
}

CublasOutputMatrix output_matrix(TensorView &view, int leading_dim) {
  return {view.data(), cuda_data_type(view), leading_dim, 0};
}

// The prefix-th matrix of view; batches are stride elements apart.
CublasInputMatrix input_matrix(const TensorView &view, uint64_t prefix,
                               int leading_dim, long long int stride) {
  return {prefix_matrix_ptr(view, prefix), cuda_data_type(view), leading_dim,
          stride};
}

CublasOutputMatrix output_matrix(TensorView &view, uint64_t prefix,
                                 int leading_dim, long long int stride) {
  return {prefix_matrix_ptr(view, prefix), cuda_data_type(view), leading_dim,
          stride};
}

// The _f32 GEMM: FP32 matrices, FP32 math (the handles use
// CUBLAS_DEFAULT_MATH, so no TF32).
void run_gemm_f32(cublasHandle_t handle, cublasOperation_t op_a,
                  cublasOperation_t op_b, const GemmShape &shape,
                  const CublasInputMatrix &a, const CublasInputMatrix &b,
                  const CublasOutputMatrix &out, const char *what) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasSgemm(handle, op_a, op_b, shape.out_cols, shape.out_rows,
                           shape.shared_dim, &alpha,
                           static_cast<const float *>(a.data), a.leading_dim,
                           static_cast<const float *>(b.data), b.leading_dim,
                           &beta, static_cast<float *>(out.data),
                           out.leading_dim),
               what);
}

void run_strided_batched_gemm_f32(cublasHandle_t handle,
                                  cublasOperation_t op_a,
                                  cublasOperation_t op_b,
                                  const GemmShape &shape,
                                  const CublasInputMatrix &a,
                                  const CublasInputMatrix &b,
                                  const CublasOutputMatrix &out,
                                  int batch_count, const char *what) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasSgemmStridedBatched(
                   handle, op_a, op_b, shape.out_cols, shape.out_rows,
                   shape.shared_dim, &alpha, static_cast<const float *>(a.data),
                   a.leading_dim, a.stride,
                   static_cast<const float *>(b.data), b.leading_dim, b.stride,
                   &beta, static_cast<float *>(out.data), out.leading_dim,
                   out.stride, batch_count),
               what);
}

// The _bf16act GEMM: BF16 inputs (activations and the BF16 weight copy),
// FP32 accumulation, BF16 output or F32 for weight gradients; tensor cores
// via cublasGemmEx.
void run_gemm_bf16act(cublasHandle_t handle, cublasOperation_t op_a,
                      cublasOperation_t op_b, const GemmShape &shape,
                      const CublasInputMatrix &a, const CublasInputMatrix &b,
                      const CublasOutputMatrix &out, const char *what) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasGemmEx(handle, op_a, op_b, shape.out_cols, shape.out_rows,
                            shape.shared_dim, &alpha, a.data, a.type,
                            a.leading_dim, b.data, b.type, b.leading_dim,
                            &beta, out.data, out.type, out.leading_dim,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               what);
}

void run_strided_batched_gemm_bf16act(cublasHandle_t handle,
                                      cublasOperation_t op_a,
                                      cublasOperation_t op_b,
                                      const GemmShape &shape,
                                      const CublasInputMatrix &a,
                                      const CublasInputMatrix &b,
                                      const CublasOutputMatrix &out,
                                      int batch_count, const char *what) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasGemmStridedBatchedEx(
                   handle, op_a, op_b, shape.out_cols, shape.out_rows,
                   shape.shared_dim, &alpha, a.data, a.type, a.leading_dim,
                   a.stride, b.data, b.type, b.leading_dim, b.stride, &beta,
                   out.data, out.type, out.leading_dim, out.stride,
                   batch_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               what);
}

void run_gemm(PrecisionVariant variant, cublasHandle_t handle, cublasOperation_t op_a,
              cublasOperation_t op_b, const GemmShape &shape,
              const CublasInputMatrix &a, const CublasInputMatrix &b,
              const CublasOutputMatrix &out, const char *what) {
  variant == PrecisionVariant::Bf16Act
      ? run_gemm_bf16act(handle, op_a, op_b, shape, a, b, out, what)
      : run_gemm_f32(handle, op_a, op_b, shape, a, b, out, what);
}

void run_strided_batched_gemm(PrecisionVariant variant, cublasHandle_t handle,
                              cublasOperation_t op_a, cublasOperation_t op_b,
                              const GemmShape &shape,
                              const CublasInputMatrix &a,
                              const CublasInputMatrix &b,
                              const CublasOutputMatrix &out, int batch_count,
                              const char *what) {
  variant == PrecisionVariant::Bf16Act
      ? run_strided_batched_gemm_bf16act(handle, op_a, op_b, shape, a, b, out,
                                         batch_count, what)
      : run_strided_batched_gemm_f32(handle, op_a, op_b, shape, a, b, out,
                                     batch_count, what);
}

GemmShape matmul_ranked_matrix_rhs_shape(const TensorView &a,
                                         const TensorView &b) {
  return {static_cast<int>(logical_prefix_count(a, 2) *
                           static_cast<uint64_t>(a.dim(a.rank() - 2))),
          static_cast<int>(a.dim(a.rank() - 1)), static_cast<int>(b.dim(1))};
}

GemmShape matmul_ranked_batch_shape(const TensorView &a, const TensorView &b) {
  return {static_cast<int>(a.dim(a.rank() - 2)),
          static_cast<int>(a.dim(a.rank() - 1)),
          static_cast<int>(b.dim(b.rank() - 1))};
}

GemmShape matmul_tensor_shape(const TensorView &a, const TensorView &b) {
  return {static_cast<int>(tensor_rows(a)), static_cast<int>(tensor_cols(a)),
          static_cast<int>(tensor_cols(b))};
}

GemmShape left_transposed_ranked_reduce_shape(const TensorView &a,
                                             const TensorView &b) {
  return {static_cast<int>(a.dim(a.rank() - 1)),
          static_cast<int>(logical_prefix_count(a, 2) *
                           static_cast<uint64_t>(a.dim(a.rank() - 2))),
          static_cast<int>(b.dim(b.rank() - 1))};
}

GemmShape left_transposed_ranked_batch_shape(const TensorView &a,
                                            const TensorView &b) {
  return {static_cast<int>(a.dim(a.rank() - 1)),
          static_cast<int>(a.dim(a.rank() - 2)),
          static_cast<int>(b.dim(b.rank() - 1))};
}

GemmShape left_transposed_tensor_shape(const TensorView &a,
                                      const TensorView &b) {
  return {static_cast<int>(tensor_cols(a)), static_cast<int>(tensor_rows(a)),
          static_cast<int>(tensor_cols(b))};
}

GemmShape right_transposed_ranked_matrix_rhs_shape(const TensorView &a,
                                                  const TensorView &b) {
  return {static_cast<int>(logical_prefix_count(a, 2) *
                           static_cast<uint64_t>(a.dim(a.rank() - 2))),
          static_cast<int>(a.dim(a.rank() - 1)), static_cast<int>(b.dim(0))};
}

GemmShape right_transposed_ranked_batch_shape(const TensorView &a,
                                             const TensorView &b) {
  return {static_cast<int>(a.dim(a.rank() - 2)),
          static_cast<int>(a.dim(a.rank() - 1)),
          static_cast<int>(b.dim(b.rank() - 2))};
}

GemmStrides matmul_contiguous_strides(const GemmShape &shape) {
  return {static_cast<long long int>(shape.out_rows) * shape.shared_dim,
          static_cast<long long int>(shape.shared_dim) * shape.out_cols,
          static_cast<long long int>(shape.out_rows) * shape.out_cols};
}

GemmStrides left_transposed_contiguous_strides(const GemmShape &shape) {
  return {static_cast<long long int>(shape.shared_dim) * shape.out_rows,
          static_cast<long long int>(shape.shared_dim) * shape.out_cols,
          static_cast<long long int>(shape.out_rows) * shape.out_cols};
}

GemmStrides right_transposed_contiguous_strides(const GemmShape &shape) {
  return {static_cast<long long int>(shape.out_rows) * shape.shared_dim,
          static_cast<long long int>(shape.out_cols) * shape.shared_dim,
          static_cast<long long int>(shape.out_rows) * shape.out_cols};
}

// The batch stride in elements, if consecutive prefix matrices are evenly
// spaced.
bool regular_prefix_stride(const TensorView &view, uint64_t prefix_count,
                           long long int &stride) {
  if (prefix_count <= 1) {
    stride = 0;
    return true;
  }
  const int64_t elem = static_cast<int64_t>(dtype_size(view.dtype()));
  if (view.rank() == 3) {
    const int64_t stride_bytes = view.stride_bytes(0);
    if (stride_bytes % elem != 0) {
      return false;
    }
    stride = static_cast<long long int>(stride_bytes / elem);
    return true;
  }
  const int64_t first_offset =
      prefix_matrix_byte_offset(view, 1, 2) -
      prefix_matrix_byte_offset(view, 0, 2);
  if (first_offset % elem != 0) {
    return false;
  }
  for (uint64_t prefix = 2; prefix < prefix_count; ++prefix) {
    const int64_t delta = prefix_matrix_byte_offset(view, prefix, 2) -
                          prefix_matrix_byte_offset(view, prefix - 1, 2);
    if (delta != first_offset) {
      return false;
    }
  }
  stride = static_cast<long long int>(first_offset / elem);
  return true;
}

} // namespace

void cublas_gemm(cublasHandle_t handle, const TensorView &a,
                 const TensorView &b, TensorView &out) {
  check_cuda_row_major(a, {DType::F32}, "gemm(a)");
  check_cuda_row_major(b, {DType::F32}, "gemm(b)");
  check_cuda_row_major(out, {DType::F32}, "gemm(out)");
  checker().check(CheckKind::Contract,
                  tensor_rows(a) != 0 && tensor_cols(a) != 0 &&
                      tensor_cols(b) != 0,
                  "cuda_cublas_plugin",
                  "gemm would run no GEMM for "
                  "zero-sized tensors");
  const GemmShape shape = matmul_tensor_shape(a, b);
  run_gemm(PrecisionVariant::F32, handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
           input_matrix(b, leading_dim(b)), input_matrix(a, leading_dim(a)),
           output_matrix(out, leading_dim(out)), "GEMM (gemm)");
}

void cublas_gemm_ranked_matrix_rhs(cublasHandle_t handle, const TensorView &a,
                                   const TensorView &b, TensorView &out) {
  // The variant follows the activation operand a; b is an activation or,
  // in _bf16act, the BF16 weight copy.
  const PrecisionVariant variant = precision_variant_of(a);
  const DType act = activation_dtype(variant);
  check_cuda_row_major(a, {act}, "gemm_ranked_matrix_rhs(a)");
  check_cuda_row_major(b, {act}, "gemm_ranked_matrix_rhs(b)");
  check_cuda_row_major(out, {act}, "gemm_ranked_matrix_rhs(out)");
  const GemmShape shape = matmul_ranked_matrix_rhs_shape(a, b);
  run_gemm(variant, handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
           input_matrix(b, shape.out_cols), input_matrix(a, shape.shared_dim),
           output_matrix(out, shape.out_cols),
           "GEMM (gemm_ranked_matrix_rhs)");
}

void cublas_gemm_ranked_matrix_rhs_t(cublasHandle_t handle,
                                     const TensorView &a,
                                     const TensorView &b, TensorView &out) {
  // The variant follows the activation operand a; b is an activation or,
  // in _bf16act, the BF16 weight copy.
  const PrecisionVariant variant = precision_variant_of(a);
  const DType act = activation_dtype(variant);
  check_cuda_row_major(a, {act}, "gemm_ranked_matrix_rhs_t(a)");
  check_cuda_row_major(b, {act}, "gemm_ranked_matrix_rhs_t(b)");
  check_cuda_row_major(out, {act}, "gemm_ranked_matrix_rhs_t(out)");
  const GemmShape shape = right_transposed_ranked_matrix_rhs_shape(a, b);
  run_gemm(variant, handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
           input_matrix(b, shape.shared_dim), input_matrix(a, shape.shared_dim),
           output_matrix(out, shape.out_cols),
           "GEMM (gemm_ranked_matrix_rhs_t)");
}

void cublas_gemm_ranked_reduce_lhs_t(cublasHandle_t handle,
                                     const TensorView &a,
                                     const TensorView &b, TensorView &out) {
  // The variant follows the activation operand a; b is an activation or,
  // in _bf16act, the BF16 weight copy.
  const PrecisionVariant variant = precision_variant_of(a);
  const DType act = activation_dtype(variant);
  check_cuda_row_major(a, {act}, "gemm_ranked_reduce_lhs_t(a)");
  check_cuda_row_major(b, {act}, "gemm_ranked_reduce_lhs_t(b)");
  check_cuda_row_major(out, {DType::F32}, "gemm_ranked_reduce_lhs_t(out)");
  const GemmShape shape = left_transposed_ranked_reduce_shape(a, b);
  run_gemm(variant, handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
           input_matrix(b, shape.out_cols), input_matrix(a, shape.out_rows),
           output_matrix(out, shape.out_cols),
           "GEMM (gemm_ranked_reduce_lhs_t)");
}

void cublas_gemm_batched(cublasHandle_t handle, const TensorView &a,
                         const TensorView &b, TensorView &out) {
  // The variant follows the activation operand a; b is an activation or,
  // in _bf16act, the BF16 weight copy.
  const PrecisionVariant variant = precision_variant_of(a);
  const DType act = activation_dtype(variant);
  check_cuda_row_major(a, {act}, "gemm_batched(a)");
  check_cuda_row_major(b, {act}, "gemm_batched(b)");
  check_cuda_row_major(out, {act}, "gemm_batched(out)");
  checker().check(CheckKind::Contract,
                  tensor_rows(a) != 0 && tensor_cols(a) != 0 &&
                      tensor_cols(b) != 0,
                  "cuda_cublas_plugin",
                  "gemm_batched would run no GEMM for "
                  "zero-sized tensors");
  const uint64_t prefix_count = logical_prefix_count(a, 2);
  const GemmShape shape = matmul_ranked_batch_shape(a, b);
  const int a_leading_dim = leading_dim(a);
  const int b_leading_dim = leading_dim(b);
  const int out_leading_dim = leading_dim(out);
  long long int stride_a = 0;
  long long int stride_b = 0;
  long long int stride_out = 0;
  if (regular_prefix_stride(a, prefix_count, stride_a) &&
      regular_prefix_stride(b, prefix_count, stride_b) &&
      regular_prefix_stride(out, prefix_count, stride_out)) {
    run_strided_batched_gemm(variant, handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
                             input_matrix(b, 0, b_leading_dim, stride_b),
                             input_matrix(a, 0, a_leading_dim, stride_a),
                             output_matrix(out, 0, out_leading_dim, stride_out),
                             static_cast<int>(prefix_count),
                             "GEMM (gemm_batched)");
    return;
  }
  checker().check(CheckKind::Capability, fallback_to_prefix_loop,
                  "cuda_cublas_plugin",
                  "gemm_batched irregular ranked layout is "
                  "not supported");
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    run_gemm(variant, handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
             input_matrix(b, prefix, b_leading_dim, 0),
             input_matrix(a, prefix, a_leading_dim, 0),
             output_matrix(out, prefix, out_leading_dim, 0),
             "GEMM (gemm_batched prefix)");
  }
}

void cublas_gemm_batched_lhs_t(cublasHandle_t handle, const TensorView &a,
                               const TensorView &b, TensorView &out) {
  // The variant follows the activation operand a; b is an activation or,
  // in _bf16act, the BF16 weight copy.
  const PrecisionVariant variant = precision_variant_of(a);
  const DType act = activation_dtype(variant);
  check_cuda_row_major(a, {act}, "gemm_batched_lhs_t(a)");
  check_cuda_row_major(b, {act}, "gemm_batched_lhs_t(b)");
  check_cuda_row_major(out, {act}, "gemm_batched_lhs_t(out)");
  checker().check(CheckKind::Contract,
                  tensor_rows(a) != 0 && tensor_cols(a) != 0 &&
                      tensor_cols(b) != 0,
                  "cuda_cublas_plugin",
                  "gemm_batched_lhs_t would run no GEMM for "
                  "zero-sized tensors");
  const uint64_t prefix_count = logical_prefix_count(a, 2);
  const GemmShape shape = left_transposed_ranked_batch_shape(a, b);
  const int a_leading_dim = leading_dim(a);
  const int b_leading_dim = leading_dim(b);
  const int out_leading_dim = leading_dim(out);
  long long int stride_a = 0;
  long long int stride_b = 0;
  long long int stride_out = 0;
  if (regular_prefix_stride(a, prefix_count, stride_a) &&
      regular_prefix_stride(b, prefix_count, stride_b) &&
      regular_prefix_stride(out, prefix_count, stride_out)) {
    run_strided_batched_gemm(variant, handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
                             input_matrix(b, 0, b_leading_dim, stride_b),
                             input_matrix(a, 0, a_leading_dim, stride_a),
                             output_matrix(out, 0, out_leading_dim, stride_out),
                             static_cast<int>(prefix_count),
                             "GEMM (gemm_batched_lhs_t)");
    return;
  }
  checker().check(CheckKind::Capability, fallback_to_prefix_loop,
                  "cuda_cublas_plugin",
                  "gemm_batched_lhs_t irregular ranked layout is "
                  "not supported");
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    run_gemm(variant, handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
             input_matrix(b, prefix, b_leading_dim, 0),
             input_matrix(a, prefix, a_leading_dim, 0),
             output_matrix(out, prefix, out_leading_dim, 0),
             "GEMM (gemm_batched_lhs_t prefix)");
  }
}

void cublas_gemm_batched_rhs_t(cublasHandle_t handle, const TensorView &a,
                               const TensorView &b, TensorView &out) {
  // The variant follows the activation operand a; b is an activation or,
  // in _bf16act, the BF16 weight copy.
  const PrecisionVariant variant = precision_variant_of(a);
  const DType act = activation_dtype(variant);
  check_cuda_row_major(a, {act}, "gemm_batched_rhs_t(a)");
  check_cuda_row_major(b, {act}, "gemm_batched_rhs_t(b)");
  check_cuda_row_major(out, {act}, "gemm_batched_rhs_t(out)");
  checker().check(CheckKind::Contract,
                  tensor_rows(a) != 0 && tensor_cols(a) != 0 &&
                      tensor_rows(b) != 0,
                  "cuda_cublas_plugin",
                  "gemm_batched_rhs_t would run no GEMM for "
                  "zero-sized tensors");
  const uint64_t prefix_count = logical_prefix_count(a, 2);
  const GemmShape shape = right_transposed_ranked_batch_shape(a, b);
  const int a_leading_dim = leading_dim(a);
  const int b_leading_dim = leading_dim(b);
  const int out_leading_dim = leading_dim(out);
  long long int stride_a = 0;
  long long int stride_b = 0;
  long long int stride_out = 0;
  if (regular_prefix_stride(a, prefix_count, stride_a) &&
      regular_prefix_stride(b, prefix_count, stride_b) &&
      regular_prefix_stride(out, prefix_count, stride_out)) {
    run_strided_batched_gemm(variant, handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
                             input_matrix(b, 0, b_leading_dim, stride_b),
                             input_matrix(a, 0, a_leading_dim, stride_a),
                             output_matrix(out, 0, out_leading_dim, stride_out),
                             static_cast<int>(prefix_count),
                             "GEMM (gemm_batched_rhs_t)");
    return;
  }
  checker().check(CheckKind::Capability, fallback_to_prefix_loop,
                  "cuda_cublas_plugin",
                  "gemm_batched_rhs_t irregular ranked layout is "
                  "not supported");
  for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
    run_gemm(variant, handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
             input_matrix(b, prefix, b_leading_dim, 0),
             input_matrix(a, prefix, a_leading_dim, 0),
             output_matrix(out, prefix, out_leading_dim, 0),
             "GEMM (gemm_batched_rhs_t prefix)");
  }
}

} // namespace cuda_cublas_plugin
