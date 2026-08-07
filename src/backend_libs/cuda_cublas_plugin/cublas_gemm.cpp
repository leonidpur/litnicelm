#include "cublas_gemm.hpp"

#include <cstdint>
#include <stdexcept>

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

struct CublasInputMatrix {
  const float *data;
  int leading_dim;
  long long int stride;
};

struct CublasOutputMatrix {
  float *data;
  int leading_dim;
  long long int stride;
};

void run_gemm(cublasHandle_t handle, cublasOperation_t op_a,
              cublasOperation_t op_b, const GemmShape &shape,
              const CublasInputMatrix &a, const CublasInputMatrix &b,
              const CublasOutputMatrix &out, const char *what) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasSgemm(handle, op_a, op_b, shape.out_cols, shape.out_rows,
                           shape.shared_dim, &alpha, a.data, a.leading_dim,
                           b.data, b.leading_dim, &beta, out.data,
                           out.leading_dim),
               what);
}

void run_strided_batched_gemm(cublasHandle_t handle, cublasOperation_t op_a,
                              cublasOperation_t op_b, const GemmShape &shape,
                              const CublasInputMatrix &a,
                              const CublasInputMatrix &b,
                              const CublasOutputMatrix &out, int batch_count,
                              const char *what) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasSgemmStridedBatched(
                   handle, op_a, op_b, shape.out_cols, shape.out_rows,
                   shape.shared_dim, &alpha, a.data, a.leading_dim, a.stride,
                   b.data, b.leading_dim, b.stride, &beta, out.data,
                   out.leading_dim, out.stride, batch_count),
               what);
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

bool regular_prefix_stride_f32(const TensorView &view, uint64_t prefix_count,
                               long long int &stride) {
  if (prefix_count <= 1) {
    stride = 0;
    return true;
  }
  if (view.rank() == 3) {
    const int64_t stride_bytes = view.stride_bytes(0);
    if (stride_bytes % static_cast<int64_t>(sizeof(float)) != 0) {
      return false;
    }
    stride = static_cast<long long int>(stride_bytes / sizeof(float));
    return true;
  }
  const int64_t first_offset =
      prefix_matrix_byte_offset(view, 1, 2) -
      prefix_matrix_byte_offset(view, 0, 2);
  if (first_offset % static_cast<int64_t>(sizeof(float)) != 0) {
    return false;
  }
  for (uint64_t prefix = 2; prefix < prefix_count; ++prefix) {
    const int64_t delta = prefix_matrix_byte_offset(view, prefix, 2) -
                          prefix_matrix_byte_offset(view, prefix - 1, 2);
    if (delta != first_offset) {
      return false;
    }
  }
  stride = static_cast<long long int>(first_offset / sizeof(float));
  return true;
}

bool all_contiguous_f32_row_major(const TensorView &a, const TensorView &b,
                                  const TensorView &out) {
  return is_cuda_f32_contiguous_row_major(a) &&
         is_cuda_f32_contiguous_row_major(b) &&
         is_cuda_f32_contiguous_row_major(out);
}

bool is_ranked_batch_gemm(const TensorView &a, const TensorView &b,
                          const TensorView &out) {
  return a.rank() >= 3 && b.rank() == a.rank() && out.rank() == a.rank();
}

bool is_contiguous_ranked_batch_gemm(const TensorView &a, const TensorView &b,
                                     const TensorView &out) {
  return is_ranked_batch_gemm(a, b, out) &&
         all_contiguous_f32_row_major(a, b, out);
}

bool is_contiguous_ranked_matmul_with_matrix_rhs(const TensorView &a,
                                                 const TensorView &b,
                                                 const TensorView &out) {
  return a.rank() >= 3 && b.rank() == 2 && out.rank() == a.rank() &&
         all_contiguous_f32_row_major(a, b, out);
}

bool is_contiguous_ranked_left_transposed_reduce(const TensorView &a,
                                                 const TensorView &b,
                                                 const TensorView &out) {
  return a.rank() >= 3 && b.rank() == a.rank() && out.rank() == 2 &&
         all_contiguous_f32_row_major(a, b, out);
}

bool is_contiguous_ranked_right_transposed_reduce(const TensorView &a,
                                                  const TensorView &b,
                                                  const TensorView &out) {
  return a.rank() >= 3 && b.rank() == 2 && out.rank() == 2 &&
         all_contiguous_f32_row_major(a, b, out);
}

} // namespace

void cublas_matmul(cublasHandle_t handle, const TensorView &a,
                   const TensorView &b, TensorView &out) {
    require_cuda_f32_row_major(a, "matmul(a)");
    require_cuda_f32_row_major(b, "matmul(b)");
    require_cuda_f32_row_major(out, "matmul(out)");
    if (is_contiguous_ranked_matmul_with_matrix_rhs(a, b, out)) {
      const GemmShape shape = matmul_ranked_matrix_rhs_shape(a, b);
      run_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
               {b.f32(), shape.out_cols, 0},
               {a.f32(), shape.shared_dim, 0}, {out.f32(), shape.out_cols, 0},
               "cublasSgemm(matmul ranked x matrix)");
      return;
    }
    if (is_contiguous_ranked_batch_gemm(a, b, out)) {
      const int batch_count = static_cast<int>(logical_prefix_count(a, 2));
      const GemmShape shape = matmul_ranked_batch_shape(a, b);
      const GemmStrides strides = matmul_contiguous_strides(shape);
      run_strided_batched_gemm(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
          {b.f32(), shape.out_cols, strides.b},
          {a.f32(), shape.shared_dim, strides.a},
          {out.f32(), shape.out_cols, strides.out}, batch_count,
          "cublasSgemmStridedBatched(matmul ranked)");
      return;
    }
    if (is_ranked_batch_gemm(a, b, out)) {
      if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
        throw std::runtime_error(
            "cuda_cublas_plugin: matmul would return without cublasSgemm for zero-sized ranked tensors");
      }
      const uint64_t prefix_count = logical_prefix_count(a, 2);
      const GemmShape shape = matmul_ranked_batch_shape(a, b);
      const int a_leading_dim = leading_dim_f32(a);
      const int b_leading_dim = leading_dim_f32(b);
      const int out_leading_dim = leading_dim_f32(out);
      long long int stride_a = 0;
      long long int stride_b = 0;
      long long int stride_out = 0;
      if (regular_prefix_stride_f32(a, prefix_count, stride_a) &&
          regular_prefix_stride_f32(b, prefix_count, stride_b) &&
          regular_prefix_stride_f32(out, prefix_count, stride_out)) {
        run_strided_batched_gemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
            {prefix_matrix_ptr(b, 0), b_leading_dim, stride_b},
            {prefix_matrix_ptr(a, 0), a_leading_dim, stride_a},
            {prefix_matrix_ptr(out, 0), out_leading_dim, stride_out},
            static_cast<int>(prefix_count),
            "cublasSgemmStridedBatched(matmul ranked non-contiguous)");
        return;
      }
      if (!fallback_to_prefix_loop) {
        throw std::runtime_error(
            "cuda_cublas_plugin: matmul irregular ranked layout is not supported");
      }
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        run_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
                 {prefix_matrix_ptr(b, prefix), b_leading_dim, 0},
                 {prefix_matrix_ptr(a, prefix), a_leading_dim, 0},
                 {prefix_matrix_ptr(out, prefix), out_leading_dim, 0},
                 "cublasSgemm(matmul ranked non-contiguous)");
      }
      return;
    }
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
      throw std::runtime_error(
          "cuda_cublas_plugin: matmul would return without cublasSgemm for zero-sized tensors");
    }
    const GemmShape shape = matmul_tensor_shape(a, b);
    // Row-major C = A * B is the same memory layout as column-major C^T = B^T * A^T.
    run_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, shape,
             {b.f32(), leading_dim_f32(b), 0},
             {a.f32(), leading_dim_f32(a), 0},
             {out.f32(), leading_dim_f32(out), 0}, "cublasSgemm(matmul)");
  }

void cublas_matmul_left_transposed(cublasHandle_t handle, const TensorView &a,
                                   const TensorView &b, TensorView &out) {
    require_cuda_f32_row_major(a, "matmul_left_transposed(a)");
    require_cuda_f32_row_major(b, "matmul_left_transposed(b)");
    require_cuda_f32_row_major(out, "matmul_left_transposed(out)");
    if (is_contiguous_ranked_left_transposed_reduce(a, b, out)) {
      const GemmShape shape = left_transposed_ranked_reduce_shape(a, b);
      run_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
               {b.f32(), shape.out_cols, 0}, {a.f32(), shape.out_rows, 0},
               {out.f32(), shape.out_cols, 0},
               "cublasSgemm(matmul_left_transposed ranked reduce)");
      return;
    }
    if (is_contiguous_ranked_batch_gemm(a, b, out)) {
      const int batch_count = static_cast<int>(logical_prefix_count(a, 2));
      const GemmShape shape = left_transposed_ranked_batch_shape(a, b);
      const GemmStrides strides = left_transposed_contiguous_strides(shape);
      run_strided_batched_gemm(
          handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
          {b.f32(), shape.out_cols, strides.b},
          {a.f32(), shape.out_rows, strides.a},
          {out.f32(), shape.out_cols, strides.out}, batch_count,
          "cublasSgemmStridedBatched(matmul_left_transposed ranked)");
      return;
    }
    if (is_ranked_batch_gemm(a, b, out)) {
      if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
        throw std::runtime_error(
            "cuda_cublas_plugin: matmul_left_transposed would return without cublasSgemm for zero-sized ranked tensors");
      }
      const uint64_t prefix_count = logical_prefix_count(a, 2);
      const GemmShape shape = left_transposed_ranked_batch_shape(a, b);
      const int a_leading_dim = leading_dim_f32(a);
      const int b_leading_dim = leading_dim_f32(b);
      const int out_leading_dim = leading_dim_f32(out);
      long long int stride_a = 0;
      long long int stride_b = 0;
      long long int stride_out = 0;
      if (regular_prefix_stride_f32(a, prefix_count, stride_a) &&
          regular_prefix_stride_f32(b, prefix_count, stride_b) &&
          regular_prefix_stride_f32(out, prefix_count, stride_out)) {
        run_strided_batched_gemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
            {prefix_matrix_ptr(b, 0), b_leading_dim, stride_b},
            {prefix_matrix_ptr(a, 0), a_leading_dim, stride_a},
            {prefix_matrix_ptr(out, 0), out_leading_dim, stride_out},
            static_cast<int>(prefix_count),
            "cublasSgemmStridedBatched(matmul_left_transposed ranked non-contiguous)");
        return;
      }
      if (!fallback_to_prefix_loop) {
        throw std::runtime_error(
            "cuda_cublas_plugin: matmul_left_transposed irregular ranked layout is not supported");
      }
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        run_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
                 {prefix_matrix_ptr(b, prefix), b_leading_dim, 0},
                 {prefix_matrix_ptr(a, prefix), a_leading_dim, 0},
                 {prefix_matrix_ptr(out, prefix), out_leading_dim, 0},
                 "cublasSgemm(matmul_left_transposed ranked non-contiguous)");
      }
      return;
    }
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_cols(b) == 0) {
      throw std::runtime_error(
          "cuda_cublas_plugin: matmul_left_transposed would return without cublasSgemm for zero-sized tensors");
    }
    const GemmShape shape = left_transposed_tensor_shape(a, b);
    run_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, shape,
             {b.f32(), leading_dim_f32(b), 0},
             {a.f32(), leading_dim_f32(a), 0},
             {out.f32(), leading_dim_f32(out), 0},
             "cublasSgemm(matmul_left_transposed)");
  }

void cublas_matmul_right_transposed(cublasHandle_t handle, const TensorView &a,
                                    const TensorView &b, TensorView &out) {
    require_cuda_f32_row_major(a, "matmul_right_transposed(a)");
    require_cuda_f32_row_major(b, "matmul_right_transposed(b)");
    require_cuda_f32_row_major(out, "matmul_right_transposed(out)");
    if (is_contiguous_ranked_right_transposed_reduce(a, b, out)) {
      throw std::runtime_error(
          "cuda_cublas_plugin: matmul_right_transposed ranked reduce would use custom kernel instead of cublasSgemm");
    }
    if (is_contiguous_ranked_matmul_with_matrix_rhs(a, b, out)) {
      const GemmShape shape = right_transposed_ranked_matrix_rhs_shape(a, b);
      run_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
               {b.f32(), shape.shared_dim, 0},
               {a.f32(), shape.shared_dim, 0}, {out.f32(), shape.out_cols, 0},
               "cublasSgemm(matmul_right_transposed ranked x matrix)");
      return;
    }
    if (is_contiguous_ranked_batch_gemm(a, b, out)) {
      const int batch_count = static_cast<int>(logical_prefix_count(a, 2));
      const GemmShape shape = right_transposed_ranked_batch_shape(a, b);
      const GemmStrides strides = right_transposed_contiguous_strides(shape);
      run_strided_batched_gemm(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
          {b.f32(), shape.shared_dim, strides.b},
          {a.f32(), shape.shared_dim, strides.a},
          {out.f32(), shape.out_cols, strides.out}, batch_count,
          "cublasSgemmStridedBatched(matmul_right_transposed ranked)");
      return;
    }
    if (is_ranked_batch_gemm(a, b, out)) {
      if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_rows(b) == 0) {
        throw std::runtime_error(
            "cuda_cublas_plugin: matmul_right_transposed would return without cublasSgemm for zero-sized ranked tensors");
      }
      const uint64_t prefix_count = logical_prefix_count(a, 2);
      const GemmShape shape = right_transposed_ranked_batch_shape(a, b);
      const int a_leading_dim = leading_dim_f32(a);
      const int b_leading_dim = leading_dim_f32(b);
      const int out_leading_dim = leading_dim_f32(out);
      long long int stride_a = 0;
      long long int stride_b = 0;
      long long int stride_out = 0;
      if (regular_prefix_stride_f32(a, prefix_count, stride_a) &&
          regular_prefix_stride_f32(b, prefix_count, stride_b) &&
          regular_prefix_stride_f32(out, prefix_count, stride_out)) {
        run_strided_batched_gemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
            {prefix_matrix_ptr(b, 0), b_leading_dim, stride_b},
            {prefix_matrix_ptr(a, 0), a_leading_dim, stride_a},
            {prefix_matrix_ptr(out, 0), out_leading_dim, stride_out},
            static_cast<int>(prefix_count),
            "cublasSgemmStridedBatched(matmul_right_transposed ranked non-contiguous)");
        return;
      }
      if (!fallback_to_prefix_loop) {
        throw std::runtime_error(
            "cuda_cublas_plugin: matmul_right_transposed irregular ranked layout is not supported");
      }
      for (uint64_t prefix = 0; prefix < prefix_count; ++prefix) {
        run_gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape,
                 {prefix_matrix_ptr(b, prefix), b_leading_dim, 0},
                 {prefix_matrix_ptr(a, prefix), a_leading_dim, 0},
                 {prefix_matrix_ptr(out, prefix), out_leading_dim, 0},
                 "cublasSgemm(matmul_right_transposed ranked non-contiguous)");
      }
      return;
    }
    if (tensor_rows(a) == 0 || tensor_cols(a) == 0 || tensor_rows(b) == 0) {
      throw std::runtime_error(
          "cuda_cublas_plugin: matmul_right_transposed would return without cublasSgemm for zero-sized tensors");
    }
    throw std::runtime_error(
        "cuda_cublas_plugin: matmul_right_transposed would use custom kernel instead of cublasSgemm");
  }

} // namespace cuda_cublas_plugin
