// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "cublaslt_linear.hpp"
#include "cuda_checks.hpp"

#include <litnicelm_utils/view_geometry.hpp>

#include <stdexcept>
#include <string>

namespace cuda_cublas_plugin {

namespace {

constexpr size_t kWorkspaceBytes = 4u << 20;

void check_lt(cublasStatus_t status, const char *what) {
  check_cublas(status, what);
}

cudaDataType_t cuda_type(DType dtype) {
  return dtype == DType::BF16 ? CUDA_R_16BF : CUDA_R_32F;
}

// Rows of a [..., C] view.
int64_t rows_of(const TensorView &view) {
  return static_cast<int64_t>(view.numel()) / view.dim(view.rank() - 1);
}

template <typename T>
void set_attr(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr,
              const T &value) {
  check_lt(cublasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)),
           "cublasLtMatmulDescSetAttribute");
}

} // namespace

CublasLtLinear::CublasLtLinear() {
  check_lt(cublasLtCreate(&handle_), "cublasLtCreate");
  check_cuda(cudaMalloc(&workspace_, kWorkspaceBytes),
             "cudaMalloc(cuBLASLt workspace)");
}

CublasLtLinear::~CublasLtLinear() {
  for (auto &entry : plans_) {
    Plan &p = entry.second;
    if (p.desc != nullptr) cublasLtMatmulDescDestroy(p.desc);
    if (p.a != nullptr) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b != nullptr) cublasLtMatrixLayoutDestroy(p.b);
    if (p.d != nullptr) cublasLtMatrixLayoutDestroy(p.d);
  }
  if (workspace_ != nullptr) cudaFree(workspace_);
  if (handle_ != nullptr) cublasLtDestroy(handle_);
}

CublasLtLinear::Plan &CublasLtLinear::plan(cublasOperation_t op_b, int64_t m,
                                           int64_t n, int64_t k,
                                           cudaDataType_t ab_type,
                                           cudaDataType_t d_type,
                                           cublasLtEpilogue_t epilogue) {
  const std::vector<int64_t> key{op_b, m, n, k, ab_type, d_type, epilogue};
  auto found = plans_.find(key);
  if (found != plans_.end()) {
    return found->second;
  }
  Plan &p = plans_[key];
  const cublasOperation_t op_a = CUBLAS_OP_N;
  check_lt(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
           "cublasLtMatmulDescCreate");
  set_attr(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, op_a);
  set_attr(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, op_b);
  set_attr(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
  const cudaDataType_t bias_type = CUDA_R_32F;
  set_attr(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_type);
  // A is m x k (column-major, ld m); B is k x n, stored n x k when
  // transposed; D is m x n.
  check_lt(cublasLtMatrixLayoutCreate(&p.a, ab_type, m, k, m), "layout A");
  if (op_b == CUBLAS_OP_N) {
    check_lt(cublasLtMatrixLayoutCreate(&p.b, ab_type, k, n, k), "layout B");
  } else {
    check_lt(cublasLtMatrixLayoutCreate(&p.b, ab_type, n, k, n), "layout B");
  }
  check_lt(cublasLtMatrixLayoutCreate(&p.d, d_type, m, n, m), "layout D");

  cublasLtMatmulPreference_t preference = nullptr;
  check_lt(cublasLtMatmulPreferenceCreate(&preference), "preference");
  const uint64_t workspace = kWorkspaceBytes;
  check_lt(cublasLtMatmulPreferenceSetAttribute(
               preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
               sizeof(workspace)),
           "preference workspace");
  // Split-K partials reduced in FP32 (or not split), as the cuBLAS path's
  // CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION.
  const uint32_t schemes =
      CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  check_lt(cublasLtMatmulPreferenceSetAttribute(
               preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
               sizeof(schemes)),
           "preference reduction scheme");
  cublasLtMatmulHeuristicResult_t result{};
  int found_count = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      handle_, p.desc, p.a, p.b, p.d, p.d, preference, 1, &result, &found_count);
  cublasLtMatmulPreferenceDestroy(preference);
  p.supported = status == CUBLAS_STATUS_SUCCESS && found_count > 0;
  if (p.supported) {
    p.algo = result.algo;
  }
  return p;
}

bool CublasLtLinear::try_forward(const TensorView &x, const TensorView &w,
                                 const TensorView &bias, bool relu,
                                 TensorView &out) {
  if (!x.is_contiguous() || !w.is_contiguous() || !out.is_contiguous() ||
      !bias.is_contiguous()) {
    return false;
  }
  // Row-major out [T, N] = x [T, K] w [K, N] is column-major
  // out^T (N x T) = w^T (N x K, ld N) x^T (K x T, ld K).
  const int64_t T = rows_of(x);
  const int64_t K = w.dim(0);
  const int64_t N = w.dim(1);
  Plan &p = plan(CUBLAS_OP_N, N, T, K, cuda_type(x.dtype()),
                 cuda_type(out.dtype()),
                 relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS);
  if (!p.supported) {
    return false;
  }
  const void *bias_ptr = bias.data();
  set_attr(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_lt(cublasLtMatmul(handle_, p.desc, &alpha, w.data(), p.a, x.data(), p.b,
                          &beta, out.data(), p.d, out.data(), p.d, &p.algo,
                          workspace_, kWorkspaceBytes, nullptr),
           "cublasLtMatmul (linear_forward)");
  return true;
}

bool CublasLtLinear::try_backward_params(const TensorView &x,
                                         const TensorView &dy, TensorView &dw,
                                         TensorView &dbias) {
  if (!x.is_contiguous() || !dy.is_contiguous() || !dw.is_contiguous() ||
      !dbias.is_contiguous()) {
    return false;
  }
  // Row-major dw [K, N] = x^T [K, T] dy [T, N] is column-major
  // dw^T (N x K) = dy^T (N x T, ld N) (x^T)^T, x stored K x T (ld K).
  // BGRADA sums A = dy^T over T: the bias gradient, length N.
  const int64_t T = rows_of(x);
  const int64_t K = dw.dim(0);
  const int64_t N = dw.dim(1);
  Plan &p = plan(CUBLAS_OP_T, N, K, T, cuda_type(x.dtype()),
                 cuda_type(dw.dtype()), CUBLASLT_EPILOGUE_BGRADA);
  if (!p.supported) {
    return false;
  }
  const void *dbias_ptr = dbias.data();
  set_attr(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, dbias_ptr);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_lt(cublasLtMatmul(handle_, p.desc, &alpha, dy.data(), p.a, x.data(),
                          p.b, &beta, dw.data(), p.d, dw.data(), p.d, &p.algo,
                          workspace_, kWorkspaceBytes, nullptr),
           "cublasLtMatmul (linear_backward_params)");
  return true;
}

} // namespace cuda_cublas_plugin
