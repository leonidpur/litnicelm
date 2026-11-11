// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "cuda_tensor_view.hpp"

#include <cublasLt.h>

#include <cstdint>
#include <map>
#include <vector>

namespace cuda_cublas_plugin {

// Linear layers through cuBLASLt with the bias, bias+ReLU or bias-gradient
// epilogue on the legacy default stream (the contracts are BackendApiV1's
// linear_forward and linear_backward_params). FP32 accumulation and
// split-K reduction; operands F32 (_f32) or BF16 with F32 bias, weight
// gradient and bias gradient (_bf16act). A plan (descriptors and the
// heuristic's first algorithm) is cached per shape. try_* return false,
// having done nothing, when an operand is not contiguous or cuBLASLt has
// no algorithm for the call; the caller then runs the unfused ops. With
// CUDA 12.2's cuBLASLt that is every _bf16act forward: there is no BF16
// output with an F32 bias (the bias-gradient epilogue does run).
class CublasLtLinear {
public:
  CublasLtLinear();
  ~CublasLtLinear();
  CublasLtLinear(const CublasLtLinear &) = delete;
  CublasLtLinear &operator=(const CublasLtLinear &) = delete;

  bool try_forward(const TensorView &x, const TensorView &w,
                   const TensorView &bias, bool relu, TensorView &out);
  bool try_backward_params(const TensorView &x, const TensorView &dy,
                           TensorView &dw, TensorView &dbias);

private:
  struct Plan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool supported = false;
  };

  // m, n, k of the column-major D (m x n) = op(A) (m x k) op(B) (k x n).
  Plan &plan(cublasOperation_t op_b, int64_t m, int64_t n, int64_t k,
             cudaDataType_t ab_type, cudaDataType_t d_type,
             cublasLtEpilogue_t epilogue);

  cublasLtHandle_t handle_ = nullptr;
  void *workspace_ = nullptr;
  std::map<std::vector<int64_t>, Plan> plans_;
};

} // namespace cuda_cublas_plugin
