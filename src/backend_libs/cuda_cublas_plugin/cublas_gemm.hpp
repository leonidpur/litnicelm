// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "cuda_tensor_view.hpp"

#include <cublas_v2.h>

namespace cuda_cublas_plugin {

void cublas_gemm(cublasHandle_t handle, const TensorView &a,
                 const TensorView &b, TensorView &out);
void cublas_gemm_ranked_matrix_rhs(cublasHandle_t handle, const TensorView &a,
                                   const TensorView &b, TensorView &out);
void cublas_gemm_ranked_matrix_rhs_t(cublasHandle_t handle, const TensorView &a,
                                     const TensorView &b, TensorView &out);
void cublas_gemm_ranked_reduce_lhs_t(cublasHandle_t handle,
                                     const TensorView &a,
                                     const TensorView &b, TensorView &out);
// out (F32) += a^T b.
void cublas_gemm_ranked_reduce_lhs_t_accumulate(cublasHandle_t handle,
                                                const TensorView &a,
                                                const TensorView &b,
                                                TensorView &out);
void cublas_gemm_batched(cublasHandle_t handle, const TensorView &a,
                         const TensorView &b, TensorView &out);
void cublas_gemm_batched_lhs_t(cublasHandle_t handle, const TensorView &a,
                               const TensorView &b, TensorView &out);
void cublas_gemm_batched_rhs_t(cublasHandle_t handle, const TensorView &a,
                               const TensorView &b, TensorView &out);

} // namespace cuda_cublas_plugin
