#pragma once

#include "cuda_tensor_view.hpp"

#include <cublas_v2.h>

namespace cuda_cublas_plugin {

void cublas_matmul(cublasHandle_t handle, const TensorView &a,
                   const TensorView &b, TensorView &out);
void cublas_matmul_left_transposed(cublasHandle_t handle, const TensorView &a,
                                   const TensorView &b, TensorView &out);
void cublas_matmul_right_transposed(cublasHandle_t handle, const TensorView &a,
                                    const TensorView &b, TensorView &out);

} // namespace cuda_cublas_plugin
