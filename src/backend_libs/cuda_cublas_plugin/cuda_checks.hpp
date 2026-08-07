#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_cublas_plugin {

inline const char *cublas_status_string(cublasStatus_t status) {
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

inline void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " failed: " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, const char *what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuda_cublas_plugin: ") + what +
                             " failed: " + cublas_status_string(status));
  }
}

inline void check_kernel_launch(const char *what) {
  check_cuda(cudaPeekAtLastError(), what);
}

} // namespace cuda_cublas_plugin
