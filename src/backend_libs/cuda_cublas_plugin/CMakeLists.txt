include(CheckLanguage)

check_language(CUDA)
if (NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler not found. Skipping litnice_backend_cublas_plugin target.")
  return()
endif()

enable_language(CUDA)

find_package(CUDAToolkit QUIET)
if (NOT CUDAToolkit_FOUND)
  message(STATUS "CUDA toolkit not found. Skipping litnice_backend_cublas_plugin target.")
  return()
endif()

add_library(litnice_backend_cublas_plugin SHARED
    ../../model/backend/device_backend.cpp
    cuda_cublas_plugin.cpp
    cuda_backend.cpp
    cublas_gemm.cpp
    cuda_tensor_view.cpp
    cuda_elementwise_kernels.cu
    cuda_reduction_kernels.cu
    cuda_softmax_kernels.cu
    cuda_loss_kernels.cu
    cuda_layernorm_kernels.cu
    cuda_embedding_kernels.cu
    cuda_optimizer_kernels.cu
)

target_include_directories(litnice_backend_cublas_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/model
    ${CMAKE_SOURCE_DIR}/src/model/memory
    ${CMAKE_SOURCE_DIR}/src/config
    ${CMAKE_SOURCE_DIR}/src/utils
)

target_compile_features(litnice_backend_cublas_plugin PRIVATE cxx_std_17)

target_link_libraries(litnice_backend_cublas_plugin PRIVATE
    CUDA::cudart
    CUDA::cublas
    dl
)

set_target_properties(litnice_backend_cublas_plugin PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
