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
    ../../model/backend/backend_observer.cpp
    ../../model/backend/observed_backend.cpp
    cuda_cublas_plugin.cpp
    cuda_backend.cpp
    cublas_gemm.cpp
    cublaslt_linear.cpp
    cuda_tensor_view.cpp
    cudnn_attention.cpp
    cuda_cast_kernels.cu
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
    litnicelm_utils
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    dl
)

# cuDNN (optional): the fused causal attention op. Without it the plugin
# reports no kFusedOpCausalAttentionBf16Act. LITNICELM_CUDNN_ROOT holds
# include/cudnn.h and lib/libcudnn.so; the cudnn-frontend headers are
# searched in LITNICELM_CUDNN_FRONTEND_INCLUDE.
set(LITNICELM_CUDNN_ROOT "$ENV{HOME}/.local/opt/cudnn-9.28.1-cu12" CACHE PATH
    "cuDNN 9 for CUDA 12 (include/, lib/)")
set(LITNICELM_CUDNN_FRONTEND_INCLUDE "$ENV{HOME}/.local/opt/cudnn-frontend/include"
    CACHE PATH "cudnn-frontend 1.x headers")
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${LITNICELM_CUDNN_ROOT}/include NO_DEFAULT_PATH)
find_path(CUDNN_FRONTEND_INCLUDE_DIR cudnn_frontend.h
          HINTS ${LITNICELM_CUDNN_FRONTEND_INCLUDE} NO_DEFAULT_PATH)
find_library(CUDNN_LIBRARY cudnn HINTS ${LITNICELM_CUDNN_ROOT}/lib NO_DEFAULT_PATH)
if (CUDNN_INCLUDE_DIR AND CUDNN_FRONTEND_INCLUDE_DIR AND CUDNN_LIBRARY)
  message(STATUS "cuDNN: ${CUDNN_LIBRARY} (fused causal attention enabled)")
  target_compile_definitions(litnice_backend_cublas_plugin PRIVATE LITNICELM_WITH_CUDNN)
  target_include_directories(litnice_backend_cublas_plugin SYSTEM PRIVATE
      ${CUDNN_INCLUDE_DIR} ${CUDNN_FRONTEND_INCLUDE_DIR})
  target_link_libraries(litnice_backend_cublas_plugin PRIVATE ${CUDNN_LIBRARY})
  get_filename_component(CUDNN_LIBRARY_DIR ${CUDNN_LIBRARY} DIRECTORY)
  set_target_properties(litnice_backend_cublas_plugin PROPERTIES
      BUILD_RPATH ${CUDNN_LIBRARY_DIR} INSTALL_RPATH ${CUDNN_LIBRARY_DIR})
else()
  message(STATUS "cuDNN not found (LITNICELM_CUDNN_ROOT=${LITNICELM_CUDNN_ROOT}); "
                 "fused causal attention disabled")
endif()

set_target_properties(litnice_backend_cublas_plugin PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
