enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# Build as a loadable MODULE.
add_library(lit_ops_gpu MODULE
    src/interface.cpp
    src/backend.cu
)

set_target_properties(lit_ops_gpu PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    OUTPUT_NAME "lit_gpu_ops"
    PREFIX ""
)

target_compile_features(lit_ops_gpu PRIVATE cxx_std_17)

# Use cuBLAS for your switchable strategy.
target_link_libraries(lit_ops_gpu PRIVATE CUDA::cublas)
