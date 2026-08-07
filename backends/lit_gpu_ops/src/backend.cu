#include "backend.hpp"

#include <cublas_v2.h>

#include <iostream>
#include <stdexcept>

namespace {

class GPUBackend : public BackendInterface {
public:
    GPUBackend() {
        const cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
    }

    ~GPUBackend() override {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    void forward(int *tokens, int batch_size) override {
        std::cout << "[GPU] Running BF16 Forward Path with cuBLAS handle "
                  << handle_ << " on batch size " << batch_size << std::endl;
        (void)tokens;
    }

    void train_step(int *tokens, int batch_size) override {
        std::cout << "[GPU] Running BF16 Train Step (Optimizer on GPU) with cuBLAS handle "
                  << handle_ << " on batch size " << batch_size << std::endl;
        (void)tokens;
    }

private:
    cublasHandle_t handle_ = nullptr;
};

}  // namespace

BackendInterface *create_gpu_backend_impl() {
    return new GPUBackend();
}

void destroy_gpu_backend_impl(BackendInterface *backend) {
    delete backend;
}
