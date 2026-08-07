#include "../../common/backend_interface.h"
#include "backend.hpp"

#include <exception>
#include <iostream>

extern "C" {
BackendInterface *create_backend() {
    try {
        return create_gpu_backend_impl();
    } catch (const std::exception &ex) {
        std::cerr << "[GPU] Failed to create backend: " << ex.what() << std::endl;
        return nullptr;
    }
}

void destroy_backend(BackendInterface *backend) {
    destroy_gpu_backend_impl(backend);
}
}
