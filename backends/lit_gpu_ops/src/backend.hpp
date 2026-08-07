#pragma once

#include "../../common/backend_interface.h"

BackendInterface *create_gpu_backend_impl();
void destroy_gpu_backend_impl(BackendInterface *backend);
