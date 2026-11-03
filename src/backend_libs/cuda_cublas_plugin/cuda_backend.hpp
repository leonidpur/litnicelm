// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "../../model/backend/device_backend.hpp"

#include <cstdint>

namespace cuda_cublas_plugin {

DeviceBackend *create_backend(uint32_t device);
DeviceBackend &backend_from_opaque(void *backend);

} // namespace cuda_cublas_plugin
