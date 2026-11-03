// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "../../common/backend_interface.h"

BackendInterface *create_gpu_backend_impl();
void destroy_gpu_backend_impl(BackendInterface *backend);
