// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

class DeviceBackend;
class TensorStore;
struct Config;

// Casts every GEMM parameter into its compute weight (F32 -> the compute
// dtype). Only for stores with a compute-weight arena: without one the
// compute weights are the parameters.
void cast_parameters_to_compute_weights(const Config &cfg, const TensorStore &store,
                                        DeviceBackend &backend);
