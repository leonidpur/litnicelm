#pragma once

class DeviceBackend;
class TensorStore;
struct Config;

// Casts every GEMM parameter into its compute weight (F32 -> the compute
// dtype). Only for stores with a compute-weight arena: without one the
// compute weights are the parameters.
void cast_parameters_to_compute_weights(const Config &cfg, const TensorStore &store,
                                        DeviceBackend &backend);
