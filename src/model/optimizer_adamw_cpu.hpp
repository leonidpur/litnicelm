#pragma once

#include <litnicelm_utils/tensor.hpp>
#include <config.hpp>

#include <cstdint>

class OptimizerAdamWCPU {
public:
  OptimizerAdamWCPU() = default;

  void step(const TrainingConfig &tc, TensorView &params, const TensorView &grads,
            TensorView &m, TensorView &v, uint64_t step,
            bool apply_weight_decay) const;
};
