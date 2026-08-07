#pragma once

#include "backend/device_backend.hpp"

class OptimizerAdamW {
public:
  explicit OptimizerAdamW(DeviceBackend &backend);

  void step(const TrainingConfig &tc, TensorView &params, const TensorView &grads,
            TensorView &m, TensorView &v, uint64_t step,
            bool apply_weight_decay) const;
  void step(uint64_t step, float learning_rate, float beta1, float beta2,
            float weight_decay, float grad_clip) const;
  void zero_state() const;

private:
  DeviceBackend *backend_ = nullptr;
};
