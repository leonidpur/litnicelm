#pragma once

#include "optimizer_adamw_cpu.hpp"

class OptimizerAdamW {
public:
  explicit OptimizerAdamW(Device device = Device::CPU);

  void step(const TrainingConfig &tc, TensorView &params, const TensorView &grads,
            TensorView &m, TensorView &v, uint64_t step,
            bool apply_weight_decay) const;
  void step(uint64_t step, float learning_rate, float beta1, float beta2,
            float weight_decay, float grad_clip) const;
  void zero_state() const;

private:
  Device device_ = Device::CPU;
  OptimizerAdamWCPU cpu_;
};
