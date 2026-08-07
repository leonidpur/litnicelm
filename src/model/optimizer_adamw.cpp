#include "optimizer_adamw.hpp"

#include <stdexcept>

OptimizerAdamW::OptimizerAdamW(Device device) : device_(device) {
  if (device_ != Device::CPU) {
    throw std::runtime_error("OptimizerAdamW: GPU backend not implemented");
  }
}

void OptimizerAdamW::step(const TrainingConfig &tc, TensorView &params,
                          const TensorView &grads, TensorView &m, TensorView &v,
                          uint64_t step, bool apply_weight_decay) const {
  cpu_.step(tc, params, grads, m, v, step, apply_weight_decay);
}

void OptimizerAdamW::step(uint64_t step, float learning_rate, float beta1,
                          float beta2, float weight_decay,
                          float grad_clip) const {
  (void)step;
  (void)learning_rate;
  (void)beta1;
  (void)beta2;
  (void)weight_decay;
  (void)grad_clip;
  throw std::runtime_error(
      "OptimizerAdamW::step(step, lr, beta1, beta2, wd, grad_clip) is a "
      "stub. Call step(tc, params, grads, m, v, step, apply_weight_decay).");
}

void OptimizerAdamW::zero_state() const {}
