#include "optimizer_adamw.hpp"

#include <stdexcept>

OptimizerAdamW::OptimizerAdamW(DeviceBackend &backend) : backend_(&backend) {
  if (backend_ == nullptr) {
    throw std::runtime_error("OptimizerAdamW: backend is null");
  }
}

void OptimizerAdamW::step(const TrainingConfig &tc, TensorView &params,
                          const TensorView &grads, TensorView &m, TensorView &v,
                          uint64_t step, bool apply_weight_decay) const {
  backend_->adamw_step(params, grads, m, v, step, tc.learning_rate, tc.beta1,
                       tc.beta2, tc.weight_decay, apply_weight_decay);
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
