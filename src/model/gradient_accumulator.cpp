#include "gradient_accumulator.hpp"

#include <stdexcept>

GradientAccumulator::GradientAccumulator(const Ops &ops, uint32_t batches_per_step,
                                         const TensorView &grads,
                                         const ArenaView &sum_arena)
    : ops_(ops), batches_per_step_(batches_per_step), grads_(grads) {
  if (batches_per_step_ > 1) {
    if (sum_arena.base == nullptr || sum_arena.bytes < grads_.bytes()) {
      throw std::runtime_error("GradientAccumulator: sum arena is missing or too small");
    }
    sum_ = TensorView(sum_arena.device, DType::F32, sum_arena.base, grads_.shape());
  }
}

void GradientAccumulator::add_batch(bool first_of_step, double loss) {
  if (first_of_step) {
    step_batches_ = 0;
    step_loss_ = MeanLoss();
  }
  if (batches_per_step_ > 1) {
    if (first_of_step) {
      ops_.copy(grads_, sum_);
    } else {
      ops_.add_inplace(sum_, grads_);
    }
  }
  step_batches_ += 1;
  step_loss_.add(loss);
}

double GradientAccumulator::finish_step() {
  if (batches_per_step_ > 1) {
    ops_.mul_scalar(sum_, 1.0f / static_cast<float>(step_batches_), grads_);
  }
  return step_loss_.mean();
}
