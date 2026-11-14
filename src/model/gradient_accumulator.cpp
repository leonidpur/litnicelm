// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "gradient_accumulator.hpp"

#include <stdexcept>

GradientAccumulator::GradientAccumulator(
    const Ops &ops, uint32_t batches_per_step,
    const std::optional<TensorView> &decay_grads,
    const std::optional<TensorView> &no_decay_grads, const ArenaView &sum_arena)
    : ops_(ops), batches_per_step_(batches_per_step), decay_grads_(decay_grads),
      no_decay_grads_(no_decay_grads) {
  if (batches_per_step_ > 1 && no_decay_grads_) {
    if (sum_arena.base == nullptr || sum_arena.bytes < no_decay_grads_->bytes()) {
      throw std::runtime_error("GradientAccumulator: sum arena is missing or too small");
    }
    no_decay_sum_ = TensorView(sum_arena.device, no_decay_grads_->dtype(),
                               sum_arena.base, no_decay_grads_->shape());
  }
}

void GradientAccumulator::begin_batch(bool first_of_step) {
  first_of_step_ = first_of_step;
  if (first_of_step && decay_grads_) {
    ops_.fill(*decay_grads_, 0.0f);
  }
  if (no_decay_grads_) {
    ops_.fill(*no_decay_grads_, 0.0f);
  }
}

void GradientAccumulator::add_batch(double loss) {
  if (first_of_step_) {
    step_batches_ = 0;
    step_loss_ = MeanLoss();
  }
  if (batches_per_step_ > 1 && no_decay_grads_) {
    if (first_of_step_) {
      ops_.copy(*no_decay_grads_, no_decay_sum_);
    } else {
      ops_.add_inplace(no_decay_sum_, *no_decay_grads_);
    }
  }
  step_batches_ += 1;
  step_loss_.add(loss);
}

double GradientAccumulator::finish_step() {
  if (batches_per_step_ > 1) {
    const float scale = 1.0f / static_cast<float>(step_batches_);
    if (decay_grads_) {
      ops_.mul_scalar(*decay_grads_, scale, *decay_grads_);
    }
    if (no_decay_grads_) {
      ops_.mul_scalar(no_decay_sum_, scale, *no_decay_grads_);
    }
  }
  return step_loss_.mean();
}
