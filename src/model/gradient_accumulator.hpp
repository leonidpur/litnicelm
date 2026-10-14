#pragma once

#include "checkpoint.hpp"
#include "mean_loss.hpp"
#include "ops.hpp"
#include "tensor.hpp"

#include <cstdint>

// Sums the gradients of a step's batches and leaves their mean in the
// gradient arena for the optimizer. The backward pass overwrites the
// gradient arena, so the sum lives in a separate arena. With one batch per
// step there is no sum arena and the gradients pass through unchanged.
class GradientAccumulator {
public:
  // grads: the gradient arena; sum_arena: required only when
  // batches_per_step > 1.
  GradientAccumulator(const Ops &ops, uint32_t batches_per_step,
                      const TensorView &grads, const ArenaView &sum_arena);

  // Takes the batch's gradients (in the gradient arena) and loss;
  // first_of_step starts a new sum.
  void add_batch(bool first_of_step, double loss);
  // Leaves the mean gradient of the step's batches in the gradient arena and
  // returns their mean loss.
  double finish_step();

private:
  const Ops &ops_;
  const uint32_t batches_per_step_;
  TensorView grads_;
  TensorView sum_;
  uint32_t step_batches_ = 0;
  MeanLoss step_loss_;
};
