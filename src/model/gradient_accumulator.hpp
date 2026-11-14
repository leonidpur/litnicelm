// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "checkpoint.hpp"
#include "mean_loss.hpp"
#include "ops.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <optional>

// Zeroes the gradient arena and turns the gradients of a step's batches
// into their mean for the optimizer. The backward pass adds the weight
// gradients (the decay zone) to the arena, so they sum in place over the
// step; it overwrites the bias and LayerNorm gradients (the no-decay zone),
// so those sum in a separate arena of the no-decay zone's size. With one
// batch per step there is no sum arena and the gradients pass through
// unchanged.
class GradientAccumulator {
public:
  // decay_grads, no_decay_grads: the gradient arena's zones (absent when
  // empty); sum_arena: required only when batches_per_step > 1 and the
  // no-decay zone is not empty.
  GradientAccumulator(const Ops &ops, uint32_t batches_per_step,
                      const std::optional<TensorView> &decay_grads,
                      const std::optional<TensorView> &no_decay_grads,
                      const ArenaView &sum_arena);

  // Before the batch's backward pass: zeroes the decay zone at the start of
  // a step and the no-decay zone before every batch.
  void begin_batch(bool first_of_step);
  // Takes the batch's gradients (in the gradient arena) and loss.
  void add_batch(double loss);
  // Leaves the mean gradient of the step's batches in the gradient arena and
  // returns their mean loss.
  double finish_step();

private:
  const Ops &ops_;
  const uint32_t batches_per_step_;
  std::optional<TensorView> decay_grads_;
  std::optional<TensorView> no_decay_grads_;
  TensorView no_decay_sum_;
  bool first_of_step_ = true;
  uint32_t step_batches_ = 0;
  MeanLoss step_loss_;
};
