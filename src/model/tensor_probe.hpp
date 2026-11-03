// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

#include <string>

// Probe lines for --probe: a tensor's statistics (mean, std, min, max, hash),
// logged as Probe events. CPU F32 tensors only; others are skipped.
namespace TensorProbe {

void log_tensor(const std::string &group, const std::string &name,
                const TensorView &tensor);
void log_loss(const TensorView &loss_scalar, const TensorView &logits,
              const TensorView &targets);

// "[d0,d1,...]"
std::string format_shape(const Shape &shape);

} // namespace TensorProbe
