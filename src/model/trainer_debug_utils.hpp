// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

class TensorView;

namespace TrainerDebugUtils {

void observe_logits(const TensorView &logits);
void print_logit_distribution_table();

} // namespace TrainerDebugUtils
