#pragma once

class TensorView;

namespace TrainerDebugUtils {

void observe_logits(const TensorView &logits);
void print_logit_distribution_table();

} // namespace TrainerDebugUtils
