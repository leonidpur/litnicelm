// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>
#include "i_output_head.hpp"
#include "ops.hpp"
#include "tensor_store.hpp"
#include "training_observer.hpp"

class TrainingDiagnosticsController;

// Inference output head for BF16 activations with FP32 logits: final
// LayerNorm in BF16, the result cast to FP32, then the LM-head GEMM in FP32
// with the FP32 parameter lm_head_w (not its BF16 compute copy). This keeps
// near-equal logits apart that BF16 would round to a tie or reorder. (Under
// fp32 the factory keeps the configured head: its logits are FP32.) Forward
// only.
class Fp32LogitsOutputHead final : public IOutputHead {
public:
  Fp32LogitsOutputHead(const Config &cfg, TensorStore &tensor_store,
                       const Ops &ops);

  void set_observer(ITrainingObserver *observer) override;
  void set_diagnostics(TrainingDiagnosticsController *) override {}
  // logits [B, S, V] F32; last_hidden (optional) BF16.
  void forward(const TensorView &x, TensorView &logits,
               TensorView *last_hidden) override;
  // Throws: inference only.
  void backward(const TensorView &dlogits, TensorView &dx) override;

private:
  const Config &cfg_;
  TensorStore &tensorStore_;
  const Ops &ops_;
  ITrainingObserver *observer_ = &default_training_observer();
};
