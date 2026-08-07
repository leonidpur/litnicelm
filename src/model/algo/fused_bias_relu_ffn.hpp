#pragma once

#include "ffn.hpp"

class FusedBiasReluFFN final : public FFN {
public:
  using FFN::FFN;

  void forward(const TensorView &x, TensorView &out) override;
  void backward(const TensorView &dout, TensorView &dx) override;
};
