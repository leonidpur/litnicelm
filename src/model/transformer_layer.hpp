#pragma once

#include <config.hpp>
#include "ffn.hpp"
#include "ops.hpp"
#include "self_attention.hpp"
#include "tensor_factory.hpp"

#include <functional>
#include <string>

// One GPT-style (decoder) transformer block:
//
//   y  = x + SelfAttention(LN1(x))
//   out = y + FFN(LN2(y))
//
// Parameters assumed from the named parameter layout (per layer i):
//   layer{i}.ln1_gamma  [1, D]
//   layer{i}.ln1_beta   [1, D]
//   layer{i}.attn_qkv_w [D, 3D]
//   layer{i}.attn_qkv_b [1, 3D]
//   layer{i}.attn_out_w [D, D]
//   layer{i}.attn_out_b [1, D]
//   layer{i}.ln2_gamma  [1, D]
//   layer{i}.ln2_beta   [1, D]
//   layer{i}.ffn_w1     [D, F]
//   layer{i}.ffn_b1     [1, F]
//   layer{i}.ffn_w2     [F, D]
//   layer{i}.ffn_b2     [1, D]
//
// NOTE: This layer is backend-agnostic and never checks CPU vs GPU.
// Backend/device checks belong in Ops / TensorFactory implementations.
class TransformerLayer {
public:
  using ParamUpdater =
      std::function<void(const std::string &, TensorView &, const TensorView &, bool)>;

  TransformerLayer(int layer_index, const Config &cfg, TensorFactory &tensors,
                   Ops &ops);

  // x:   [T, D]
  // out: [T, D]
  void forward(const TensorView &x, TensorView &out);
  void backward(const TensorView &dout, TensorView &dx,
                const ParamUpdater &update_param);

private:
  int idx_;
  const Config &cfg_;
  TensorFactory &tensorFactory_;
  Ops &ops_;

  SelfAttention attn_;
  FFN ffn_;
  TensorView cache_x_;
  TensorView cache_y_;
  TensorView cache_ln1_;
  TensorView cache_ln2_;
  bool has_cache_ = false;
};
