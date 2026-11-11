// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#include "linear_params.hpp"

#include "gradient_store.hpp"
#include "tensor_store.hpp"

namespace {

LinearParams linear_params(const TensorView &weight,
                           const TensorView &compute_weight,
                           const TensorView &bias,
                           const GradientStore *gradient_store) {
  LinearParams out{weight, compute_weight, bias, TensorView{}, TensorView{}};
  if (gradient_store != nullptr) {
    out.dweight = gradient_store->grad_for_param(weight);
    out.dbias = gradient_store->grad_for_param(bias);
  }
  return out;
}

} // namespace

AttentionParams attention_params(int layer, const TensorStore &tensor_store,
                                 const GradientStore *gradient_store) {
  return AttentionParams{
      linear_params(tensor_store.param_attn_qkv_w(layer),
                    tensor_store.compute_attn_qkv_w(layer),
                    tensor_store.param_attn_qkv_b(layer), gradient_store),
      linear_params(tensor_store.param_attn_out_w(layer),
                    tensor_store.compute_attn_out_w(layer),
                    tensor_store.param_attn_out_b(layer), gradient_store)};
}

FfnParams ffn_params(int layer, const TensorStore &tensor_store,
                     const GradientStore *gradient_store) {
  return FfnParams{
      linear_params(tensor_store.param_ffn_w1(layer),
                    tensor_store.compute_ffn_w1(layer),
                    tensor_store.param_ffn_b1(layer), gradient_store),
      linear_params(tensor_store.param_ffn_w2(layer),
                    tensor_store.compute_ffn_w2(layer),
                    tensor_store.param_ffn_b2(layer), gradient_store)};
}
