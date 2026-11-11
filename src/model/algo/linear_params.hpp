// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "tensor.hpp"

class GradientStore;
class TensorStore;

// The parameters of one affine map out = x W + b and their gradients,
// resolved once by the factory so that nodes do not look them up per call.
struct LinearParams {
  TensorView weight;          // [in, out], the master (optimizer) copy
  TensorView compute_weight;  // [in, out], what the GEMMs read
  TensorView bias;            // [1, out]
  TensorView dweight;         // empty without a gradient store (inference)
  TensorView dbias;
};

// Attention parameters of one layer: layer{i}.attn_qkv_{w,b} [D, 3D] /
// [1, 3D] and layer{i}.attn_out_{w,b} [D, D] / [1, D].
struct AttentionParams {
  LinearParams qkv;
  LinearParams out;
};

AttentionParams attention_params(int layer, const TensorStore &tensor_store,
                                 const GradientStore *gradient_store);

// FFN parameters of one layer: layer{i}.ffn_{w1,b1} [D, F] / [1, F] and
// layer{i}.ffn_{w2,b2} [F, D] / [1, D].
struct FfnParams {
  LinearParams w1;
  LinearParams w2;
};

FfnParams ffn_params(int layer, const TensorStore &tensor_store,
                     const GradientStore *gradient_store);
