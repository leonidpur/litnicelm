// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include "../../model/tensor.hpp"

#include <cstdint>
#include <memory>

namespace cuda_cublas_plugin {

// Causal scaled-dot-product attention through cuDNN's fused (flash) SDPA
// graphs on the legacy default stream: BF16 Q, K, V, O and their gradients,
// FP32 softmax statistics and accumulation. Q, K and V are read in place
// from the packed qkv rows (strided, no copies); dQ, dK and dV are written
// the same way into dqkv. Graphs are built on first use of a shape and kept;
// the workspace grows on demand. The contracts are BackendApiV1's
// causal_attention_forward / causal_attention_backward.
class CudnnAttention {
public:
  CudnnAttention();
  ~CudnnAttention();
  CudnnAttention(const CudnnAttention &) = delete;
  CudnnAttention &operator=(const CudnnAttention &) = delete;

  // Whether the plugin was built with cuDNN.
  static bool available();

  void forward(const TensorView &qkv, int64_t n_heads, float scale,
               TensorView &out, TensorView &stats);
  void backward(const TensorView &qkv, const TensorView &out,
                const TensorView &dout, const TensorView &stats,
                int64_t n_heads, float scale, TensorView &dqkv);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cuda_cublas_plugin
