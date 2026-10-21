#include "compute_weight_sync.hpp"

#include "backend/device_backend.hpp"
#include "tensor_store.hpp"

#include <config.hpp>

void cast_parameters_to_compute_weights(const Config &cfg, const TensorStore &store,
                                        DeviceBackend &backend) {
  auto cast = [&](const TensorView &param, TensorView compute) {
    backend.cast(param, compute);
  };
  const int layers = static_cast<int>(cfg.model.n_layers);
  for (int l = 0; l < layers; ++l) {
    cast(store.param_attn_qkv_w(l), store.compute_attn_qkv_w(l));
    cast(store.param_attn_out_w(l), store.compute_attn_out_w(l));
    cast(store.param_ffn_w1(l), store.compute_ffn_w1(l));
    cast(store.param_ffn_w2(l), store.compute_ffn_w2(l));
  }
  cast(store.param_lm_head_w(), store.compute_lm_head_w());
}
