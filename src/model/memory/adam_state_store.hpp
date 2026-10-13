#pragma once

#include <config.hpp>
#include "checkpoint.hpp"
#include "named_layout.hpp"
#include <litnicelm_utils/tensor.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class AdamStateStore {
public:
  struct StatePair {
    TensorView m;
    TensorView v;
    bool apply_weight_decay = true;
  };

  struct LayerParamViews {
    StatePair ln1_gamma;
    StatePair ln1_beta;
    StatePair attn_qkv_w;
    StatePair attn_qkv_b;
    StatePair attn_out_w;
    StatePair attn_out_b;
    StatePair ln2_gamma;
    StatePair ln2_beta;
    StatePair ffn_w1;
    StatePair ffn_b1;
    StatePair ffn_w2;
    StatePair ffn_b2;
  };

  // optimizer_layout: the parameter shapes in the optimizer-state dtype;
  // the AdamW arena holds m, then v, each laid out by it.
  AdamStateStore(const Config &cfg, const ModelAlgoFactory &algo,
                 const NamedLayout &param_layout,
                 const NamedLayout &optimizer_layout,
                 void *params_base, uint64_t params_bytes,
                 const AdamStateView &adam_state);

  // Bytes of the AdamW arena: m and v, each laid out by optimizer_layout.
  static uint64_t arena_bytes(const NamedLayout &optimizer_layout) {
    return optimizer_layout.total_bytes() * 2;
  }

  // m or v of one decay zone as a flat view; empty if the zone is.
  std::optional<TensorView> m_zone_view(DecayZone zone) const;
  std::optional<TensorView> v_zone_view(DecayZone zone) const;

  const StatePair &state_for_param(const TensorView &param) const;

  const StatePair &param_tok_embedding() const;
  const StatePair &param_pos_embedding() const;
  const StatePair &param_lnf_gamma() const;
  const StatePair &param_lnf_beta() const;
  const StatePair &param_lm_head_w() const;

  const StatePair &param_ffn_w1(int layer) const;
  const StatePair &param_ffn_b1(int layer) const;
  const StatePair &param_ffn_w2(int layer) const;
  const StatePair &param_ffn_b2(int layer) const;
  const StatePair &param_attn_qkv_w(int layer) const;
  const StatePair &param_attn_qkv_b(int layer) const;
  const StatePair &param_attn_out_w(int layer) const;
  const StatePair &param_attn_out_b(int layer) const;
  const StatePair &param_ln1_gamma(int layer) const;
  const StatePair &param_ln1_beta(int layer) const;
  const StatePair &param_ln2_gamma(int layer) const;
  const StatePair &param_ln2_beta(int layer) const;

private:
  const Config &cfg_;
  uint8_t *params_base_ = nullptr;
  uint64_t params_bytes_ = 0;
  uint8_t *adam_base_ = nullptr;
  uint64_t adam_bytes_ = 0;
  uint64_t param_bytes_ = 0;
  Device device_ = Device::CPU;
  NamedLayout optimizer_layout_;
  std::unordered_map<std::string, LayoutSlice> state_slice_by_name_;
  // Byte offset of v in the AdamW arena (m starts at 0).
  uint64_t v_offset_ = 0;

  StatePair tok_embedding_;
  StatePair pos_embedding_;
  StatePair lnf_gamma_;
  StatePair lnf_beta_;
  StatePair lm_head_w_;
  std::vector<LayerParamViews> layer_param_views_;
  std::unordered_map<const void *, StatePair> state_by_param_data_;

  void check_layer(int layer) const;
  void build_state_views(const NamedLayout &param_layout,
                         const ModelAlgoFactory &algo);
  void register_state(const TensorView &param, const StatePair &state);
  TensorView make_param_view(const LayoutSlice &s, Shape shape) const;
  // The m or v view (at state_base_offset) of the parameter in slice s,
  // looked up by name in the optimizer layout.
  TensorView make_state_view(const LayoutSlice &s, Shape shape,
                             uint64_t state_base_offset) const;
  StatePair make_state_pair(const LayoutSlice &s, Shape shape,
                            bool apply_weight_decay) const;
  std::string lname(int layer, const char *suffix) const;
};
