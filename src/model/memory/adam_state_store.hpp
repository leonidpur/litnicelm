#pragma once

#include <config.hpp>
#include "checkpoint.hpp"
#include "named_layout.hpp"
#include "tensor.hpp"

#include <cstdint>
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

  AdamStateStore(const Config &cfg, const NamedLayout &param_layout,
                   void *params_base, uint64_t params_bytes,
                   const AdamStateView &adam_state);

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

  StatePair tok_embedding_;
  StatePair pos_embedding_;
  StatePair lnf_gamma_;
  StatePair lnf_beta_;
  StatePair lm_head_w_;
  std::vector<LayerParamViews> layer_param_views_;
  std::unordered_map<const void *, StatePair> state_by_param_data_;

  void check_layer(int layer) const;
  void build_state_views(const NamedLayout &param_layout);
  void register_state(const TensorView &param, const StatePair &state);
  TensorView make_param_view_f32(const LayoutSlice &s, Shape shape) const;
  TensorView make_state_view_f32(const LayoutSlice &s, Shape shape,
                                 uint64_t state_base_offset) const;
  StatePair make_state_pair_f32(const LayoutSlice &s, Shape shape,
                                bool apply_weight_decay) const;
  std::string lname(int layer, const char *suffix) const;
};
