#pragma once

#include <config.hpp>
#include "named_layout.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <string>

class TensorFactory {
public:
  TensorFactory(const Config &cfg, const NamedLayout &param_layout,
                void *params_base, uint64_t params_bytes, Device device,
                const NamedLayout &temp_layout, void *temp_base,
                uint64_t temp_bytes);

  TensorView tok_embedding() const;
  TensorView pos_embedding() const;
  TensorView lnf_gamma() const;
  TensorView lnf_beta() const;
  TensorView lm_head_w() const;

  TensorView layer_ln1_gamma(int layer) const;
  TensorView layer_ln1_beta(int layer) const;
  TensorView layer_ln2_gamma(int layer) const;
  TensorView layer_ln2_beta(int layer) const;

  TensorView layer_attn_qkv_w(int layer) const;
  TensorView layer_attn_qkv_b(int layer) const;
  TensorView layer_attn_out_w(int layer) const;
  TensorView layer_attn_out_b(int layer) const;

  TensorView layer_attn_wq(int layer) const;
  TensorView layer_attn_wk(int layer) const;
  TensorView layer_attn_wv(int layer) const;
  TensorView layer_attn_bq(int layer) const;
  TensorView layer_attn_bk(int layer) const;
  TensorView layer_attn_bv(int layer) const;

  TensorView layer_ffn_w1(int layer) const;
  TensorView layer_ffn_b1(int layer) const;
  TensorView layer_ffn_w2(int layer) const;
  TensorView layer_ffn_b2(int layer) const;
  TensorView param_ffn_w1(int layer) const;
  TensorView param_ffn_b1(int layer) const;
  TensorView param_ffn_w2(int layer) const;
  TensorView param_ffn_b2(int layer) const;
  TensorView param_attn_qkv_w(int layer) const;
  TensorView param_attn_qkv_b(int layer) const;
  TensorView param_attn_out_w(int layer) const;
  TensorView param_attn_out_b(int layer) const;
  TensorView param_ln1_gamma(int layer) const;
  TensorView param_ln1_beta(int layer) const;
  TensorView param_ln2_gamma(int layer) const;
  TensorView param_ln2_beta(int layer) const;
  TensorView param_tok_embedding() const;
  TensorView param_pos_embedding() const;
  TensorView param_lnf_gamma() const;
  TensorView param_lnf_beta() const;
  TensorView param_lm_head_w() const;

  TensorView temp_ds_ids(int64_t rows) const;
  TensorView temp_ds_targets(int64_t rows) const;
  TensorView temp_infer_ids(int64_t rows) const;
  TensorView temp_infer_logits(int64_t rows) const;
  TensorView temp_tr_logits(int64_t rows) const;
  TensorView temp_tr_loss() const;
  TensorView temp_tr_X(int64_t rows) const;
  TensorView temp_tr_Y(int64_t rows) const;
  TensorView temp_tr_Xn(int64_t rows) const;
  TensorView temp_bw_XnT(int64_t rows) const;
  TensorView temp_bw_d_lm_w(int64_t rows) const;
  TensorView temp_bw_lm_wT() const;
  TensorView temp_bw_d_xn(int64_t rows) const;
  TensorView temp_bw_d_xlast(int64_t rows) const;
  TensorView temp_bw_d_lnf_g() const;
  TensorView temp_bw_d_lnf_b() const;
  TensorView temp_bw_d_tok() const;
  TensorView temp_bw_d_pos() const;

  TensorView temp_layer_ln1(int layer, int64_t rows) const;
  TensorView temp_layer_attn_out(int layer, int64_t rows) const;
  TensorView temp_layer_resid1(int layer, int64_t rows) const;
  TensorView temp_layer_ln2(int layer, int64_t rows) const;
  TensorView temp_layer_ffn_out(int layer, int64_t rows) const;
  TensorView temp_layer_bw_d_prev(int layer, int64_t rows) const;
  TensorView temp_layer_dln2(int layer, int64_t rows) const;
  TensorView temp_layer_dy_ln2(int layer, int64_t rows) const;
  TensorView temp_layer_dln2_gamma(int layer) const;
  TensorView temp_layer_dln2_beta(int layer) const;
  TensorView temp_layer_dy_total(int layer, int64_t rows) const;
  TensorView temp_layer_dln1(int layer, int64_t rows) const;
  TensorView temp_layer_dx_ln1(int layer, int64_t rows) const;
  TensorView temp_layer_dln1_gamma(int layer) const;
  TensorView temp_layer_dln1_beta(int layer) const;

  TensorView temp_attn_qkv(int layer, int64_t rows) const;
  TensorView temp_attn_context(int layer, int64_t rows) const;
  TensorView temp_attn_scores(int layer, int64_t rows) const;
  TensorView temp_attn_weights(int layer, int64_t rows) const;
  TensorView temp_attn_head(int layer, int64_t rows) const;
  TensorView temp_attn_contextT(int layer, int64_t rows) const;
  TensorView temp_attn_dWo(int layer) const;
  TensorView temp_attn_dbo(int layer) const;
  TensorView temp_attn_WoT(int layer) const;
  TensorView temp_attn_dcontext(int layer, int64_t rows) const;
  TensorView temp_attn_dqkv(int layer, int64_t rows) const;
  TensorView temp_attn_KhT(int layer, int64_t rows) const;
  TensorView temp_attn_VhT(int layer, int64_t rows) const;
  TensorView temp_attn_dweights(int layer, int64_t rows) const;
  TensorView temp_attn_weightsT(int layer, int64_t rows) const;
  TensorView temp_attn_dscores(int layer, int64_t rows) const;
  TensorView temp_attn_dscoresT(int layer, int64_t rows) const;
  TensorView temp_attn_WqkvT(int layer) const;
  TensorView temp_attn_xT(int layer, int64_t rows) const;
  TensorView temp_attn_dWqkv(int layer) const;
  TensorView temp_attn_dbqkv(int layer) const;

  TensorView temp_ffn_h(int layer, int64_t rows) const;
  TensorView temp_ffn_a(int layer, int64_t rows) const;
  TensorView temp_ffn_aT(int layer, int64_t rows) const;
  TensorView temp_ffn_dW2(int layer) const;
  TensorView temp_ffn_db2(int layer) const;
  TensorView temp_ffn_W2T(int layer) const;
  TensorView temp_ffn_da(int layer, int64_t rows) const;
  TensorView temp_ffn_dh(int layer, int64_t rows) const;
  TensorView temp_ffn_xT(int layer, int64_t rows) const;
  TensorView temp_ffn_dW1(int layer) const;
  TensorView temp_ffn_db1(int layer) const;
  TensorView temp_ffn_W1T(int layer) const;

  TensorView by_name_f32(const std::string &name, Shape2D shape) const;
  void initialize_parameters_deterministic() const;

private:
  const Config &cfg_;
  const NamedLayout &param_layout_;
  const NamedLayout &temp_layout_;
  uint8_t *base_ = nullptr;
  uint64_t bytes_ = 0;
  uint8_t *temp_base_ = nullptr;
  uint64_t temp_bytes_ = 0;
  Device device_ = Device::CPU;

  void check_layer(int layer) const;
  const LayoutSlice &require_slice(const std::string &name) const;
  const LayoutSlice &require_temp_slice(const std::string &name) const;

  TensorView make_view_f32(const LayoutSlice &s, Shape2D shape) const;
  TensorView make_subview_f32(const LayoutSlice &s, Shape2D full_shape,
                              int64_t col_offset, Shape2D sub_shape) const;
  TensorView make_temp_view(const LayoutSlice &s, Shape2D shape) const;
  TensorView temp_by_name(const std::string &name, Shape2D shape,
                          DType dtype = DType::F32) const;
  TensorView temp_by_name_subshape(const std::string &name, Shape2D full_shape,
                                   Shape2D shape,
                                   DType dtype = DType::F32) const;
  TensorView temp_by_name_prefix(int layer, const char *suffix, Shape2D shape,
                                 DType dtype = DType::F32) const;
  TensorView temp_by_name_prefix_subshape(int layer, const char *suffix,
                                          Shape2D full_shape, Shape2D shape,
                                          DType dtype = DType::F32) const;
  int64_t temp_token_rows() const;

  std::string lname(int layer, const char *suffix) const;
};
