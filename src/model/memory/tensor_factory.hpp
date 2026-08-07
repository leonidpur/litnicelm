#pragma once

#include <config.hpp>
#include "named_layout.hpp"
#include "backend/device_backend.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <string>

class TensorFactory {
public:
  enum class TempLayoutKind {
    Training,
    Inference,
  };

  struct LayerParamViews {
    TensorView ln1_gamma;
    TensorView ln1_beta;
    TensorView attn_qkv_w;
    TensorView attn_qkv_b;
    TensorView attn_out_w;
    TensorView attn_out_b;
    TensorView ln2_gamma;
    TensorView ln2_beta;
    TensorView ffn_w1;
    TensorView ffn_b1;
    TensorView ffn_w2;
    TensorView ffn_b2;
  };

  struct LayerTempViews {
    TensorView ln1;
    TensorView bw_d_prev;
    TensorView attn_out;
    TensorView resid1;
    TensorView ln2;
    TensorView ffn_out;
    TensorView hidden;

    TensorView attn_qkv;
    TensorView attn_context;
    TensorView attn_scores;
    TensorView attn_weights;
    TensorView attn_weights_cache;
    TensorView attn_head;

    TensorView ffn_h;
    TensorView ffn_a;

    TensorView dln2;
    TensorView dy_ln2;
    TensorView dy_total;
    TensorView dln1;
    TensorView dx_ln1;

    TensorView attn_contextT;
    TensorView attn_WoT;
    TensorView attn_dcontext;
    TensorView attn_dqkv;
    TensorView attn_KhT;
    TensorView attn_VhT;
    TensorView attn_dweights;
    TensorView attn_weightsT;
    TensorView attn_dscores;
    TensorView attn_dscoresT;
    TensorView attn_WqkvT;
    TensorView attn_xT;

    TensorView ffn_aT;
    TensorView ffn_W2T;
    TensorView ffn_da;
    TensorView ffn_dh;
    TensorView ffn_xT;
    TensorView ffn_W1T;
  };

  TensorFactory(const Config &cfg, const NamedLayout &param_layout,
                void *params_base, uint64_t params_bytes, Device device,
                const NamedLayout &temp_layout, void *temp_base,
                uint64_t temp_bytes, TempLayoutKind temp_kind);

  const TensorView &tok_embedding() const;
  const TensorView &pos_embedding() const;
  const TensorView &lnf_gamma() const;
  const TensorView &lnf_beta() const;
  const TensorView &lm_head_w() const;

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
  const TensorView &param_ffn_w1(int layer) const;
  const TensorView &param_ffn_b1(int layer) const;
  const TensorView &param_ffn_w2(int layer) const;
  const TensorView &param_ffn_b2(int layer) const;
  const TensorView &param_attn_qkv_w(int layer) const;
  const TensorView &param_attn_qkv_b(int layer) const;
  const TensorView &param_attn_out_w(int layer) const;
  const TensorView &param_attn_out_b(int layer) const;
  const TensorView &param_ln1_gamma(int layer) const;
  const TensorView &param_ln1_beta(int layer) const;
  const TensorView &param_ln2_gamma(int layer) const;
  const TensorView &param_ln2_beta(int layer) const;
  const TensorView &param_tok_embedding() const;
  const TensorView &param_pos_embedding() const;
  const TensorView &param_lnf_gamma() const;
  const TensorView &param_lnf_beta() const;
  const TensorView &param_lm_head_w() const;

  TensorView temp_ds_ids() const;
  TensorView temp_ds_ids(int64_t rows) const;
  TensorView temp_ds_ids(int64_t batch_size, int64_t seq_len) const;
  TensorView temp_ds_targets() const;
  TensorView temp_ds_targets(int64_t rows) const;
  TensorView temp_ds_targets(int64_t batch_size, int64_t seq_len) const;
  bool has_inference_io_temps() const;
  TensorView temp_infer_ids(int64_t rows) const;
  TensorView temp_infer_logits(int64_t rows) const;
  TensorView temp_tr_logits(int64_t rows) const;
  TensorView temp_tr_logits(int64_t batch_size, int64_t seq_len) const;
  TensorView temp_tr_loss() const;
  TensorView temp_tr_X(int64_t rows) const;
  TensorView temp_tr_X(int64_t batch_size, int64_t seq_len) const;
  TensorView temp_tr_Y(int64_t rows) const;
  TensorView temp_tr_Y(int64_t batch_size, int64_t seq_len) const;
  TensorView temp_tr_Xn(int64_t rows) const;
  TensorView temp_tr_Xn(int64_t batch_size, int64_t seq_len) const;
  TensorView temp_bw_XnT(int64_t rows) const;
  TensorView temp_bw_lm_wT() const;
  TensorView temp_bw_d_xn(int64_t rows) const;
  TensorView temp_bw_d_xn(int64_t batch_size, int64_t seq_len) const;
  TensorView temp_bw_d_xlast(int64_t rows) const;
  TensorView temp_bw_d_xlast(int64_t batch_size, int64_t seq_len) const;

  TensorView temp_layer_ln1(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_layer_attn_out(int layer, int64_t batch_size,
                                 int64_t seq_len) const;
  TensorView temp_layer_resid1(int layer, int64_t batch_size,
                               int64_t seq_len) const;
  TensorView temp_layer_ln2(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_layer_ffn_out(int layer, int64_t batch_size,
                                int64_t seq_len) const;
  TensorView temp_layer_hidden(int layer, int64_t batch_size,
                               int64_t seq_len) const;
  TensorView temp_layer_bw_d_prev(int layer, int64_t rows) const;
  TensorView temp_layer_bw_d_prev(int layer, int64_t batch_size,
                                  int64_t seq_len) const;
  TensorView temp_layer_dln2(int layer, int64_t rows) const;
  TensorView temp_layer_dln2(int layer, int64_t batch_size,
                             int64_t seq_len) const;
  TensorView temp_layer_dy_ln2(int layer, int64_t rows) const;
  TensorView temp_layer_dy_ln2(int layer, int64_t batch_size,
                               int64_t seq_len) const;
  TensorView temp_layer_dy_total(int layer, int64_t rows) const;
  TensorView temp_layer_dy_total(int layer, int64_t batch_size,
                                 int64_t seq_len) const;
  TensorView temp_layer_dln1(int layer, int64_t rows) const;
  TensorView temp_layer_dln1(int layer, int64_t batch_size,
                             int64_t seq_len) const;
  TensorView temp_layer_dx_ln1(int layer, int64_t rows) const;
  TensorView temp_layer_dx_ln1(int layer, int64_t batch_size,
                               int64_t seq_len) const;

  TensorView temp_attn_qkv(int layer, int64_t rows) const;
  TensorView temp_attn_qkv(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_attn_context(int layer, int64_t rows) const;
  TensorView temp_attn_context(int layer, int64_t batch_size,
                               int64_t seq_len) const;
  TensorView temp_attn_scores(int layer, int64_t rows) const;
  TensorView temp_attn_scores(int layer, int64_t batch_size,
                              int64_t seq_len) const;
  TensorView temp_attn_weights(int layer, int64_t rows) const;
  TensorView temp_attn_weights(int layer, int64_t batch_size,
                               int64_t seq_len) const;
  TensorView temp_attn_cached_weights(int layer, int64_t rows) const;
  TensorView temp_attn_cached_weights(int layer, int64_t batch_size,
                                      int64_t seq_len) const;
  TensorView temp_attn_head(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_attn_contextT(int layer, int64_t rows) const;
  TensorView temp_attn_WoT(int layer) const;
  TensorView temp_attn_dcontext(int layer, int64_t rows) const;
  TensorView temp_attn_dcontext(int layer, int64_t batch_size,
                                int64_t seq_len) const;
  TensorView temp_attn_dqkv(int layer, int64_t rows) const;
  TensorView temp_attn_dqkv(int layer, int64_t batch_size,
                            int64_t seq_len) const;
  TensorView temp_attn_KhT(int layer, int64_t rows) const;
  TensorView temp_attn_VhT(int layer, int64_t rows) const;
  TensorView temp_attn_dweights(int layer, int64_t rows) const;
  TensorView temp_attn_dweights(int layer, int64_t batch_size,
                                int64_t seq_len) const;
  TensorView temp_attn_weightsT(int layer, int64_t rows) const;
  TensorView temp_attn_dscores(int layer, int64_t rows) const;
  TensorView temp_attn_dscores(int layer, int64_t batch_size,
                               int64_t seq_len) const;
  TensorView temp_attn_dscoresT(int layer, int64_t rows) const;
  TensorView temp_attn_WqkvT(int layer) const;
  TensorView temp_attn_xT(int layer, int64_t rows) const;

  TensorView temp_ffn_h(int layer, int64_t rows) const;
  TensorView temp_ffn_h(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_ffn_a(int layer, int64_t rows) const;
  TensorView temp_ffn_a(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_ffn_aT(int layer, int64_t rows) const;
  TensorView temp_ffn_W2T(int layer) const;
  TensorView temp_ffn_da(int layer, int64_t rows) const;
  TensorView temp_ffn_da(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_ffn_dh(int layer, int64_t rows) const;
  TensorView temp_ffn_dh(int layer, int64_t batch_size, int64_t seq_len) const;
  TensorView temp_ffn_xT(int layer, int64_t rows) const;
  TensorView temp_ffn_W1T(int layer) const;

  void initialize_parameters_deterministic(DeviceBackend &device_backend) const;

private:
  const Config &cfg_;
  uint8_t *base_ = nullptr;
  uint64_t bytes_ = 0;
  uint8_t *temp_base_ = nullptr;
  uint64_t temp_bytes_ = 0;
  Device device_ = Device::CPU;
  TempLayoutKind temp_kind_ = TempLayoutKind::Training;
  TensorView tok_embedding_;
  TensorView pos_embedding_;
  TensorView lnf_gamma_;
  TensorView lnf_beta_;
  TensorView lm_head_w_;
  std::vector<LayerParamViews> layer_param_views_;
  TensorView ds_ids_;
  TensorView ds_targets_;
  TensorView infer_ids_;
  TensorView infer_logits_;
  TensorView tr_logits_;
  TensorView tr_loss_;
  TensorView tr_X_;
  TensorView tr_Y_;
  TensorView tr_Xn_;
  TensorView bw_XnT_;
  TensorView bw_lm_wT_;
  TensorView bw_d_xn_;
  TensorView bw_d_xlast_;
  std::vector<LayerTempViews> layer_temp_views_;

  void check_layer(int layer) const;
  void build_param_views(const NamedLayout &param_layout);
  void build_training_temp_views(const NamedLayout &temp_layout);
  void build_inference_temp_views(const NamedLayout &temp_layout);

  TensorView make_view_f32(const LayoutSlice &s, Shape shape) const;
  TensorView make_subview_f32(const TensorView &view, int64_t col_offset,
                              Shape sub_shape) const;
  TensorView make_temp_view(const LayoutSlice &s, Shape shape) const;
  TensorView prefix_storage(const TensorView &view, Shape shape) const;
  TensorView prefix_batch_seq(const TensorView &view, int64_t batch_size,
                              int64_t seq_len, const char *label,
                              int64_t expected_last_dim = -1) const;
  TensorView prefix_batch_seq_square(const TensorView &view,
                                     int64_t batch_size, int64_t seq_len,
                                     const char *label) const;
  TensorView prefix_head_batch_seq_square(const TensorView &view,
                                          int64_t batch_size,
                                          int64_t seq_len,
                                          const char *label) const;
  int64_t temp_batch_tokens() const;

  std::string lname(int layer, const char *suffix) const;
  std::string infer_lname(int layer, const char *suffix) const;
};
