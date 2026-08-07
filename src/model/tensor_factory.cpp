#include "tensor_factory.hpp"

#include <utils/assert.hpp>

#include <algorithm>
#include <stdexcept>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("TensorFactory: ") + std::string(msg);                  \
  })

TensorFactory::TensorFactory(const Config &cfg, const NamedLayout &param_layout,
                             void *params_base, uint64_t params_bytes,
                             Device device, const NamedLayout &temp_layout,
                             void *temp_base, uint64_t temp_bytes)
    : cfg_(cfg), param_layout_(param_layout), temp_layout_(temp_layout),
      base_(reinterpret_cast<uint8_t *>(params_base)), bytes_(params_bytes),
      temp_base_(reinterpret_cast<uint8_t *>(temp_base)), temp_bytes_(temp_bytes),
      device_(device) {
  require(base_ != nullptr, "params_base is null");
  require(bytes_ > 0, "params_bytes must be > 0");
  require(param_layout_.total_bytes() <= bytes_,
          "param_layout.total_bytes() exceeds provided params_bytes");
  require(temp_base_ != nullptr, "temp_base is null");
  require(temp_bytes_ > 0, "temp_bytes must be > 0");
  require(temp_layout_.total_bytes() <= temp_bytes_,
          "temp_layout.total_bytes() exceeds provided temp_bytes");
}

void TensorFactory::check_layer(int layer) const {
  require(layer >= 0, "layer < 0");
  require(static_cast<uint32_t>(layer) < cfg_.model.n_layers,
          "layer out of range");
}

std::string TensorFactory::lname(int layer, const char *suffix) const {
  return "layer" + std::to_string(layer) + "." + suffix;
}

const LayoutSlice &TensorFactory::require_slice(const std::string &name) const {
  const LayoutSlice *s = param_layout_.find(name);
  require(s != nullptr, "missing slice: " + name);
  require(s->offset + s->bytes <= bytes_, "slice out of bounds: " + name);
  require((s->offset % alignof(float)) == 0,
          "slice offset not float-aligned: " + name);
  require((s->bytes % sizeof(float)) == 0,
          "slice bytes not multiple of float: " + name);
  return *s;
}

const LayoutSlice &TensorFactory::require_temp_slice(const std::string &name) const {
  const LayoutSlice *s = temp_layout_.find(name);
  require(s != nullptr, "missing temporary slice: " + name);
  require(s->offset + s->bytes <= temp_bytes_, "temporary slice out of bounds: " + name);
  return *s;
}

TensorView TensorFactory::make_view_f32(const LayoutSlice &s,
                                        Shape2D shape) const {
  const uint64_t expected = nbytes(shape, DType::F32);
  require(expected == s.bytes,
          "shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  void *ptr = base_ + s.offset;
  return TensorView(device_, DType::F32, ptr, shape);
}

TensorView TensorFactory::make_subview_f32(const LayoutSlice &s,
                                           Shape2D full_shape,
                                           int64_t col_offset,
                                           Shape2D sub_shape) const {
  require(col_offset >= 0, "negative col_offset for " + s.name);
  require(col_offset + sub_shape.c <= full_shape.c,
          "subview exceeds cols for " + s.name);
  require(sub_shape.r == full_shape.r,
          "subview must keep same rows for " + s.name);

  const uint64_t full_bytes = nbytes(full_shape, DType::F32);
  require(full_bytes == s.bytes, "full_shape mismatch for " + s.name);

  const int64_t elem = static_cast<int64_t>(dtype_size(DType::F32));
  const int64_t stride_c = elem;
  const int64_t stride_r = full_shape.c * elem;

  uint8_t *base_ptr = base_ + s.offset;
  void *sub_ptr = base_ptr + col_offset * elem;

  return TensorView(device_, DType::F32, sub_ptr, sub_shape, stride_c, stride_r);
}

TensorView TensorFactory::make_temp_view(const LayoutSlice &s, Shape2D shape) const {
  const uint64_t expected = nbytes(shape, s.dtype);
  require(expected == s.bytes,
          "temporary shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  void *ptr = temp_base_ + s.offset;
  return TensorView(device_, s.dtype, ptr, shape);
}

TensorView TensorFactory::by_name_f32(const std::string &name,
                                      Shape2D shape) const {
  return make_view_f32(require_slice(name), shape);
}

TensorView TensorFactory::temp_by_name(const std::string &name, Shape2D shape,
                                       DType dtype) const {
  require(dtype == DType::F32 || dtype == DType::I32,
          "temporary tensors currently support F32/I32 only");
  const LayoutSlice &s = require_temp_slice(name);
  require(s.dtype == dtype, "temporary dtype mismatch for " + name);
  return make_temp_view(s, shape);
}

TensorView TensorFactory::temp_by_name_subshape(const std::string &name,
                                                Shape2D full_shape,
                                                Shape2D shape,
                                                DType dtype) const {
  require(shape.r <= full_shape.r && shape.c <= full_shape.c,
          "temporary subshape exceeds full shape for " + name);
  require(dtype == DType::F32 || dtype == DType::I32,
          "temporary tensors currently support F32/I32 only");
  const LayoutSlice &s = require_temp_slice(name);
  require(s.dtype == dtype, "temporary dtype mismatch for " + name);
  const uint64_t full_bytes = nbytes(full_shape, dtype);
  require(full_bytes == s.bytes,
          "temporary full shape bytes mismatch for " + name + ": expected " +
              std::to_string(full_bytes) + " got " + std::to_string(s.bytes));
  const int64_t elem = static_cast<int64_t>(dtype_size(dtype));
  void *ptr = temp_base_ + s.offset;
  return TensorView(device_, dtype, ptr, shape, elem, full_shape.c * elem);
}

TensorView TensorFactory::temp_by_name_prefix(int layer, const char *suffix,
                                              Shape2D shape, DType dtype) const {
  check_layer(layer);
  return temp_by_name(lname(layer, suffix), shape, dtype);
}

TensorView TensorFactory::temp_by_name_prefix_subshape(int layer,
                                                       const char *suffix,
                                                       Shape2D full_shape,
                                                       Shape2D shape,
                                                       DType dtype) const {
  check_layer(layer);
  return temp_by_name_subshape(lname(layer, suffix), full_shape, shape, dtype);
}

int64_t TensorFactory::temp_token_rows() const {
  const uint64_t training_tokens =
      static_cast<uint64_t>(cfg_.training.batch_size) *
      static_cast<uint64_t>(cfg_.training.window_training);
  return static_cast<int64_t>(
      std::max<uint64_t>(training_tokens, cfg_.model.window_capacity));
}

TensorView TensorFactory::temp_ds_ids(int64_t rows) const {
  return temp_by_name_subshape("ds.ids", {temp_token_rows(), 1}, {rows, 1},
                               DType::I32);
}

TensorView TensorFactory::temp_ds_targets(int64_t rows) const {
  return temp_by_name_subshape("ds.targets", {temp_token_rows(), 1}, {rows, 1},
                               DType::I32);
}

TensorView TensorFactory::temp_infer_ids(int64_t rows) const {
  return temp_by_name_subshape("infer.ids", {cfg_.model.window_capacity, 1},
                               {rows, 1}, DType::I32);
}

TensorView TensorFactory::temp_infer_logits(int64_t rows) const {
  const int64_t V = static_cast<int64_t>(cfg_.model.target_vocab_size);
  return temp_by_name_subshape("infer.logits", {cfg_.model.window_capacity, V},
                               {rows, V});
}

TensorView TensorFactory::temp_tr_logits(int64_t rows) const {
  const int64_t V = static_cast<int64_t>(cfg_.model.target_vocab_size);
  return temp_by_name_subshape("tr.logits", {temp_token_rows(), V}, {rows, V});
}

TensorView TensorFactory::temp_tr_loss() const {
  return temp_by_name("tr.loss", {1, 1});
}

TensorView TensorFactory::temp_tr_X(int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_subshape("tr.X", {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_tr_Y(int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_subshape("tr.Y", {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_tr_Xn(int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_subshape("tr.Xn", {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_bw_XnT(int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_subshape("bw.XnT", {D, temp_token_rows()}, {D, rows});
}

TensorView TensorFactory::temp_bw_d_lm_w(int64_t rows) const {
  (void)rows;
  return temp_by_name("bw.d_lm_w",
                      {static_cast<int64_t>(cfg_.model.d_model),
                       static_cast<int64_t>(cfg_.model.target_vocab_size)});
}

TensorView TensorFactory::temp_bw_lm_wT() const {
  return temp_by_name("bw.lm_wT",
                      {static_cast<int64_t>(cfg_.model.target_vocab_size),
                       static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_bw_d_xn(int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_subshape("bw.d_xn", {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_bw_d_xlast(int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_subshape("bw.d_xlast", {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_bw_d_lnf_g() const {
  return temp_by_name("bw.d_lnf_g", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_bw_d_lnf_b() const {
  return temp_by_name("bw.d_lnf_b", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_bw_d_tok() const {
  return temp_by_name("bw.d_tok",
                      {static_cast<int64_t>(cfg_.model.target_vocab_size),
                       static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_bw_d_pos() const {
  return temp_by_name("bw.d_pos",
                      {static_cast<int64_t>(cfg_.model.window_capacity),
                       static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_layer_ln1(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "ln1", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_attn_out(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn_out",
                                      {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_layer_resid1(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "resid1", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_ln2(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "ln2", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_ffn_out(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "ffn_out",
                                      {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_layer_bw_d_prev(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "bw.d_prev",
                                      {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_layer_dln2(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "dln2", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_dy_ln2(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "dy_ln2", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_dln2_gamma(int layer) const {
  return temp_by_name_prefix(layer, "dln2_gamma", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_layer_dln2_beta(int layer) const {
  return temp_by_name_prefix(layer, "dln2_beta", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_layer_dy_total(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "dy_total",
                                      {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_layer_dln1(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "dln1", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_dx_ln1(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "dx_ln1", {temp_token_rows(), D},
                                      {rows, D});
}

TensorView TensorFactory::temp_layer_dln1_gamma(int layer) const {
  return temp_by_name_prefix(layer, "dln1_gamma", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_layer_dln1_beta(int layer) const {
  return temp_by_name_prefix(layer, "dln1_beta", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_attn_qkv(int layer, int64_t rows) const {
  const int64_t D3 = 3 * static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn.qkv",
                                      {temp_token_rows(), D3}, {rows, D3});
}

TensorView TensorFactory::temp_attn_context(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn.context",
                                      {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_attn_scores(int layer, int64_t rows) const {
  return temp_by_name_prefix_subshape(layer, "attn.scores",
                                      {temp_token_rows(), temp_token_rows()},
                                      {rows, rows});
}

TensorView TensorFactory::temp_attn_weights(int layer, int64_t rows) const {
  return temp_by_name_prefix_subshape(layer, "attn.weights",
                                      {temp_token_rows(), temp_token_rows()},
                                      {rows, rows});
}

TensorView TensorFactory::temp_attn_head(int layer, int64_t rows) const {
  const int64_t dh = static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads);
  return temp_by_name_prefix_subshape(layer, "attn.head",
                                      {temp_token_rows(), dh}, {rows, dh});
}

TensorView TensorFactory::temp_attn_contextT(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn.contextT",
                                      {D, temp_token_rows()}, {D, rows});
}

TensorView TensorFactory::temp_attn_dWo(int layer) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix(layer, "attn.dWo", {D, D});
}

TensorView TensorFactory::temp_attn_dbo(int layer) const {
  return temp_by_name_prefix(layer, "attn.dbo", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_attn_WoT(int layer) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix(layer, "attn.WoT", {D, D});
}

TensorView TensorFactory::temp_attn_dcontext(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn.dcontext",
                                      {temp_token_rows(), D}, {rows, D});
}

TensorView TensorFactory::temp_attn_dqkv(int layer, int64_t rows) const {
  const int64_t D3 = 3 * static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn.dqkv",
                                      {temp_token_rows(), D3}, {rows, D3});
}

TensorView TensorFactory::temp_attn_KhT(int layer, int64_t rows) const {
  const int64_t dh = static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads);
  return temp_by_name_prefix_subshape(layer, "attn.KhT",
                                      {dh, temp_token_rows()}, {dh, rows});
}

TensorView TensorFactory::temp_attn_VhT(int layer, int64_t rows) const {
  const int64_t dh = static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads);
  return temp_by_name_prefix_subshape(layer, "attn.VhT",
                                      {dh, temp_token_rows()}, {dh, rows});
}

TensorView TensorFactory::temp_attn_dweights(int layer, int64_t rows) const {
  return temp_by_name_prefix_subshape(layer, "attn.dweights",
                                      {temp_token_rows(), temp_token_rows()},
                                      {rows, rows});
}

TensorView TensorFactory::temp_attn_weightsT(int layer, int64_t rows) const {
  return temp_by_name_prefix_subshape(layer, "attn.weightsT",
                                      {temp_token_rows(), temp_token_rows()},
                                      {rows, rows});
}

TensorView TensorFactory::temp_attn_dscores(int layer, int64_t rows) const {
  return temp_by_name_prefix_subshape(layer, "attn.dscores",
                                      {temp_token_rows(), temp_token_rows()},
                                      {rows, rows});
}

TensorView TensorFactory::temp_attn_dscoresT(int layer, int64_t rows) const {
  return temp_by_name_prefix_subshape(layer, "attn.dscoresT",
                                      {temp_token_rows(), temp_token_rows()},
                                      {rows, rows});
}

TensorView TensorFactory::temp_attn_WqkvT(int layer) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix(layer, "attn.WqkvT", {3 * D, D});
}

TensorView TensorFactory::temp_attn_xT(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "attn.xT",
                                      {D, temp_token_rows()}, {D, rows});
}

TensorView TensorFactory::temp_attn_dWqkv(int layer) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix(layer, "attn.dWqkv", {D, 3 * D});
}

TensorView TensorFactory::temp_attn_dbqkv(int layer) const {
  return temp_by_name_prefix(layer, "attn.dbqkv",
                             {1, 3 * static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_ffn_h(int layer, int64_t rows) const {
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);
  return temp_by_name_prefix_subshape(layer, "ffn.h",
                                      {temp_token_rows(), F}, {rows, F});
}

TensorView TensorFactory::temp_ffn_a(int layer, int64_t rows) const {
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);
  return temp_by_name_prefix_subshape(layer, "ffn.a",
                                      {temp_token_rows(), F}, {rows, F});
}

TensorView TensorFactory::temp_ffn_aT(int layer, int64_t rows) const {
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);
  return temp_by_name_prefix_subshape(layer, "ffn.aT",
                                      {F, temp_token_rows()}, {F, rows});
}

TensorView TensorFactory::temp_ffn_dW2(int layer) const {
  return temp_by_name_prefix(layer, "ffn.dW2",
                             {static_cast<int64_t>(cfg_.model.d_ff),
                              static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_ffn_db2(int layer) const {
  return temp_by_name_prefix(layer, "ffn.db2", {1, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::temp_ffn_W2T(int layer) const {
  return temp_by_name_prefix(layer, "ffn.W2T",
                             {static_cast<int64_t>(cfg_.model.d_model),
                              static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorFactory::temp_ffn_da(int layer, int64_t rows) const {
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);
  return temp_by_name_prefix_subshape(layer, "ffn.da",
                                      {temp_token_rows(), F}, {rows, F});
}

TensorView TensorFactory::temp_ffn_dh(int layer, int64_t rows) const {
  const int64_t F = static_cast<int64_t>(cfg_.model.d_ff);
  return temp_by_name_prefix_subshape(layer, "ffn.dh",
                                      {temp_token_rows(), F}, {rows, F});
}

TensorView TensorFactory::temp_ffn_xT(int layer, int64_t rows) const {
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return temp_by_name_prefix_subshape(layer, "ffn.xT",
                                      {D, temp_token_rows()}, {D, rows});
}

TensorView TensorFactory::temp_ffn_dW1(int layer) const {
  return temp_by_name_prefix(layer, "ffn.dW1",
                             {static_cast<int64_t>(cfg_.model.d_model),
                              static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorFactory::temp_ffn_db1(int layer) const {
  return temp_by_name_prefix(layer, "ffn.db1", {1, static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorFactory::temp_ffn_W1T(int layer) const {
  return temp_by_name_prefix(layer, "ffn.W1T",
                             {static_cast<int64_t>(cfg_.model.d_ff),
                              static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorFactory::tok_embedding() const {
  const int64_t V = cfg_.model.target_vocab_size;
  const int64_t D = cfg_.model.d_model;
  return by_name_f32("tok_embedding", {V, D});
}

TensorView TensorFactory::pos_embedding() const {
  const int64_t S = cfg_.model.window_capacity;
  const int64_t D = cfg_.model.d_model;
  return by_name_f32("pos_embedding", {S, D});
}

TensorView TensorFactory::lnf_gamma() const {
  const int64_t D = cfg_.model.d_model;
  return by_name_f32("lnf_gamma", {1, D});
}

TensorView TensorFactory::lnf_beta() const {
  const int64_t D = cfg_.model.d_model;
  return by_name_f32("lnf_beta", {1, D});
}

TensorView TensorFactory::lm_head_w() const {
  const int64_t D = cfg_.model.d_model;
  const int64_t V = cfg_.model.target_vocab_size;
  return by_name_f32("lm_head_w", {D, V});
}

TensorView TensorFactory::layer_ln1_gamma(int layer) const {
  check_layer(layer);
  return by_name_f32(lname(layer, "ln1_gamma"), {1, (int64_t)cfg_.model.d_model});
}

TensorView TensorFactory::layer_ln1_beta(int layer) const {
  check_layer(layer);
  return by_name_f32(lname(layer, "ln1_beta"), {1, (int64_t)cfg_.model.d_model});
}

TensorView TensorFactory::layer_ln2_gamma(int layer) const {
  check_layer(layer);
  return by_name_f32(lname(layer, "ln2_gamma"), {1, (int64_t)cfg_.model.d_model});
}

TensorView TensorFactory::layer_ln2_beta(int layer) const {
  check_layer(layer);
  return by_name_f32(lname(layer, "ln2_beta"), {1, (int64_t)cfg_.model.d_model});
}

TensorView TensorFactory::layer_attn_qkv_w(int layer) const {
  check_layer(layer);
  const int64_t D = cfg_.model.d_model;
  return by_name_f32(lname(layer, "attn_qkv_w"), {D, 3 * D});
}

TensorView TensorFactory::layer_attn_qkv_b(int layer) const {
  check_layer(layer);
  const int64_t D = cfg_.model.d_model;
  return by_name_f32(lname(layer, "attn_qkv_b"), {1, 3 * D});
}

TensorView TensorFactory::layer_attn_out_w(int layer) const {
  check_layer(layer);
  const int64_t D = cfg_.model.d_model;
  return by_name_f32(lname(layer, "attn_out_w"), {D, D});
}

TensorView TensorFactory::layer_attn_out_b(int layer) const {
  check_layer(layer);
  const int64_t D = cfg_.model.d_model;
  return by_name_f32(lname(layer, "attn_out_b"), {1, D});
}

TensorView TensorFactory::layer_attn_wq(int layer) const {
  check_layer(layer);
  const auto &s = require_slice(lname(layer, "attn_qkv_w"));
  const int64_t D = cfg_.model.d_model;
  return make_subview_f32(s, {D, 3 * D}, 0, {D, D});
}

TensorView TensorFactory::layer_attn_wk(int layer) const {
  check_layer(layer);
  const auto &s = require_slice(lname(layer, "attn_qkv_w"));
  const int64_t D = cfg_.model.d_model;
  return make_subview_f32(s, {D, 3 * D}, D, {D, D});
}

TensorView TensorFactory::layer_attn_wv(int layer) const {
  check_layer(layer);
  const auto &s = require_slice(lname(layer, "attn_qkv_w"));
  const int64_t D = cfg_.model.d_model;
  return make_subview_f32(s, {D, 3 * D}, 2 * D, {D, D});
}

TensorView TensorFactory::layer_attn_bq(int layer) const {
  check_layer(layer);
  const auto &s = require_slice(lname(layer, "attn_qkv_b"));
  const int64_t D = cfg_.model.d_model;
  return make_subview_f32(s, {1, 3 * D}, 0, {1, D});
}

TensorView TensorFactory::layer_attn_bk(int layer) const {
  check_layer(layer);
  const auto &s = require_slice(lname(layer, "attn_qkv_b"));
  const int64_t D = cfg_.model.d_model;
  return make_subview_f32(s, {1, 3 * D}, D, {1, D});
}

TensorView TensorFactory::layer_attn_bv(int layer) const {
  check_layer(layer);
  const auto &s = require_slice(lname(layer, "attn_qkv_b"));
  const int64_t D = cfg_.model.d_model;
  return make_subview_f32(s, {1, 3 * D}, 2 * D, {1, D});
}

TensorView TensorFactory::layer_ffn_w1(int layer) const {
  check_layer(layer);
  const int64_t D = cfg_.model.d_model;
  const int64_t F = cfg_.model.d_ff;
  return by_name_f32(lname(layer, "ffn_w1"), {D, F});
}

TensorView TensorFactory::layer_ffn_b1(int layer) const {
  check_layer(layer);
  const int64_t F = cfg_.model.d_ff;
  return by_name_f32(lname(layer, "ffn_b1"), {1, F});
}

TensorView TensorFactory::layer_ffn_w2(int layer) const {
  check_layer(layer);
  const int64_t F = cfg_.model.d_ff;
  const int64_t D = cfg_.model.d_model;
  return by_name_f32(lname(layer, "ffn_w2"), {F, D});
}

TensorView TensorFactory::layer_ffn_b2(int layer) const {
  check_layer(layer);
  const int64_t D = cfg_.model.d_model;
  return by_name_f32(lname(layer, "ffn_b2"), {1, D});
}

TensorView TensorFactory::param_ffn_w1(int layer) const { return layer_ffn_w1(layer); }
TensorView TensorFactory::param_ffn_b1(int layer) const { return layer_ffn_b1(layer); }
TensorView TensorFactory::param_ffn_w2(int layer) const { return layer_ffn_w2(layer); }
TensorView TensorFactory::param_ffn_b2(int layer) const { return layer_ffn_b2(layer); }
TensorView TensorFactory::param_attn_qkv_w(int layer) const {
  return layer_attn_qkv_w(layer);
}
TensorView TensorFactory::param_attn_qkv_b(int layer) const {
  return layer_attn_qkv_b(layer);
}
TensorView TensorFactory::param_attn_out_w(int layer) const {
  return layer_attn_out_w(layer);
}
TensorView TensorFactory::param_attn_out_b(int layer) const {
  return layer_attn_out_b(layer);
}
TensorView TensorFactory::param_ln1_gamma(int layer) const {
  return layer_ln1_gamma(layer);
}
TensorView TensorFactory::param_ln1_beta(int layer) const {
  return layer_ln1_beta(layer);
}
TensorView TensorFactory::param_ln2_gamma(int layer) const {
  return layer_ln2_gamma(layer);
}
TensorView TensorFactory::param_ln2_beta(int layer) const {
  return layer_ln2_beta(layer);
}
TensorView TensorFactory::param_tok_embedding() const { return tok_embedding(); }
TensorView TensorFactory::param_pos_embedding() const { return pos_embedding(); }
TensorView TensorFactory::param_lnf_gamma() const { return lnf_gamma(); }
TensorView TensorFactory::param_lnf_beta() const { return lnf_beta(); }
TensorView TensorFactory::param_lm_head_w() const { return lm_head_w(); }

void TensorFactory::initialize_parameters_deterministic() const {
  require(device_ == Device::CPU, "initialize_parameters_deterministic CPU only");
  float *p = reinterpret_cast<float *>(base_);
  const uint64_t n = bytes_ / sizeof(float);
  for (uint64_t i = 0; i < n; ++i) {
    const int64_t centered =
        static_cast<int64_t>(i % 1024ULL) - static_cast<int64_t>(512);
    p[i] = static_cast<float>(centered) * 1e-4f;
  }
}

#undef require
