#include "tensor_store.hpp"

#include <utils/assert.hpp>

#include <algorithm>
#include <stdexcept>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("TensorStore: ") + std::string(msg);                   \
  })

namespace {
bool uses_inplace_ffn_activation(const Config &cfg) {
  return cfg.model_algo.ffn == "inplace_fused_bias_relu";
}

bool uses_fused_inplace_attention(const Config &cfg) {
  return cfg.model_algo.attention == "fused_inplace" ||
         cfg.model_algo.attention == "fused_inplace_multistream";
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

class LayoutCursor {
public:
  LayoutCursor(const std::vector<LayoutSlice> &slices, const char *label)
      : slices_(slices), label_(label) {}

  const LayoutSlice &next(const std::string &expected) {
    require(index_ < slices_.size(),
            std::string(label_) + " layout ended before slice " + expected);
    const LayoutSlice &slice = slices_[index_++];
    require(slice.name == expected,
            std::string(label_) + " layout mismatch: expected " + expected +
                " got " + slice.name);
    return slice;
  }

  void finish() const {
    require(index_ == slices_.size(),
            std::string(label_) + " layout has unexpected extra slices");
  }

private:
  const std::vector<LayoutSlice> &slices_;
  const char *label_;
  size_t index_ = 0;
};

} // namespace

class TensorStore::TempTensorSubStore {
public:
  TempTensorSubStore(const Config &cfg, uint8_t *temp_base,
                       uint64_t temp_bytes, Device device)
      : cfg_(cfg), temp_base_(temp_base), temp_bytes_(temp_bytes),
        device_(device) {}
  virtual ~TempTensorSubStore() = default;

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
  TensorView temp_attn_qkv(int layer, int64_t batch_size,
                           int64_t seq_len) const;
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
  TensorView temp_attn_head(int layer, int64_t batch_size,
                            int64_t seq_len) const;
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

protected:
  const Config &cfg_;
  uint8_t *temp_base_ = nullptr;
  uint64_t temp_bytes_ = 0;
  Device device_ = Device::CPU;

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
  std::vector<TensorStore::LayerTempViews> layer_temp_views_;

  void check_layer(int layer) const;
  TensorView make_temp_view(const LayoutSlice &s, Shape shape) const;
  TensorView prefix_storage(const TensorView &view, Shape shape) const;
  virtual TensorView prefix_batch_seq(const TensorView &view,
                                      int64_t batch_size, int64_t seq_len,
                                      const char *label,
                                      int64_t expected_last_dim = -1) const = 0;
  virtual TensorView prefix_batch_seq_square(const TensorView &view,
                                             int64_t batch_size,
                                             int64_t seq_len,
                                             const char *label) const = 0;
  virtual TensorView prefix_head_batch_seq_square(
      const TensorView &view, int64_t batch_size, int64_t seq_len,
      const char *label) const = 0;
  int64_t temp_batch_tokens() const;
  std::string lname(int layer, const char *suffix) const;
  std::string infer_lname(int layer, const char *suffix) const;
};

class TrainingTempTensorSubStore final
    : public TensorStore::TempTensorSubStore {
public:
  TrainingTempTensorSubStore(const Config &cfg, const NamedLayout &temp_layout,
                               uint8_t *temp_base, uint64_t temp_bytes,
                               Device device);

private:
  TensorView prefix_batch_seq(const TensorView &view, int64_t batch_size,
                              int64_t seq_len, const char *label,
                              int64_t expected_last_dim) const override;
  TensorView prefix_batch_seq_square(const TensorView &view,
                                     int64_t batch_size, int64_t seq_len,
                                     const char *label) const override;
  TensorView prefix_head_batch_seq_square(const TensorView &view,
                                          int64_t batch_size,
                                          int64_t seq_len,
                                          const char *label) const override;
};

class InferenceTempTensorSubStore final
    : public TensorStore::TempTensorSubStore {
public:
  InferenceTempTensorSubStore(const Config &cfg,
                                const NamedLayout &temp_layout,
                                uint8_t *temp_base, uint64_t temp_bytes,
                                Device device);

private:
  TensorView prefix_batch_seq(const TensorView &view, int64_t batch_size,
                              int64_t seq_len, const char *label,
                              int64_t expected_last_dim) const override;
  TensorView prefix_batch_seq_square(const TensorView &view,
                                     int64_t batch_size, int64_t seq_len,
                                     const char *label) const override;
  TensorView prefix_head_batch_seq_square(const TensorView &view,
                                          int64_t batch_size,
                                          int64_t seq_len,
                                          const char *label) const override;
};

TensorStore::~TensorStore() = default;

TensorStore::TensorStore(const Config &cfg, const NamedLayout &param_layout,
                             void *params_base, uint64_t params_bytes,
                             Device device, const NamedLayout &temp_layout,
                             void *temp_base, uint64_t temp_bytes,
                             TempLayoutKind temp_kind)
    : cfg_(cfg), base_(reinterpret_cast<uint8_t *>(params_base)),
      bytes_(params_bytes), device_(device) {
  require(base_ != nullptr, "params_base is null");
  require(bytes_ > 0, "params_bytes must be > 0");
  require(param_layout.total_bytes() <= bytes_,
          "param_layout.total_bytes() exceeds provided params_bytes");
  auto *temp_base_bytes = reinterpret_cast<uint8_t *>(temp_base);
  require(temp_base_bytes != nullptr, "temp_base is null");
  require(temp_bytes > 0, "temp_bytes must be > 0");
  require(temp_layout.total_bytes() <= temp_bytes,
          "temp_layout.total_bytes() exceeds provided temp_bytes");
  build_param_views(param_layout);
  if (temp_kind == TempLayoutKind::Training) {
    temp_tensor_substore_ = std::make_unique<TrainingTempTensorSubStore>(
        cfg_, temp_layout, temp_base_bytes, temp_bytes, device_);
  } else {
    temp_tensor_substore_ = std::make_unique<InferenceTempTensorSubStore>(
        cfg_, temp_layout, temp_base_bytes, temp_bytes, device_);
  }
}

void TensorStore::check_layer(int layer) const {
  require(layer >= 0, "layer < 0");
  require(static_cast<uint32_t>(layer) < cfg_.model.n_layers,
          "layer out of range");
}

std::string TensorStore::lname(int layer, const char *suffix) const {
  return "layer" + std::to_string(layer) + "." + suffix;
}

void TensorStore::TempTensorSubStore::check_layer(int layer) const {
  require(layer >= 0, "layer < 0");
  require(static_cast<uint32_t>(layer) < cfg_.model.n_layers,
          "layer out of range");
}

std::string TensorStore::TempTensorSubStore::lname(int layer,
                                                       const char *suffix) const {
  return "layer" + std::to_string(layer) + "." + suffix;
}

std::string TensorStore::TempTensorSubStore::infer_lname(
    int layer, const char *suffix) const {
  return "infer." + lname(layer, suffix);
}

void TensorStore::build_param_views(const NamedLayout &param_layout) {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t ffn_dim = static_cast<int64_t>(cfg_.model.d_ff);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t qkv_dim = 3 * model_dim;

  LayoutCursor cursor(param_layout.slices(), "parameter");
  tok_embedding_ = make_view_f32(cursor.next("tok_embedding"),
                                 {vocab_size, model_dim});
  pos_embedding_ = make_view_f32(
      cursor.next("pos_embedding"),
      {static_cast<int64_t>(cfg_.model.max_seq_len), model_dim});
  for (uint32_t layer = 0; layer < cfg_.model.n_layers; ++layer) {
    LayerParamViews views;
    views.ln1_gamma =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ln1_gamma")),
                      {1, model_dim});
    views.ln1_beta =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ln1_beta")),
                      {1, model_dim});
    views.attn_qkv_w =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "attn_qkv_w")),
                      {model_dim, qkv_dim});
    views.attn_qkv_b =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "attn_qkv_b")),
                      {1, qkv_dim});
    views.attn_out_w =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "attn_out_w")),
                      {model_dim, model_dim});
    views.attn_out_b =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "attn_out_b")),
                      {1, model_dim});
    views.ln2_gamma =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ln2_gamma")),
                      {1, model_dim});
    views.ln2_beta =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ln2_beta")),
                      {1, model_dim});
    views.ffn_w1 =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ffn_w1")),
                      {model_dim, ffn_dim});
    views.ffn_b1 =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ffn_b1")),
                      {1, ffn_dim});
    views.ffn_w2 =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ffn_w2")),
                      {ffn_dim, model_dim});
    views.ffn_b2 =
        make_view_f32(cursor.next(lname(static_cast<int>(layer), "ffn_b2")),
                      {1, model_dim});
    layer_param_views_.push_back(views);
  }
  lnf_gamma_ = make_view_f32(cursor.next("lnf_gamma"), {1, model_dim});
  lnf_beta_ = make_view_f32(cursor.next("lnf_beta"), {1, model_dim});
  lm_head_w_ = make_view_f32(cursor.next("lm_head_w"), {model_dim, vocab_size});
  cursor.finish();
}

TrainingTempTensorSubStore::TrainingTempTensorSubStore(
    const Config &cfg, const NamedLayout &temp_layout, uint8_t *temp_base,
    uint64_t temp_bytes, Device device)
    : TempTensorSubStore(cfg, temp_base, temp_bytes, device) {
  const int64_t training_batch_size =
      static_cast<int64_t>(cfg_.training.batch_size);
  const int64_t training_seq_len =
      static_cast<int64_t>(cfg_.training.train_seq_len);
  const int64_t temp_token_count = temp_batch_tokens();
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t qkv_dim = 3 * model_dim;
  const int64_t ffn_dim = static_cast<int64_t>(cfg_.model.d_ff);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t head_count = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t head_dim =
      static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads);

  layer_temp_views_.clear();
  LayoutCursor cursor(temp_layout.slices(), "temporary(training)");
  ds_ids_ = make_temp_view(cursor.next("ds.ids"),
                           Shape{training_batch_size, training_seq_len});
  ds_targets_ = make_temp_view(cursor.next("ds.targets"),
                               Shape{training_batch_size, training_seq_len});

  tr_logits_ = make_temp_view(cursor.next("tr.logits"),
                              Shape{training_batch_size, training_seq_len, vocab_size});
  tr_loss_ = make_temp_view(cursor.next("tr.loss"), Shape{1, 1});
  tr_X_ = make_temp_view(cursor.next("tr.X"),
                         Shape{training_batch_size, training_seq_len, model_dim});
  tr_Y_ = make_temp_view(cursor.next("tr.Y"),
                         Shape{training_batch_size, training_seq_len, model_dim});
  tr_Xn_ = make_temp_view(cursor.next("tr.Xn"),
                          Shape{training_batch_size, training_seq_len, model_dim});

  bw_XnT_ = make_temp_view(cursor.next("bw.XnT"),
                           Shape{model_dim, temp_token_count});
  bw_lm_wT_ = make_temp_view(cursor.next("bw.lm_wT"), Shape{vocab_size, model_dim});
  bw_d_xn_ = make_temp_view(
      cursor.next("bw.d_xn"),
      Shape{training_batch_size, training_seq_len, model_dim});
  bw_d_xlast_ = make_temp_view(
      cursor.next("bw.d_xlast"),
      Shape{training_batch_size, training_seq_len, model_dim});

  for (uint32_t layer = 0; layer < cfg_.model.n_layers; ++layer) {
    TensorStore::LayerTempViews views;
    views.ln1 = make_temp_view(cursor.next(lname(static_cast<int>(layer), "ln1")),
                               Shape{training_batch_size, training_seq_len, model_dim});
    views.bw_d_prev =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "bw.d_prev")),
                       Shape{training_batch_size, training_seq_len, model_dim});
    views.attn_out =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn_out")),
                       Shape{training_batch_size, training_seq_len, model_dim});
    views.resid1 =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "resid1")),
                       Shape{training_batch_size, training_seq_len, model_dim});
    views.ln2 = make_temp_view(cursor.next(lname(static_cast<int>(layer), "ln2")),
                               Shape{training_batch_size, training_seq_len, model_dim});
    views.ffn_out =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn_out")),
                       Shape{training_batch_size, training_seq_len, model_dim});
    views.hidden =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "hidden")),
                       Shape{training_batch_size, training_seq_len, model_dim});

    views.attn_qkv =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.qkv")),
                       Shape{training_batch_size, training_seq_len, qkv_dim});
    views.attn_context = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "attn.context")),
        Shape{training_batch_size, training_seq_len, model_dim});
    views.attn_scores = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "attn.scores")),
        Shape{training_batch_size, training_seq_len, training_seq_len});
    if (!uses_fused_inplace_attention(cfg_)) {
      views.attn_weights = make_temp_view(
          cursor.next(lname(static_cast<int>(layer), "attn.weights")),
          Shape{training_batch_size, training_seq_len, training_seq_len});
    }
    views.attn_weights_cache =
        make_temp_view(cursor.next(lname(static_cast<int>(layer),
                                         "attn.weights_cache")),
                       Shape{head_count, training_batch_size, training_seq_len,
                             training_seq_len});
    views.attn_head = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "attn.head")),
        Shape{training_batch_size, training_seq_len, head_dim});

    views.ffn_h =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.h")),
                       Shape{training_batch_size, training_seq_len, ffn_dim});
    if (!uses_inplace_ffn_activation(cfg_)) {
      views.ffn_a =
          make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.a")),
                         Shape{training_batch_size, training_seq_len, ffn_dim});
    }

    views.dln2 = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "dln2")),
        Shape{training_batch_size, training_seq_len, model_dim});
    views.dy_ln2 = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "dy_ln2")),
        Shape{training_batch_size, training_seq_len, model_dim});
    views.dy_total = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "dy_total")),
        Shape{training_batch_size, training_seq_len, model_dim});
    views.dln1 = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "dln1")),
        Shape{training_batch_size, training_seq_len, model_dim});
    views.dx_ln1 = make_temp_view(
        cursor.next(lname(static_cast<int>(layer), "dx_ln1")),
        Shape{training_batch_size, training_seq_len, model_dim});

    views.attn_contextT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.contextT")),
                       Shape{model_dim, temp_token_count});
    views.attn_WoT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.WoT")),
                       Shape{model_dim, model_dim});
    views.attn_dcontext =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.dcontext")),
                       Shape{training_batch_size, training_seq_len, model_dim});
    views.attn_dqkv =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.dqkv")),
                       Shape{training_batch_size, training_seq_len, qkv_dim});
    views.attn_KhT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.KhT")),
                       Shape{head_dim, temp_token_count});
    views.attn_VhT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.VhT")),
                       Shape{head_dim, temp_token_count});
    views.attn_dweights =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.dweights")),
                       Shape{training_batch_size, training_seq_len, training_seq_len});
    views.attn_weightsT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.weightsT")),
                       Shape{training_batch_size, training_seq_len, training_seq_len});
    views.attn_dscores =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.dscores")),
                       Shape{training_batch_size, training_seq_len, training_seq_len});
    views.attn_dscoresT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.dscoresT")),
                       Shape{training_batch_size, training_seq_len, training_seq_len});
    views.attn_WqkvT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.WqkvT")),
                       Shape{qkv_dim, model_dim});
    views.attn_xT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "attn.xT")),
                       Shape{model_dim, temp_token_count});

    views.ffn_aT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.aT")),
                       Shape{ffn_dim, temp_token_count});
    views.ffn_W2T =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.W2T")),
                       Shape{model_dim, ffn_dim});
    views.ffn_da =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.da")),
                       Shape{training_batch_size, training_seq_len, ffn_dim});
    if (!uses_inplace_ffn_activation(cfg_)) {
      views.ffn_dh =
          make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.dh")),
                         Shape{training_batch_size, training_seq_len, ffn_dim});
    }
    views.ffn_xT =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.xT")),
                       Shape{model_dim, temp_token_count});
    views.ffn_W1T =
        make_temp_view(cursor.next(lname(static_cast<int>(layer), "ffn.W1T")),
                       Shape{ffn_dim, model_dim});

    layer_temp_views_.push_back(views);
  }
  cursor.finish();
}

InferenceTempTensorSubStore::InferenceTempTensorSubStore(
    const Config &cfg, const NamedLayout &temp_layout, uint8_t *temp_base,
    uint64_t temp_bytes, Device device)
    : TempTensorSubStore(cfg, temp_base, temp_bytes, device) {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t qkv_dim = 3 * model_dim;
  const int64_t ffn_dim = static_cast<int64_t>(cfg_.model.d_ff);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t max_seq_len =
      static_cast<int64_t>(cfg_.model.max_seq_len);
  const int64_t head_count = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t head_dim =
      static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads);

  layer_temp_views_.clear();
  LayoutCursor cursor(temp_layout.slices(), "temporary(inference)");
  infer_ids_ = make_temp_view(cursor.next("infer.ids"), Shape{1, max_seq_len});
  infer_logits_ =
      make_temp_view(cursor.next("infer.logits"),
                     Shape{1, max_seq_len, vocab_size});
  tr_X_ = make_temp_view(cursor.next("infer.X"),
                         Shape{1, max_seq_len, model_dim});
  tr_Y_ = make_temp_view(cursor.next("infer.Y"),
                         Shape{1, max_seq_len, model_dim});
  tr_Xn_ = make_temp_view(cursor.next("infer.Xn"),
                          Shape{1, max_seq_len, model_dim});

  for (uint32_t layer = 0; layer < cfg_.model.n_layers; ++layer) {
    TensorStore::LayerTempViews views;
    views.ln1 = make_temp_view(cursor.next(infer_lname(static_cast<int>(layer), "ln1")),
                               Shape{1, max_seq_len, model_dim});
    views.attn_out = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "attn_out")),
        Shape{1, max_seq_len, model_dim});
    views.resid1 = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "resid1")),
        Shape{1, max_seq_len, model_dim});
    views.ln2 = make_temp_view(cursor.next(infer_lname(static_cast<int>(layer), "ln2")),
                               Shape{1, max_seq_len, model_dim});
    views.ffn_out = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "ffn_out")),
        Shape{1, max_seq_len, model_dim});
    views.hidden = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "hidden")),
        Shape{1, max_seq_len, model_dim});

    views.attn_qkv = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "attn.qkv")),
        Shape{1, max_seq_len, qkv_dim});
    views.attn_context = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "attn.context")),
        Shape{1, max_seq_len, model_dim});
    views.attn_scores = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "attn.scores")),
        Shape{1, max_seq_len, max_seq_len});
    if (!uses_fused_inplace_attention(cfg_)) {
      views.attn_weights = make_temp_view(
          cursor.next(infer_lname(static_cast<int>(layer), "attn.weights")),
          Shape{1, max_seq_len, max_seq_len});
    }
    views.attn_weights_cache = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "attn.weights_cache")),
        Shape{head_count, 1, max_seq_len, max_seq_len});
    views.attn_head = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "attn.head")),
        Shape{1, max_seq_len, head_dim});

    views.ffn_h = make_temp_view(
        cursor.next(infer_lname(static_cast<int>(layer), "ffn.h")),
        Shape{1, max_seq_len, ffn_dim});
    if (!uses_inplace_ffn_activation(cfg_)) {
      views.ffn_a = make_temp_view(
          cursor.next(infer_lname(static_cast<int>(layer), "ffn.a")),
          Shape{1, max_seq_len, ffn_dim});
    }

    layer_temp_views_.push_back(views);
  }
  cursor.finish();
}

TensorView TensorStore::make_view_f32(const LayoutSlice &s,
                                        Shape shape) const {
  const uint64_t expected = nbytes(shape, DType::F32);
  require(expected == s.bytes,
          "shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  require(s.offset + s.bytes <= bytes_, "slice out of bounds: " + s.name);
  require((s.offset % alignof(float)) == 0,
          "slice offset not float-aligned: " + s.name);
  require((s.bytes % sizeof(float)) == 0,
          "slice bytes not multiple of float: " + s.name);
  void *ptr = base_ + s.offset;
  return TensorView(device_, DType::F32, ptr, shape);
}

TensorView TensorStore::make_subview_f32(const TensorView &view,
                                           int64_t col_offset,
                                           Shape sub_shape) const {
  require(view.dtype() == DType::F32, "subview source must be f32");
  require(sub_shape.dim(0) == view.shape().dim(0),
          "subview must keep full row count for f32 split");
  require(col_offset >= 0, "negative col_offset for f32 split");
  require(col_offset + sub_shape.dim(1) <= view.shape().dim(1),
          "subview exceeds column bounds for f32 split");
  return view.subcols(col_offset, sub_shape.dim(1));
}

TensorView TensorStore::TempTensorSubStore::make_temp_view(
    const LayoutSlice &s, Shape shape) const {
  const uint64_t expected = nbytes(shape, s.dtype);
  require(expected == s.bytes,
          "temporary shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  require(s.offset + s.bytes <= temp_bytes_,
          "temporary slice out of bounds: " + s.name);
  void *ptr = temp_base_ + s.offset;
  return TensorView(device_, s.dtype, ptr, shape);
}

TensorView TensorStore::TempTensorSubStore::prefix_storage(
    const TensorView &view, Shape shape) const {
  require(shape.dim(0) >= 0 && shape.dim(0) <= view.shape().dim(0),
          "shape row prefix out of bounds");
  require(shape.dim(1) >= 0 && shape.dim(1) <= view.shape().dim(1),
          "shape col prefix out of bounds");
  return view.subrows(0, shape.dim(0)).subcols(0, shape.dim(1));
}

TensorView TrainingTempTensorSubStore::prefix_batch_seq(
    const TensorView &view, int64_t batch_size, int64_t seq_len,
    const char *label, int64_t expected_last_dim) const {
  require(view.rank() == 3, std::string(label) + " rank mismatch");
  if (expected_last_dim >= 0) {
    require(view.dim(2) == expected_last_dim,
            std::string(label) + " last-dim mismatch");
  }
  require(view.dim(0) == batch_size && view.dim(1) == seq_len,
          std::string(label) + " training shape mismatch");
  return view;
}

TensorView TrainingTempTensorSubStore::prefix_batch_seq_square(
    const TensorView &view, int64_t batch_size, int64_t seq_len,
    const char *label) const {
  require(view.rank() == 3, std::string(label) + " rank mismatch");
  require(view.dim(0) == batch_size && view.dim(1) == seq_len &&
              view.dim(2) == seq_len,
          std::string(label) + " training shape mismatch");
  return view;
}

TensorView TrainingTempTensorSubStore::prefix_head_batch_seq_square(
    const TensorView &view, int64_t batch_size, int64_t seq_len,
    const char *label) const {
  require(view.rank() == 4, std::string(label) + " rank mismatch");
  require(view.dim(0) == static_cast<int64_t>(cfg_.model.n_heads),
          std::string(label) + " head-count mismatch");
  require(view.dim(1) == batch_size && view.dim(2) == seq_len &&
              view.dim(3) == seq_len,
          std::string(label) + " training shape mismatch");
  return view;
}

TensorView InferenceTempTensorSubStore::prefix_batch_seq(
    const TensorView &view, int64_t batch_size, int64_t seq_len,
    const char *label, int64_t expected_last_dim) const {
  require(view.rank() == 3, std::string(label) + " rank mismatch");
  if (expected_last_dim >= 0) {
    require(view.dim(2) == expected_last_dim,
            std::string(label) + " last-dim mismatch");
  }
  require(batch_size >= 0 && batch_size <= view.dim(0) &&
              seq_len >= 0 && seq_len <= view.dim(1),
          std::string(label) + " inference shape mismatch");
  return view.slice(0, 0, batch_size).slice(1, 0, seq_len);
}

TensorView InferenceTempTensorSubStore::prefix_batch_seq_square(
    const TensorView &view, int64_t batch_size, int64_t seq_len,
    const char *label) const {
  require(view.rank() == 3, std::string(label) + " rank mismatch");
  require(batch_size >= 0 && batch_size <= view.dim(0) &&
              seq_len >= 0 && seq_len <= view.dim(1) &&
              seq_len >= 0 && seq_len <= view.dim(2),
          std::string(label) + " inference shape mismatch");
  return view.slice(0, 0, batch_size).slice(1, 0, seq_len).slice(2, 0, seq_len);
}

TensorView InferenceTempTensorSubStore::prefix_head_batch_seq_square(
    const TensorView &view, int64_t batch_size, int64_t seq_len,
    const char *label) const {
  require(view.rank() == 4, std::string(label) + " rank mismatch");
  require(view.dim(0) == static_cast<int64_t>(cfg_.model.n_heads),
          std::string(label) + " head-count mismatch");
  require(batch_size >= 0 && batch_size <= view.dim(1) &&
              seq_len >= 0 && seq_len <= view.dim(2) &&
              seq_len >= 0 && seq_len <= view.dim(3),
          std::string(label) + " inference shape mismatch");
  return view.slice(1, 0, batch_size).slice(2, 0, seq_len).slice(3, 0, seq_len);
}

int64_t TensorStore::TempTensorSubStore::temp_batch_tokens() const {
  const uint64_t training_batch_tokens =
      static_cast<uint64_t>(cfg_.training.batch_size) *
      static_cast<uint64_t>(cfg_.training.train_seq_len);
  return static_cast<int64_t>(training_batch_tokens);
}

TensorView TensorStore::TempTensorSubStore::temp_ds_ids(int64_t rows) const {
  return prefix_storage(
      TensorView(ds_ids_.device(), ds_ids_.dtype(), ds_ids_.data(),
                 Shape{static_cast<int64_t>(ds_ids_.numel()), 1}),
      {rows, 1});
}

TensorView TensorStore::TempTensorSubStore::temp_ds_ids() const { return ds_ids_; }

TensorView TensorStore::TempTensorSubStore::temp_ds_ids(int64_t batch_size, int64_t seq_len) const {
  require(ds_ids_.rank() == 2 && ds_ids_.dim(0) == batch_size &&
              ds_ids_.dim(1) == seq_len,
          "temp_ds_ids shape mismatch");
  return ds_ids_;
}

TensorView TensorStore::TempTensorSubStore::temp_ds_targets(int64_t rows) const {
  return prefix_storage(
      TensorView(ds_targets_.device(), ds_targets_.dtype(), ds_targets_.data(),
                 Shape{static_cast<int64_t>(ds_targets_.numel()), 1}),
      {rows, 1});
}

TensorView TensorStore::TempTensorSubStore::temp_ds_targets() const { return ds_targets_; }

TensorView TensorStore::TempTensorSubStore::temp_ds_targets(int64_t batch_size,
                                          int64_t seq_len) const {
  require(ds_targets_.rank() == 2 && ds_targets_.dim(0) == batch_size &&
              ds_targets_.dim(1) == seq_len,
          "temp_ds_targets shape mismatch");
  return ds_targets_;
}

bool TensorStore::TempTensorSubStore::has_inference_io_temps() const {
  return infer_ids_.data() != nullptr && infer_logits_.data() != nullptr;
}

TensorView TensorStore::TempTensorSubStore::temp_infer_ids(int64_t rows) const {
  require(has_inference_io_temps(),
          "temp_infer_ids unavailable for this temp layout");
  require(infer_ids_.rank() == 2 && infer_ids_.dim(0) == 1 &&
              rows >= 0 && rows <= infer_ids_.dim(1),
          "temp_infer_ids shape mismatch");
  return infer_ids_.subcols(0, rows);
}

TensorView TensorStore::TempTensorSubStore::temp_infer_logits(int64_t rows) const {
  require(has_inference_io_temps(),
          "temp_infer_logits unavailable for this temp layout");
  require(infer_logits_.rank() == 3 && infer_logits_.dim(0) == 1 &&
              rows >= 0 && rows <= infer_logits_.dim(1) &&
              infer_logits_.dim(2) ==
                  static_cast<int64_t>(cfg_.model.target_vocab_size),
          "temp_infer_logits shape mismatch");
  return infer_logits_.slice(1, 0, rows);
}

TensorView TensorStore::TempTensorSubStore::temp_tr_logits(int64_t rows) const {
  return prefix_storage(
                        TensorView(tr_logits_.device(), tr_logits_.dtype(),
                                   tr_logits_.data(),
                                   Shape{static_cast<int64_t>(
                                             tr_logits_.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.target_vocab_size)),
                                         static_cast<int64_t>(
                                             cfg_.model.target_vocab_size)}),
                        {rows, static_cast<int64_t>(cfg_.model.target_vocab_size)});
}

TensorView TensorStore::TempTensorSubStore::temp_tr_logits(int64_t batch_size,
                                         int64_t seq_len) const {
  return prefix_batch_seq(tr_logits_, batch_size, seq_len,
                          "temp_tr_logits",
                          static_cast<int64_t>(cfg_.model.target_vocab_size));
}

TensorView TensorStore::TempTensorSubStore::temp_tr_loss() const { return tr_loss_; }

TensorView TensorStore::TempTensorSubStore::temp_tr_X(int64_t rows) const {
  return prefix_storage(
                        TensorView(tr_X_.device(), tr_X_.dtype(), tr_X_.data(),
                                   Shape{static_cast<int64_t>(
                                             tr_X_.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                        {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_tr_X(int64_t batch_size,
                                    int64_t seq_len) const {
  return prefix_batch_seq(tr_X_, batch_size, seq_len, "temp_tr_X",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_tr_Y(int64_t rows) const {
  return prefix_storage(
                        TensorView(tr_Y_.device(), tr_Y_.dtype(), tr_Y_.data(),
                                   Shape{static_cast<int64_t>(
                                             tr_Y_.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                        {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_tr_Y(int64_t batch_size,
                                    int64_t seq_len) const {
  return prefix_batch_seq(tr_Y_, batch_size, seq_len, "temp_tr_Y",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_tr_Xn(int64_t rows) const {
  return prefix_storage(
                        TensorView(tr_Xn_.device(), tr_Xn_.dtype(), tr_Xn_.data(),
                                   Shape{static_cast<int64_t>(
                                             tr_Xn_.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                        {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_tr_Xn(int64_t batch_size,
                                     int64_t seq_len) const {
  return prefix_batch_seq(tr_Xn_, batch_size, seq_len, "temp_tr_Xn",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_bw_XnT(int64_t rows) const {
  return prefix_storage(bw_XnT_, {static_cast<int64_t>(cfg_.model.d_model), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_bw_lm_wT() const { return bw_lm_wT_; }

TensorView TensorStore::TempTensorSubStore::temp_bw_d_xn(int64_t rows) const {
  return prefix_storage(
      TensorView(
          bw_d_xn_.device(), bw_d_xn_.dtype(), bw_d_xn_.data(),
          Shape{static_cast<int64_t>(bw_d_xn_.numel() /
                                     static_cast<uint64_t>(cfg_.model.d_model)),
                static_cast<int64_t>(cfg_.model.d_model)}),
      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_bw_d_xn(int64_t batch_size,
                                       int64_t seq_len) const {
  require(bw_d_xn_.rank() == 3 && bw_d_xn_.dim(0) == batch_size &&
              bw_d_xn_.dim(1) == seq_len &&
              bw_d_xn_.dim(2) == static_cast<int64_t>(cfg_.model.d_model),
          "temp_bw_d_xn shape mismatch");
  return bw_d_xn_;
}

TensorView TensorStore::TempTensorSubStore::temp_bw_d_xlast(int64_t rows) const {
  return prefix_storage(
      TensorView(
          bw_d_xlast_.device(), bw_d_xlast_.dtype(), bw_d_xlast_.data(),
          Shape{static_cast<int64_t>(bw_d_xlast_.numel() /
                                     static_cast<uint64_t>(cfg_.model.d_model)),
                static_cast<int64_t>(cfg_.model.d_model)}),
      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_bw_d_xlast(int64_t batch_size,
                                          int64_t seq_len) const {
  require(bw_d_xlast_.rank() == 3 && bw_d_xlast_.dim(0) == batch_size &&
              bw_d_xlast_.dim(1) == seq_len &&
              bw_d_xlast_.dim(2) == static_cast<int64_t>(cfg_.model.d_model),
          "temp_bw_d_xlast shape mismatch");
  return bw_d_xlast_;
}

TensorView TensorStore::TempTensorSubStore::temp_layer_ln1(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(
      layer_temp_views_[static_cast<size_t>(layer)].ln1, batch_size, seq_len,
      "temp_layer_ln1", static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_layer_attn_out(int layer, int64_t batch_size,
                                              int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(layer_temp_views_[static_cast<size_t>(layer)].attn_out,
                          batch_size, seq_len, "temp_layer_attn_out",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_layer_resid1(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(layer_temp_views_[static_cast<size_t>(layer)].resid1,
                          batch_size, seq_len, "temp_layer_resid1",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_layer_ln2(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(
      layer_temp_views_[static_cast<size_t>(layer)].ln2, batch_size, seq_len,
      "temp_layer_ln2", static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_layer_ffn_out(int layer, int64_t batch_size,
                                             int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(layer_temp_views_[static_cast<size_t>(layer)].ffn_out,
                          batch_size, seq_len, "temp_layer_ffn_out",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_layer_hidden(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(layer_temp_views_[static_cast<size_t>(layer)].hidden,
                          batch_size, seq_len, "temp_layer_hidden",
                          static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_layer_bw_d_prev(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].bw_d_prev;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_layer_bw_d_prev(int layer, int64_t batch_size,
                                               int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].bw_d_prev;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len,
          "temp_layer_bw_d_prev shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dln2(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dln2;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dln2(int layer, int64_t batch_size,
                                          int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dln2;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len,
          "temp_layer_dln2 shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dy_ln2(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dy_ln2;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dy_ln2(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dy_ln2;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len,
          "temp_layer_dy_ln2 shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dy_total(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dy_total;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dy_total(int layer, int64_t batch_size,
                                              int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dy_total;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len,
          "temp_layer_dy_total shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dln1(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dln1;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dln1(int layer, int64_t batch_size,
                                          int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dln1;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len,
          "temp_layer_dln1 shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dx_ln1(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dx_ln1;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_layer_dx_ln1(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].dx_ln1;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len,
          "temp_layer_dx_ln1 shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_qkv(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].attn_qkv;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 3 * cfg_.model.d_model)),
                                         3 * static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, 3 * static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_qkv(int layer, int64_t batch_size,
                                        int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(
      layer_temp_views_[static_cast<size_t>(layer)].attn_qkv, batch_size,
      seq_len, "temp_attn_qkv",
      3 * static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_attn_context(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_context;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_context(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(
      layer_temp_views_[static_cast<size_t>(layer)].attn_context, batch_size,
      seq_len, "temp_attn_context",
      static_cast<int64_t>(cfg_.model.d_model));
}

TensorView TensorStore::TempTensorSubStore::temp_attn_scores(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_scores;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(seq_len)),
                                         seq_len}),
                      {rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_scores(int layer, int64_t batch_size,
                                           int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq_square(
      layer_temp_views_[static_cast<size_t>(layer)].attn_scores, batch_size,
      seq_len, "temp_attn_scores");
}

TensorView TensorStore::TempTensorSubStore::temp_attn_weights(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_weights;
  require(t.data() != nullptr,
          "temp_attn_weights is not allocated for model_algo.attention=" +
              cfg_.model_algo.attention);
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(seq_len)),
                                         seq_len}),
                      {rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_weights(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  require(layer_temp_views_[static_cast<size_t>(layer)]
              .attn_weights.data() != nullptr,
          "temp_attn_weights is not allocated for model_algo.attention=" +
              cfg_.model_algo.attention);
  return prefix_batch_seq_square(
      layer_temp_views_[static_cast<size_t>(layer)].attn_weights, batch_size,
      seq_len, "temp_attn_weights");
}

TensorView TensorStore::TempTensorSubStore::temp_attn_cached_weights(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_weights_cache;
  return prefix_storage(
      TensorView(t.device(), t.dtype(), t.data(),
                 Shape{static_cast<int64_t>(t.numel() /
                                            static_cast<uint64_t>(seq_len)),
                       seq_len}),
      {static_cast<int64_t>(cfg_.model.n_heads) * rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_cached_weights(
    int layer, int64_t batch_size, int64_t seq_len) const {
  check_layer(layer);
  return prefix_head_batch_seq_square(
      layer_temp_views_[static_cast<size_t>(layer)].attn_weights_cache,
      batch_size, seq_len, "temp_attn_cached_weights");
}

TensorView TensorStore::TempTensorSubStore::temp_attn_head(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(
      layer_temp_views_[static_cast<size_t>(layer)].attn_head, batch_size,
      seq_len, "temp_attn_head",
      static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads));
}

TensorView TensorStore::TempTensorSubStore::temp_attn_contextT(int layer, int64_t rows) const {
  check_layer(layer);
  return prefix_storage(layer_temp_views_[static_cast<size_t>(layer)].attn_contextT,
                      {static_cast<int64_t>(cfg_.model.d_model), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_WoT(int layer) const {
  check_layer(layer);
  return layer_temp_views_[static_cast<size_t>(layer)].attn_WoT;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dcontext(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dcontext;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_model)),
                                         static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dcontext(int layer, int64_t batch_size,
                                             int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dcontext;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len &&
              t.dim(2) == static_cast<int64_t>(cfg_.model.d_model),
          "temp_attn_dcontext shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dqkv(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].attn_dqkv;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 3 * cfg_.model.d_model)),
                                         3 * static_cast<int64_t>(cfg_.model.d_model)}),
                      {rows, 3 * static_cast<int64_t>(cfg_.model.d_model)});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dqkv(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].attn_dqkv;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len &&
              t.dim(2) == 3 * static_cast<int64_t>(cfg_.model.d_model),
          "temp_attn_dqkv shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_KhT(int layer, int64_t rows) const {
  check_layer(layer);
  return prefix_storage(
      layer_temp_views_[static_cast<size_t>(layer)].attn_KhT,
      {static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_VhT(int layer, int64_t rows) const {
  check_layer(layer);
  return prefix_storage(
      layer_temp_views_[static_cast<size_t>(layer)].attn_VhT,
      {static_cast<int64_t>(cfg_.model.d_model / cfg_.model.n_heads), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dweights(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dweights;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(seq_len)),
                                         seq_len}),
                      {rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dweights(int layer, int64_t batch_size,
                                             int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dweights;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len &&
              t.dim(2) == seq_len,
          "temp_attn_dweights shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_weightsT(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_weightsT;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(seq_len)),
                                         seq_len}),
                      {rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dscores(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dscores;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(seq_len)),
                                         seq_len}),
                      {rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dscores(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dscores;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len &&
              t.dim(2) == seq_len,
          "temp_attn_dscores shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_dscoresT(int layer, int64_t rows) const {
  check_layer(layer);
  const int64_t seq_len = static_cast<int64_t>(cfg_.training.train_seq_len);
  const TensorView &t =
      layer_temp_views_[static_cast<size_t>(layer)].attn_dscoresT;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(seq_len)),
                                         seq_len}),
                      {rows, seq_len});
}

TensorView TensorStore::TempTensorSubStore::temp_attn_WqkvT(int layer) const {
  check_layer(layer);
  return layer_temp_views_[static_cast<size_t>(layer)].attn_WqkvT;
}

TensorView TensorStore::TempTensorSubStore::temp_attn_xT(int layer, int64_t rows) const {
  check_layer(layer);
  return prefix_storage(layer_temp_views_[static_cast<size_t>(layer)].attn_xT,
                      {static_cast<int64_t>(cfg_.model.d_model), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_h(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].ffn_h;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_ff)),
                                         static_cast<int64_t>(cfg_.model.d_ff)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_h(int layer, int64_t batch_size,
                                     int64_t seq_len) const {
  check_layer(layer);
  return prefix_batch_seq(layer_temp_views_[static_cast<size_t>(layer)].ffn_h,
                          batch_size, seq_len, "temp_ffn_h",
                          static_cast<int64_t>(cfg_.model.d_ff));
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_a(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].ffn_a;
  require(t.data() != nullptr,
          "temp_ffn_a is not allocated for model_algo.ffn=" +
              cfg_.model_algo.ffn);
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_ff)),
                                         static_cast<int64_t>(cfg_.model.d_ff)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_a(int layer, int64_t batch_size,
                                     int64_t seq_len) const {
  check_layer(layer);
  require(layer_temp_views_[static_cast<size_t>(layer)].ffn_a.data() != nullptr,
          "temp_ffn_a is not allocated for model_algo.ffn=" +
              cfg_.model_algo.ffn);
  return prefix_batch_seq(layer_temp_views_[static_cast<size_t>(layer)].ffn_a,
                          batch_size, seq_len, "temp_ffn_a",
                          static_cast<int64_t>(cfg_.model.d_ff));
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_aT(int layer, int64_t rows) const {
  check_layer(layer);
  return prefix_storage(layer_temp_views_[static_cast<size_t>(layer)].ffn_aT,
                      {static_cast<int64_t>(cfg_.model.d_ff), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_W2T(int layer) const {
  check_layer(layer);
  return layer_temp_views_[static_cast<size_t>(layer)].ffn_W2T;
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_da(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].ffn_da;
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_ff)),
                                         static_cast<int64_t>(cfg_.model.d_ff)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_da(int layer, int64_t batch_size,
                                      int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].ffn_da;
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len &&
              t.dim(2) == static_cast<int64_t>(cfg_.model.d_ff),
          "temp_ffn_da shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_dh(int layer, int64_t rows) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].ffn_dh;
  require(t.data() != nullptr,
          "temp_ffn_dh is not allocated for model_algo.ffn=" +
              cfg_.model_algo.ffn);
  return prefix_storage(TensorView(t.device(), t.dtype(), t.data(),
                                   Shape{static_cast<int64_t>(
                                             t.numel() /
                                             static_cast<uint64_t>(
                                                 cfg_.model.d_ff)),
                                         static_cast<int64_t>(cfg_.model.d_ff)}),
                      {rows, static_cast<int64_t>(cfg_.model.d_ff)});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_dh(int layer, int64_t batch_size,
                                      int64_t seq_len) const {
  check_layer(layer);
  const TensorView &t = layer_temp_views_[static_cast<size_t>(layer)].ffn_dh;
  require(t.data() != nullptr,
          "temp_ffn_dh is not allocated for model_algo.ffn=" +
              cfg_.model_algo.ffn);
  require(t.rank() == 3 && t.dim(0) == batch_size && t.dim(1) == seq_len &&
              t.dim(2) == static_cast<int64_t>(cfg_.model.d_ff),
          "temp_ffn_dh shape mismatch");
  return t;
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_xT(int layer, int64_t rows) const {
  check_layer(layer);
  return prefix_storage(layer_temp_views_[static_cast<size_t>(layer)].ffn_xT,
                      {static_cast<int64_t>(cfg_.model.d_model), rows});
}

TensorView TensorStore::TempTensorSubStore::temp_ffn_W1T(int layer) const {
  check_layer(layer);
  return layer_temp_views_[static_cast<size_t>(layer)].ffn_W1T;
}

TensorView TensorStore::temp_ds_ids() const { return temp_tensor_substore_->temp_ds_ids(); }
TensorView TensorStore::temp_ds_ids(int64_t rows) const {
  return temp_tensor_substore_->temp_ds_ids(rows);
}
TensorView TensorStore::temp_ds_ids(int64_t batch_size,
                                      int64_t seq_len) const {
  return temp_tensor_substore_->temp_ds_ids(batch_size, seq_len);
}
TensorView TensorStore::temp_ds_targets() const {
  return temp_tensor_substore_->temp_ds_targets();
}
TensorView TensorStore::temp_ds_targets(int64_t rows) const {
  return temp_tensor_substore_->temp_ds_targets(rows);
}
TensorView TensorStore::temp_ds_targets(int64_t batch_size,
                                          int64_t seq_len) const {
  return temp_tensor_substore_->temp_ds_targets(batch_size, seq_len);
}
bool TensorStore::has_inference_io_temps() const {
  return temp_tensor_substore_->has_inference_io_temps();
}
TensorView TensorStore::temp_infer_ids(int64_t rows) const {
  return temp_tensor_substore_->temp_infer_ids(rows);
}
TensorView TensorStore::temp_infer_logits(int64_t rows) const {
  return temp_tensor_substore_->temp_infer_logits(rows);
}
TensorView TensorStore::temp_tr_logits(int64_t rows) const {
  return temp_tensor_substore_->temp_tr_logits(rows);
}
TensorView TensorStore::temp_tr_logits(int64_t batch_size,
                                         int64_t seq_len) const {
  return temp_tensor_substore_->temp_tr_logits(batch_size, seq_len);
}
TensorView TensorStore::temp_tr_loss() const { return temp_tensor_substore_->temp_tr_loss(); }
TensorView TensorStore::temp_tr_X(int64_t rows) const {
  return temp_tensor_substore_->temp_tr_X(rows);
}
TensorView TensorStore::temp_tr_X(int64_t batch_size,
                                    int64_t seq_len) const {
  return temp_tensor_substore_->temp_tr_X(batch_size, seq_len);
}
TensorView TensorStore::temp_tr_Y(int64_t rows) const {
  return temp_tensor_substore_->temp_tr_Y(rows);
}
TensorView TensorStore::temp_tr_Y(int64_t batch_size,
                                    int64_t seq_len) const {
  return temp_tensor_substore_->temp_tr_Y(batch_size, seq_len);
}
TensorView TensorStore::temp_tr_Xn(int64_t rows) const {
  return temp_tensor_substore_->temp_tr_Xn(rows);
}
TensorView TensorStore::temp_tr_Xn(int64_t batch_size,
                                     int64_t seq_len) const {
  return temp_tensor_substore_->temp_tr_Xn(batch_size, seq_len);
}
TensorView TensorStore::temp_bw_XnT(int64_t rows) const {
  return temp_tensor_substore_->temp_bw_XnT(rows);
}
TensorView TensorStore::temp_bw_lm_wT() const {
  return temp_tensor_substore_->temp_bw_lm_wT();
}
TensorView TensorStore::temp_bw_d_xn(int64_t rows) const {
  return temp_tensor_substore_->temp_bw_d_xn(rows);
}
TensorView TensorStore::temp_bw_d_xn(int64_t batch_size,
                                       int64_t seq_len) const {
  return temp_tensor_substore_->temp_bw_d_xn(batch_size, seq_len);
}
TensorView TensorStore::temp_bw_d_xlast(int64_t rows) const {
  return temp_tensor_substore_->temp_bw_d_xlast(rows);
}
TensorView TensorStore::temp_bw_d_xlast(int64_t batch_size,
                                          int64_t seq_len) const {
  return temp_tensor_substore_->temp_bw_d_xlast(batch_size, seq_len);
}
TensorView TensorStore::temp_layer_ln1(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_ln1(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_attn_out(int layer, int64_t batch_size,
                                              int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_attn_out(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_resid1(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_resid1(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_ln2(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_ln2(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_ffn_out(int layer, int64_t batch_size,
                                             int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_ffn_out(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_hidden(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_hidden(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_bw_d_prev(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_layer_bw_d_prev(layer, rows);
}
TensorView TensorStore::temp_layer_bw_d_prev(int layer, int64_t batch_size,
                                               int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_bw_d_prev(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_dln2(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_layer_dln2(layer, rows);
}
TensorView TensorStore::temp_layer_dln2(int layer, int64_t batch_size,
                                          int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_dln2(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_dy_ln2(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_layer_dy_ln2(layer, rows);
}
TensorView TensorStore::temp_layer_dy_ln2(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_dy_ln2(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_dy_total(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_layer_dy_total(layer, rows);
}
TensorView TensorStore::temp_layer_dy_total(int layer, int64_t batch_size,
                                              int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_dy_total(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_dln1(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_layer_dln1(layer, rows);
}
TensorView TensorStore::temp_layer_dln1(int layer, int64_t batch_size,
                                          int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_dln1(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_layer_dx_ln1(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_layer_dx_ln1(layer, rows);
}
TensorView TensorStore::temp_layer_dx_ln1(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_layer_dx_ln1(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_qkv(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_qkv(layer, rows);
}
TensorView TensorStore::temp_attn_qkv(int layer, int64_t batch_size,
                                        int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_qkv(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_context(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_context(layer, rows);
}
TensorView TensorStore::temp_attn_context(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_context(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_scores(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_scores(layer, rows);
}
TensorView TensorStore::temp_attn_scores(int layer, int64_t batch_size,
                                           int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_scores(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_weights(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_weights(layer, rows);
}
TensorView TensorStore::temp_attn_weights(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_weights(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_cached_weights(int layer,
                                                   int64_t rows) const {
  return temp_tensor_substore_->temp_attn_cached_weights(layer, rows);
}
TensorView TensorStore::temp_attn_cached_weights(
    int layer, int64_t batch_size, int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_cached_weights(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_head(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_head(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_contextT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_contextT(layer, rows);
}
TensorView TensorStore::temp_attn_WoT(int layer) const {
  return temp_tensor_substore_->temp_attn_WoT(layer);
}
TensorView TensorStore::temp_attn_dcontext(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_dcontext(layer, rows);
}
TensorView TensorStore::temp_attn_dcontext(int layer, int64_t batch_size,
                                             int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_dcontext(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_dqkv(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_dqkv(layer, rows);
}
TensorView TensorStore::temp_attn_dqkv(int layer, int64_t batch_size,
                                         int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_dqkv(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_KhT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_KhT(layer, rows);
}
TensorView TensorStore::temp_attn_VhT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_VhT(layer, rows);
}
TensorView TensorStore::temp_attn_dweights(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_dweights(layer, rows);
}
TensorView TensorStore::temp_attn_dweights(int layer, int64_t batch_size,
                                             int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_dweights(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_weightsT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_weightsT(layer, rows);
}
TensorView TensorStore::temp_attn_dscores(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_dscores(layer, rows);
}
TensorView TensorStore::temp_attn_dscores(int layer, int64_t batch_size,
                                            int64_t seq_len) const {
  return temp_tensor_substore_->temp_attn_dscores(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_attn_dscoresT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_dscoresT(layer, rows);
}
TensorView TensorStore::temp_attn_WqkvT(int layer) const {
  return temp_tensor_substore_->temp_attn_WqkvT(layer);
}
TensorView TensorStore::temp_attn_xT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_attn_xT(layer, rows);
}
TensorView TensorStore::temp_ffn_h(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_ffn_h(layer, rows);
}
TensorView TensorStore::temp_ffn_h(int layer, int64_t batch_size,
                                     int64_t seq_len) const {
  return temp_tensor_substore_->temp_ffn_h(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_ffn_a(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_ffn_a(layer, rows);
}
TensorView TensorStore::temp_ffn_a(int layer, int64_t batch_size,
                                     int64_t seq_len) const {
  return temp_tensor_substore_->temp_ffn_a(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_ffn_aT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_ffn_aT(layer, rows);
}
TensorView TensorStore::temp_ffn_W2T(int layer) const {
  return temp_tensor_substore_->temp_ffn_W2T(layer);
}
TensorView TensorStore::temp_ffn_da(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_ffn_da(layer, rows);
}
TensorView TensorStore::temp_ffn_da(int layer, int64_t batch_size,
                                      int64_t seq_len) const {
  return temp_tensor_substore_->temp_ffn_da(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_ffn_dh(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_ffn_dh(layer, rows);
}
TensorView TensorStore::temp_ffn_dh(int layer, int64_t batch_size,
                                      int64_t seq_len) const {
  return temp_tensor_substore_->temp_ffn_dh(layer, batch_size, seq_len);
}
TensorView TensorStore::temp_ffn_xT(int layer, int64_t rows) const {
  return temp_tensor_substore_->temp_ffn_xT(layer, rows);
}
TensorView TensorStore::temp_ffn_W1T(int layer) const {
  return temp_tensor_substore_->temp_ffn_W1T(layer);
}

TensorView TensorStore::layer_ln1_gamma(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln1_gamma;
}

TensorView TensorStore::layer_ln1_beta(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln1_beta;
}

TensorView TensorStore::layer_ln2_gamma(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln2_gamma;
}

TensorView TensorStore::layer_ln2_beta(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln2_beta;
}

TensorView TensorStore::layer_attn_qkv_w(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_qkv_w;
}

TensorView TensorStore::layer_attn_qkv_b(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_qkv_b;
}

TensorView TensorStore::layer_attn_out_w(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_out_w;
}

TensorView TensorStore::layer_attn_out_b(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_out_b;
}

TensorView TensorStore::layer_attn_wq(int layer) const {
  check_layer(layer);
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return make_subview_f32(layer_param_views_[static_cast<size_t>(layer)].attn_qkv_w,
                          0, {D, D});
}

TensorView TensorStore::layer_attn_wk(int layer) const {
  check_layer(layer);
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return make_subview_f32(layer_param_views_[static_cast<size_t>(layer)].attn_qkv_w,
                          D, {D, D});
}

TensorView TensorStore::layer_attn_wv(int layer) const {
  check_layer(layer);
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return make_subview_f32(layer_param_views_[static_cast<size_t>(layer)].attn_qkv_w,
                          2 * D, {D, D});
}

TensorView TensorStore::layer_attn_bq(int layer) const {
  check_layer(layer);
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return make_subview_f32(layer_param_views_[static_cast<size_t>(layer)].attn_qkv_b,
                          0, {1, D});
}

TensorView TensorStore::layer_attn_bk(int layer) const {
  check_layer(layer);
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return make_subview_f32(layer_param_views_[static_cast<size_t>(layer)].attn_qkv_b,
                          D, {1, D});
}

TensorView TensorStore::layer_attn_bv(int layer) const {
  check_layer(layer);
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  return make_subview_f32(layer_param_views_[static_cast<size_t>(layer)].attn_qkv_b,
                          2 * D, {1, D});
}

TensorView TensorStore::layer_ffn_w1(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_w1;
}

TensorView TensorStore::layer_ffn_b1(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_b1;
}

TensorView TensorStore::layer_ffn_w2(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_w2;
}

TensorView TensorStore::layer_ffn_b2(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_b2;
}

const TensorView &TensorStore::tok_embedding() const { return tok_embedding_; }
const TensorView &TensorStore::pos_embedding() const { return pos_embedding_; }
const TensorView &TensorStore::lnf_gamma() const { return lnf_gamma_; }
const TensorView &TensorStore::lnf_beta() const { return lnf_beta_; }
const TensorView &TensorStore::lm_head_w() const { return lm_head_w_; }

const TensorView &TensorStore::param_ffn_w1(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_w1;
}
const TensorView &TensorStore::param_ffn_b1(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_b1;
}
const TensorView &TensorStore::param_ffn_w2(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_w2;
}
const TensorView &TensorStore::param_ffn_b2(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_b2;
}
const TensorView &TensorStore::param_attn_qkv_w(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_qkv_w;
}
const TensorView &TensorStore::param_attn_qkv_b(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_qkv_b;
}
const TensorView &TensorStore::param_attn_out_w(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_out_w;
}
const TensorView &TensorStore::param_attn_out_b(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_out_b;
}
const TensorView &TensorStore::param_ln1_gamma(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln1_gamma;
}
const TensorView &TensorStore::param_ln1_beta(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln1_beta;
}
const TensorView &TensorStore::param_ln2_gamma(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln2_gamma;
}
const TensorView &TensorStore::param_ln2_beta(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln2_beta;
}
const TensorView &TensorStore::param_tok_embedding() const { return tok_embedding_; }
const TensorView &TensorStore::param_pos_embedding() const { return pos_embedding_; }
const TensorView &TensorStore::param_lnf_gamma() const { return lnf_gamma_; }
const TensorView &TensorStore::param_lnf_beta() const { return lnf_beta_; }
const TensorView &TensorStore::param_lm_head_w() const { return lm_head_w_; }

void TensorStore::initialize_parameters_deterministic(
    DeviceBackend &device_backend) const {
  const uint64_t n = bytes_ / sizeof(float);
  std::vector<float> staging(static_cast<size_t>(n), 0.0f);

  auto offset_floats = [&](const TensorView &view) -> uint64_t {
    auto *ptr = reinterpret_cast<uint8_t *>(view.data());
    require(ptr >= base_ && ptr < base_ + bytes_,
            "parameter view pointer outside arena");
    const uint64_t offset = static_cast<uint64_t>(ptr - base_);
    require((offset % sizeof(float)) == 0,
            "parameter view pointer is not float-aligned");
    return offset / sizeof(float);
  };

  auto fill_view = [&](const TensorView &view, float value) {
    const uint64_t offset = offset_floats(view);
    const uint64_t count = view.numel();
    require(offset + count <= staging.size(),
            "parameter view exceeds staging buffer");
    std::fill(staging.begin() + static_cast<std::ptrdiff_t>(offset),
              staging.begin() + static_cast<std::ptrdiff_t>(offset + count),
              value);
  };

  auto init_weight = [&](const TensorView &view, float scale) {
    const uint64_t offset = offset_floats(view);
    const uint64_t count = view.numel();
    require(offset + count <= staging.size(),
            "parameter view exceeds staging buffer");
    for (uint64_t i = 0; i < count; ++i) {
      const uint64_t bits = splitmix64(offset + i);
      const int64_t centered =
          static_cast<int64_t>(bits & 0xFFFFFULL) -
          static_cast<int64_t>(0x80000ULL);
      staging[static_cast<size_t>(offset + i)] =
          (static_cast<float>(centered) / static_cast<float>(0x80000ULL)) *
          scale;
    }
  };

  init_weight(tok_embedding_, 0.02f);
  init_weight(pos_embedding_, 0.02f);
  for (const LayerParamViews &layer : layer_param_views_) {
    fill_view(layer.ln1_gamma, 1.0f);
    fill_view(layer.ln1_beta, 0.0f);
    init_weight(layer.attn_qkv_w, 0.02f);
    fill_view(layer.attn_qkv_b, 0.0f);
    init_weight(layer.attn_out_w, 0.02f);
    fill_view(layer.attn_out_b, 0.0f);
    fill_view(layer.ln2_gamma, 1.0f);
    fill_view(layer.ln2_beta, 0.0f);
    init_weight(layer.ffn_w1, 0.02f);
    fill_view(layer.ffn_b1, 0.0f);
    init_weight(layer.ffn_w2, 0.02f);
    fill_view(layer.ffn_b2, 0.0f);
  }
  fill_view(lnf_gamma_, 1.0f);
  fill_view(lnf_beta_, 0.0f);
  init_weight(lm_head_w_, 0.02f);

  device_backend.copy_host2device(base_, staging.data(), bytes_);
}

#undef require
