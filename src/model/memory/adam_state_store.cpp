#include "adam_state_store.hpp"

#include <utils/assert.hpp>

#include <stdexcept>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("AdamStateStore: ") + std::string(msg);                \
  })

namespace {
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

AdamStateStore::AdamStateStore(const Config &cfg,
                                   const NamedLayout &param_layout,
                                   void *params_base, uint64_t params_bytes,
                                   const AdamStateView &adam_state)
    : cfg_(cfg),
      params_base_(reinterpret_cast<uint8_t *>(params_base)),
      params_bytes_(params_bytes),
      adam_base_(reinterpret_cast<uint8_t *>(adam_state.base)),
      adam_bytes_(adam_state.bytes),
      param_bytes_(param_layout.total_bytes()),
      device_(adam_state.device) {
  require(params_base_ != nullptr, "params_base is null");
  require(params_bytes_ > 0, "params_bytes must be > 0");
  require(param_bytes_ <= params_bytes_,
          "param_layout.total_bytes() exceeds provided params_bytes");
  require(adam_base_ != nullptr, "adam_base is null");
  require(adam_bytes_ >= param_bytes_ * 2,
          "adam state must hold both m and v buffers");
  build_state_views(param_layout);
}

const AdamStateStore::StatePair &
AdamStateStore::state_for_param(const TensorView &param) const {
  const auto it = state_by_param_data_.find(param.data());
  require(it != state_by_param_data_.end(),
          "missing optimizer state for parameter");
  return it->second;
}

const AdamStateStore::StatePair &AdamStateStore::param_tok_embedding() const {
  return tok_embedding_;
}

const AdamStateStore::StatePair &AdamStateStore::param_pos_embedding() const {
  return pos_embedding_;
}

const AdamStateStore::StatePair &AdamStateStore::param_lnf_gamma() const {
  return lnf_gamma_;
}

const AdamStateStore::StatePair &AdamStateStore::param_lnf_beta() const {
  return lnf_beta_;
}

const AdamStateStore::StatePair &AdamStateStore::param_lm_head_w() const {
  return lm_head_w_;
}

const AdamStateStore::StatePair &AdamStateStore::param_ffn_w1(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_w1;
}

const AdamStateStore::StatePair &AdamStateStore::param_ffn_b1(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_b1;
}

const AdamStateStore::StatePair &AdamStateStore::param_ffn_w2(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_w2;
}

const AdamStateStore::StatePair &AdamStateStore::param_ffn_b2(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ffn_b2;
}

const AdamStateStore::StatePair &
AdamStateStore::param_attn_qkv_w(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_qkv_w;
}

const AdamStateStore::StatePair &
AdamStateStore::param_attn_qkv_b(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_qkv_b;
}

const AdamStateStore::StatePair &
AdamStateStore::param_attn_out_w(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_out_w;
}

const AdamStateStore::StatePair &
AdamStateStore::param_attn_out_b(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].attn_out_b;
}

const AdamStateStore::StatePair &AdamStateStore::param_ln1_gamma(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln1_gamma;
}

const AdamStateStore::StatePair &AdamStateStore::param_ln1_beta(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln1_beta;
}

const AdamStateStore::StatePair &AdamStateStore::param_ln2_gamma(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln2_gamma;
}

const AdamStateStore::StatePair &AdamStateStore::param_ln2_beta(int layer) const {
  check_layer(layer);
  return layer_param_views_[static_cast<size_t>(layer)].ln2_beta;
}

void AdamStateStore::check_layer(int layer) const {
  require(layer >= 0, "layer < 0");
  require(static_cast<uint32_t>(layer) < cfg_.model.n_layers,
          "layer out of range");
}

void AdamStateStore::build_state_views(const NamedLayout &param_layout) {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t ffn_dim = static_cast<int64_t>(cfg_.model.d_ff);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t qkv_dim = 3 * model_dim;

  LayoutCursor cursor(param_layout.slices(), "parameter");

  const LayoutSlice &tok_embedding_slice = cursor.next("tok_embedding");
  tok_embedding_ =
      make_state_pair_f32(tok_embedding_slice, {vocab_size, model_dim}, true);
  register_state(make_param_view_f32(tok_embedding_slice, {vocab_size, model_dim}),
                 tok_embedding_);

  const LayoutSlice &pos_embedding_slice = cursor.next("pos_embedding");
  pos_embedding_ = make_state_pair_f32(
      pos_embedding_slice,
      {static_cast<int64_t>(cfg_.model.max_seq_len), model_dim}, true);
  register_state(
      make_param_view_f32(
          pos_embedding_slice,
          {static_cast<int64_t>(cfg_.model.max_seq_len), model_dim}),
      pos_embedding_);

  layer_param_views_.resize(cfg_.model.n_layers);
  for (uint32_t layer = 0; layer < cfg_.model.n_layers; ++layer) {
    LayerParamViews &views = layer_param_views_[layer];

    const LayoutSlice &ln1_gamma_slice =
        cursor.next(lname(static_cast<int>(layer), "ln1_gamma"));
    views.ln1_gamma = make_state_pair_f32(ln1_gamma_slice, {1, model_dim}, false);
    register_state(make_param_view_f32(ln1_gamma_slice, {1, model_dim}),
                   views.ln1_gamma);

    const LayoutSlice &ln1_beta_slice =
        cursor.next(lname(static_cast<int>(layer), "ln1_beta"));
    views.ln1_beta = make_state_pair_f32(ln1_beta_slice, {1, model_dim}, false);
    register_state(make_param_view_f32(ln1_beta_slice, {1, model_dim}),
                   views.ln1_beta);

    const LayoutSlice &attn_qkv_w_slice =
        cursor.next(lname(static_cast<int>(layer), "attn_qkv_w"));
    views.attn_qkv_w =
        make_state_pair_f32(attn_qkv_w_slice, {model_dim, qkv_dim}, true);
    register_state(make_param_view_f32(attn_qkv_w_slice, {model_dim, qkv_dim}),
                   views.attn_qkv_w);

    const LayoutSlice &attn_qkv_b_slice =
        cursor.next(lname(static_cast<int>(layer), "attn_qkv_b"));
    views.attn_qkv_b = make_state_pair_f32(attn_qkv_b_slice, {1, qkv_dim}, false);
    register_state(make_param_view_f32(attn_qkv_b_slice, {1, qkv_dim}),
                   views.attn_qkv_b);

    const LayoutSlice &attn_out_w_slice =
        cursor.next(lname(static_cast<int>(layer), "attn_out_w"));
    views.attn_out_w =
        make_state_pair_f32(attn_out_w_slice, {model_dim, model_dim}, true);
    register_state(
        make_param_view_f32(attn_out_w_slice, {model_dim, model_dim}),
        views.attn_out_w);

    const LayoutSlice &attn_out_b_slice =
        cursor.next(lname(static_cast<int>(layer), "attn_out_b"));
    views.attn_out_b =
        make_state_pair_f32(attn_out_b_slice, {1, model_dim}, false);
    register_state(make_param_view_f32(attn_out_b_slice, {1, model_dim}),
                   views.attn_out_b);

    const LayoutSlice &ln2_gamma_slice =
        cursor.next(lname(static_cast<int>(layer), "ln2_gamma"));
    views.ln2_gamma = make_state_pair_f32(ln2_gamma_slice, {1, model_dim}, false);
    register_state(make_param_view_f32(ln2_gamma_slice, {1, model_dim}),
                   views.ln2_gamma);

    const LayoutSlice &ln2_beta_slice =
        cursor.next(lname(static_cast<int>(layer), "ln2_beta"));
    views.ln2_beta = make_state_pair_f32(ln2_beta_slice, {1, model_dim}, false);
    register_state(make_param_view_f32(ln2_beta_slice, {1, model_dim}),
                   views.ln2_beta);

    const LayoutSlice &ffn_w1_slice =
        cursor.next(lname(static_cast<int>(layer), "ffn_w1"));
    views.ffn_w1 = make_state_pair_f32(ffn_w1_slice, {model_dim, ffn_dim}, true);
    register_state(make_param_view_f32(ffn_w1_slice, {model_dim, ffn_dim}),
                   views.ffn_w1);

    const LayoutSlice &ffn_b1_slice =
        cursor.next(lname(static_cast<int>(layer), "ffn_b1"));
    views.ffn_b1 = make_state_pair_f32(ffn_b1_slice, {1, ffn_dim}, false);
    register_state(make_param_view_f32(ffn_b1_slice, {1, ffn_dim}),
                   views.ffn_b1);

    const LayoutSlice &ffn_w2_slice =
        cursor.next(lname(static_cast<int>(layer), "ffn_w2"));
    views.ffn_w2 = make_state_pair_f32(ffn_w2_slice, {ffn_dim, model_dim}, true);
    register_state(make_param_view_f32(ffn_w2_slice, {ffn_dim, model_dim}),
                   views.ffn_w2);

    const LayoutSlice &ffn_b2_slice =
        cursor.next(lname(static_cast<int>(layer), "ffn_b2"));
    views.ffn_b2 = make_state_pair_f32(ffn_b2_slice, {1, model_dim}, false);
    register_state(make_param_view_f32(ffn_b2_slice, {1, model_dim}),
                   views.ffn_b2);
  }

  const LayoutSlice &lnf_gamma_slice = cursor.next("lnf_gamma");
  lnf_gamma_ = make_state_pair_f32(lnf_gamma_slice, {1, model_dim}, false);
  register_state(make_param_view_f32(lnf_gamma_slice, {1, model_dim}),
                 lnf_gamma_);

  const LayoutSlice &lnf_beta_slice = cursor.next("lnf_beta");
  lnf_beta_ = make_state_pair_f32(lnf_beta_slice, {1, model_dim}, false);
  register_state(make_param_view_f32(lnf_beta_slice, {1, model_dim}), lnf_beta_);

  const LayoutSlice &lm_head_w_slice = cursor.next("lm_head_w");
  lm_head_w_ = make_state_pair_f32(lm_head_w_slice, {model_dim, vocab_size}, true);
  register_state(make_param_view_f32(lm_head_w_slice, {model_dim, vocab_size}),
                 lm_head_w_);

  cursor.finish();
}

void AdamStateStore::register_state(const TensorView &param,
                                      const StatePair &state) {
  state_by_param_data_.emplace(param.data(), state);
}

TensorView AdamStateStore::make_param_view_f32(const LayoutSlice &s,
                                                 Shape shape) const {
  const uint64_t expected = nbytes(shape, DType::F32);
  require(expected == s.bytes,
          "shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  require(s.offset + s.bytes <= params_bytes_, "slice out of bounds: " + s.name);
  require(s.offset + s.bytes <= param_bytes_,
          "slice exceeds parameter layout: " + s.name);
  require((s.offset % alignof(float)) == 0,
          "slice offset not float-aligned: " + s.name);
  require((s.bytes % sizeof(float)) == 0,
          "slice bytes not multiple of float: " + s.name);
  void *ptr = params_base_ + s.offset;
  return TensorView(device_, DType::F32, ptr, shape);
}

TensorView AdamStateStore::make_state_view_f32(const LayoutSlice &s, Shape shape,
                                                 uint64_t state_base_offset) const {
  const uint64_t expected = nbytes(shape, DType::F32);
  require(expected == s.bytes,
          "shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  require(state_base_offset + s.offset + s.bytes <= adam_bytes_,
          "state slice out of bounds: " + s.name);
  void *ptr = adam_base_ + state_base_offset + s.offset;
  return TensorView(device_, DType::F32, ptr, shape);
}

AdamStateStore::StatePair AdamStateStore::make_state_pair_f32(
    const LayoutSlice &s, Shape shape, bool apply_weight_decay) const {
  return {make_state_view_f32(s, shape, 0),
          make_state_view_f32(s, shape, param_bytes_), apply_weight_decay};
}

std::string AdamStateStore::lname(int layer, const char *suffix) const {
  return "layer" + std::to_string(layer) + "." + suffix;
}

#undef require
