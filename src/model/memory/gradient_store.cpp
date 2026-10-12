#include "gradient_store.hpp"
#include "model_algo_factory.hpp"

#include <litnicelm_utils/assert.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("GradientStore: ") + std::string(msg);                 \
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

GradientStore::GradientStore(const Config &cfg,
                             const ModelAlgoFactory &algo,
                             const NamedLayout &param_layout,
                             const NamedLayout &grad_layout,
                             void *params_base, uint64_t params_bytes,
                             const ArenaView &grad_arena)
    : cfg_(cfg),
      params_base_(reinterpret_cast<uint8_t *>(params_base)),
      params_bytes_(params_bytes),
      grad_base_(reinterpret_cast<uint8_t *>(grad_arena.base)),
      grad_bytes_(grad_arena.bytes),
      device_(grad_arena.device),
      grad_layout_(grad_layout),
      full_gradient_view_(
          grad_arena.device, grad_layout.uniform_dtype(), grad_arena.base,
          Shape{static_cast<int64_t>(grad_layout.total_bytes() /
                                     dtype_size(grad_layout.uniform_dtype())),
                1}) {
  require(params_base_ != nullptr, "params_base is null");
  require(params_bytes_ > 0, "params_bytes must be > 0");
  require(grad_base_ != nullptr, "grad_arena.base is null");
  require(grad_bytes_ >= grad_layout.total_bytes(),
          "grad arena must cover the gradient layout");
  require((grad_layout.total_bytes() % dtype_size(grad_layout.uniform_dtype())) == 0,
          "gradient layout bytes must be a multiple of the element size");
  for (const LayoutSlice &s : grad_layout.slices()) {
    grad_slice_by_name_.emplace(s.name, s);
  }
  build_gradient_views(param_layout, algo);
}

std::optional<TensorView> GradientStore::zone_view(DecayZone zone) const {
  return grad_layout_.zone_view(grad_base_, device_, zone);
}

TensorView GradientStore::grad_for_param(const TensorView &param) const {
  const auto it = grad_by_param_data_.find(param.data());
  require(it != grad_by_param_data_.end(),
          "missing gradient slot for parameter");
  return it->second;
}

TensorView GradientStore::full_gradient_view() const {
  return full_gradient_view_;
}

void GradientStore::build_gradient_views(
    const NamedLayout &param_layout,
    const ModelAlgoFactory &algo) {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t ffn_dim = static_cast<int64_t>(cfg_.model.d_ff);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.vocab_size);
  const int64_t qkv_dim = 3 * model_dim;

  LayoutCursor cursor(param_layout.slices(), "parameter");

  const LayoutSlice &tok_embedding_slice = cursor.next("tok_embedding");
  register_gradient_slot(
      "tok_embedding",
      make_param_view(tok_embedding_slice, {vocab_size, model_dim}),
      make_grad_view(tok_embedding_slice, {vocab_size, model_dim}));

  if (algo.create_position_encoding(cfg_)->needs_position_table()) {
    const LayoutSlice &pos_embedding_slice = cursor.next("pos_embedding");
    register_gradient_slot(
        "pos_embedding",
        make_param_view(
            pos_embedding_slice,
            {static_cast<int64_t>(cfg_.model.max_seq_len), model_dim}),
        make_grad_view(
            pos_embedding_slice,
            {static_cast<int64_t>(cfg_.model.max_seq_len), model_dim}));
  }

  for (uint32_t layer = 0; layer < cfg_.model.n_layers; ++layer) {
    const int l = static_cast<int>(layer);

    const LayoutSlice &ln1_gamma_slice = cursor.next(lname(l, "ln1_gamma"));
    register_gradient_slot(lname(l, "ln1_gamma"),
                           make_param_view(ln1_gamma_slice, {1, model_dim}),
                           make_grad_view(ln1_gamma_slice, {1, model_dim}));

    const LayoutSlice &ln1_beta_slice = cursor.next(lname(l, "ln1_beta"));
    register_gradient_slot(lname(l, "ln1_beta"),
                           make_param_view(ln1_beta_slice, {1, model_dim}),
                           make_grad_view(ln1_beta_slice, {1, model_dim}));

    const LayoutSlice &attn_qkv_w_slice = cursor.next(lname(l, "attn_qkv_w"));
    register_gradient_slot(
        lname(l, "attn_qkv_w"),
        make_param_view(attn_qkv_w_slice, {model_dim, qkv_dim}),
        make_grad_view(attn_qkv_w_slice, {model_dim, qkv_dim}));

    const LayoutSlice &attn_qkv_b_slice = cursor.next(lname(l, "attn_qkv_b"));
    register_gradient_slot(lname(l, "attn_qkv_b"),
                           make_param_view(attn_qkv_b_slice, {1, qkv_dim}),
                           make_grad_view(attn_qkv_b_slice, {1, qkv_dim}));

    const LayoutSlice &attn_out_w_slice = cursor.next(lname(l, "attn_out_w"));
    register_gradient_slot(
        lname(l, "attn_out_w"),
        make_param_view(attn_out_w_slice, {model_dim, model_dim}),
        make_grad_view(attn_out_w_slice, {model_dim, model_dim}));

    const LayoutSlice &attn_out_b_slice = cursor.next(lname(l, "attn_out_b"));
    register_gradient_slot(lname(l, "attn_out_b"),
                           make_param_view(attn_out_b_slice, {1, model_dim}),
                           make_grad_view(attn_out_b_slice, {1, model_dim}));

    const LayoutSlice &ln2_gamma_slice = cursor.next(lname(l, "ln2_gamma"));
    register_gradient_slot(lname(l, "ln2_gamma"),
                           make_param_view(ln2_gamma_slice, {1, model_dim}),
                           make_grad_view(ln2_gamma_slice, {1, model_dim}));

    const LayoutSlice &ln2_beta_slice = cursor.next(lname(l, "ln2_beta"));
    register_gradient_slot(lname(l, "ln2_beta"),
                           make_param_view(ln2_beta_slice, {1, model_dim}),
                           make_grad_view(ln2_beta_slice, {1, model_dim}));

    const LayoutSlice &ffn_w1_slice = cursor.next(lname(l, "ffn_w1"));
    register_gradient_slot(
        lname(l, "ffn_w1"),
        make_param_view(ffn_w1_slice, {model_dim, ffn_dim}),
        make_grad_view(ffn_w1_slice, {model_dim, ffn_dim}));

    const LayoutSlice &ffn_b1_slice = cursor.next(lname(l, "ffn_b1"));
    register_gradient_slot(lname(l, "ffn_b1"),
                           make_param_view(ffn_b1_slice, {1, ffn_dim}),
                           make_grad_view(ffn_b1_slice, {1, ffn_dim}));

    const LayoutSlice &ffn_w2_slice = cursor.next(lname(l, "ffn_w2"));
    register_gradient_slot(
        lname(l, "ffn_w2"),
        make_param_view(ffn_w2_slice, {ffn_dim, model_dim}),
        make_grad_view(ffn_w2_slice, {ffn_dim, model_dim}));

    const LayoutSlice &ffn_b2_slice = cursor.next(lname(l, "ffn_b2"));
    register_gradient_slot(lname(l, "ffn_b2"),
                           make_param_view(ffn_b2_slice, {1, model_dim}),
                           make_grad_view(ffn_b2_slice, {1, model_dim}));
  }

  const LayoutSlice &lnf_gamma_slice = cursor.next("lnf_gamma");
  register_gradient_slot("lnf_gamma",
                         make_param_view(lnf_gamma_slice, {1, model_dim}),
                         make_grad_view(lnf_gamma_slice, {1, model_dim}));

  const LayoutSlice &lnf_beta_slice = cursor.next("lnf_beta");
  register_gradient_slot("lnf_beta",
                         make_param_view(lnf_beta_slice, {1, model_dim}),
                         make_grad_view(lnf_beta_slice, {1, model_dim}));

  const LayoutSlice &lm_head_w_slice = cursor.next("lm_head_w");
  register_gradient_slot(
      "lm_head_w",
      make_param_view(lm_head_w_slice, {model_dim, vocab_size}),
      make_grad_view(lm_head_w_slice, {model_dim, vocab_size}));

  cursor.finish();
}

void GradientStore::register_gradient_slot(const std::string &name,
                                             const TensorView &param,
                                             const TensorView &grad) {
  grad_by_param_data_.emplace(param.data(), grad);
  named_gradient_slots_.push_back(NamedGradientSlot{name, grad});
}

std::string GradientStore::first_nonfinite_diagnostic(
    DeviceBackend &backend) const {
  for (const auto &slot : named_gradient_slots_) {
    Tensor staged =
        host_tensor_stage::stage_to_cpu(backend, slot.grad,
                                        "GradientStore::first_nonfinite_diagnostic");
    const TensorView view = staged.view();
    for (int64_t r = 0; r < view.shape().dim(0); ++r) {
      for (int64_t c = 0; c < view.shape().dim(1); ++c) {
        const float v = view.at_f32(r, c);
        if (!std::isfinite(v)) {
          std::ostringstream oss;
          oss << "first non-finite gradient in " << slot.name << " at ["
              << r << "," << c << "] value=" << v;
          return oss.str();
        }
      }
    }
  }
  return "no named gradient slot with non-finite values was found";
}

TensorView GradientStore::make_param_view(const LayoutSlice &s,
                                         Shape shape) const {
  const uint64_t expected = nbytes(shape, s.dtype);
  require(expected == s.bytes,
          "shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  require(s.offset + s.bytes <= params_bytes_, "slice out of bounds: " + s.name);
  void *ptr = params_base_ + s.offset;
  return TensorView(device_, s.dtype, ptr, shape);
}

TensorView GradientStore::make_grad_view(const LayoutSlice &param_slice,
                                        Shape shape) const {
  const auto it = grad_slice_by_name_.find(param_slice.name);
  require(it != grad_slice_by_name_.end(),
          "no gradient slice for " + param_slice.name);
  const LayoutSlice &s = it->second;
  const uint64_t expected = nbytes(shape, s.dtype);
  require(expected == s.bytes,
          "gradient shape bytes mismatch for " + s.name + ": expected " +
              std::to_string(expected) + " got " + std::to_string(s.bytes));
  require(s.offset + s.bytes <= grad_bytes_,
          "gradient slice out of bounds: " + s.name);
  void *ptr = grad_base_ + s.offset;
  return TensorView(device_, s.dtype, ptr, shape);
}

void GradientStore::check_layer(int layer) const {
  require(layer >= 0, "layer < 0");
  require(static_cast<uint32_t>(layer) < cfg_.model.n_layers,
          "layer out of range");
}

std::string GradientStore::lname(int layer, const char *suffix) const {
  return "layer" + std::to_string(layer) + "." + suffix;
}

#undef require
