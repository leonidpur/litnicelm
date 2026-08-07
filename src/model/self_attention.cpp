#include "self_attention.hpp"

#include <utils/assert.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond), [&]() {                                                 \
    return std::string("SelfAttention: ") + std::string(msg);                  \
  })

SelfAttention::SelfAttention(int layer_index, const Config &cfg,
                             TensorFactory &tensor_factory, Ops &ops)
    : idx_(layer_index), cfg_(cfg), tensorFactory_(tensor_factory), ops_(ops) {
  validate_contract();
}

void SelfAttention::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
}

void SelfAttention::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t num_heads = static_cast<int64_t>(cfg_.model.n_heads);
  require(num_heads > 0, "n_heads must be > 0");
  require((model_dim % num_heads) == 0,
          "d_model must be divisible by n_heads");

  const TensorView &Wqkv = tensorFactory_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorFactory_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorFactory_.param_attn_out_w(idx_);
  const TensorView &bo = tensorFactory_.param_attn_out_b(idx_);

  require(Wqkv.shape().r == model_dim && Wqkv.shape().c == 3 * model_dim,
          "Wqkv must be [D, 3D]");
  require(bqkv.shape().r == 1 && bqkv.shape().c == 3 * model_dim,
          "bqkv must be [1, 3D]");
  require(Wo.shape().r == model_dim && Wo.shape().c == model_dim,
          "Wo must be [D, D]");
  require(bo.shape().r == 1 && bo.shape().c == model_dim,
          "bo must be [1, D]");

  require(Wqkv.device() == bqkv.device() && Wqkv.device() == Wo.device() &&
              Wqkv.device() == bo.device(),
          "attention parameter devices must match");
  require(Wqkv.dtype() == bqkv.dtype() && Wqkv.dtype() == Wo.dtype() &&
              Wqkv.dtype() == bo.dtype(),
          "attention parameter dtypes must match");
}

static void row_sum_f32(const TensorView &x, TensorView &out_1xC) {
  require(x.device() == Device::CPU && out_1xC.device() == Device::CPU,
          "row_sum_f32 CPU only");
  require(x.dtype() == DType::F32 && out_1xC.dtype() == DType::F32,
          "row_sum_f32 F32 only");
  require(out_1xC.shape().r == 1 && out_1xC.shape().c == x.shape().c,
          "row_sum_f32 out shape mismatch");
  const int64_t R = x.shape().r;
  const int64_t C = x.shape().c;
  for (int64_t c = 0; c < C; ++c) {
    float acc = 0.0f;
    for (int64_t r = 0; r < R; ++r) {
      acc += x.at_f32(r, c);
    }
    out_1xC.set_f32(0, c, acc);
  }
}

static void softmax_backward_rows_f32(const TensorView &softmax,
                                      const TensorView &dout,
                                      TensorView &dx) {
  require(softmax.device() == Device::CPU && dout.device() == Device::CPU &&
              dx.device() == Device::CPU,
          "softmax_backward CPU only");
  require(softmax.dtype() == DType::F32 && dout.dtype() == DType::F32 &&
              dx.dtype() == DType::F32,
          "softmax_backward F32 only");
  require(softmax.shape().r == dout.shape().r && softmax.shape().c == dout.shape().c,
          "softmax_backward shape mismatch");
  require(dx.shape().r == softmax.shape().r && dx.shape().c == softmax.shape().c,
          "softmax_backward dx shape mismatch");

  const int64_t R = softmax.shape().r;
  const int64_t C = softmax.shape().c;
  for (int64_t r = 0; r < R; ++r) {
    float dot = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
      dot += softmax.at_f32(r, c) * dout.at_f32(r, c);
    }
    for (int64_t c = 0; c < C; ++c) {
      const float s = softmax.at_f32(r, c);
      const float g = s * (dout.at_f32(r, c) - dot);
      dx.set_f32(r, c, g);
    }
  }
}

void SelfAttention::forward(const TensorView &x, TensorView &out) {
  observer_->on_attention_start(idx_);
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  const int64_t token_rows = x.shape().r;
  const int64_t model_dim = x.shape().c;
  require(model_dim == static_cast<int64_t>(cfg_.model.d_model), "x.c != d_model");
  require(out.shape().r == token_rows && out.shape().c == model_dim,
          "out must be [T, D]");

  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView &Wqkv = tensorFactory_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorFactory_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorFactory_.param_attn_out_w(idx_);
  const TensorView &bo = tensorFactory_.param_attn_out_b(idx_);

  require(Wqkv.device() == x.device() && Wo.device() == x.device(),
          "param device mismatch");
  require(Wqkv.dtype() == x.dtype() && Wo.dtype() == x.dtype(),
          "param dtype mismatch");

  TensorView qkv = tensorFactory_.temp_attn_qkv(idx_, token_rows);
  ops_.matmul(x, Wqkv, qkv);
  ops_.add_bias_rowwise(qkv, bqkv, qkv);

  TensorView Q = qkv.subcols(0, model_dim);
  TensorView K = qkv.subcols(model_dim, model_dim);
  TensorView V = qkv.subcols(2 * model_dim, model_dim);

  TensorView context = tensorFactory_.temp_attn_context(idx_, token_rows);
  ops_.fill(context, 0.0f);

  TensorView scores = tensorFactory_.temp_attn_scores(idx_, token_rows);
  TensorView weights = tensorFactory_.temp_attn_weights(idx_, token_rows);

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;

    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);

    ops_.matmul_right_transposed(Qh, Kh, scores);

    ops_.mul_scalar(scores, scale, scores);
    ops_.apply_causal_mask_inplace(scores);
    ops_.softmax_rows(scores, weights);

    TensorView head = tensorFactory_.temp_attn_head(idx_, token_rows);
    ops_.matmul(weights, Vh, head);

    TensorView context_h = context.subcols(col0, dh);
    ops_.copy(head, context_h);
  }

  ops_.matmul(context, Wo, out);
  ops_.add_bias_rowwise(out, bo, out);

  cache_x_ = x;
  cache_qkv_ = qkv;
  cache_context_ = context;
  has_cache_ = true;
  observer_->on_attention_end(idx_);
}

void SelfAttention::backward(const TensorView &dout, TensorView &dx,
                             const ParamUpdater &update_param) {
  observer_->on_attention_start(idx_);
  require(has_cache_, "backward called before forward");
  require(dout.shape().r == cache_x_.shape().r && dout.shape().c == cache_x_.shape().c,
          "dout shape mismatch");
  require(dx.shape().r == cache_x_.shape().r && dx.shape().c == cache_x_.shape().c,
          "dx shape mismatch");

  const int64_t token_rows = cache_x_.shape().r;
  const int64_t model_dim = cache_x_.shape().c;
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  const int64_t dh = model_dim / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView &Wqkv = tensorFactory_.param_attn_qkv_w(idx_);
  const TensorView &bqkv = tensorFactory_.param_attn_qkv_b(idx_);
  const TensorView &Wo = tensorFactory_.param_attn_out_w(idx_);
  const TensorView &bo = tensorFactory_.param_attn_out_b(idx_);
  TensorView dWo = tensorFactory_.temp_attn_dWo(idx_);
  ops_.matmul_left_transposed(cache_context_, dout, dWo);
  TensorView dbo = tensorFactory_.temp_attn_dbo(idx_);
  row_sum_f32(dout, dbo);

  TensorView dcontext = tensorFactory_.temp_attn_dcontext(idx_, token_rows);
  ops_.matmul_right_transposed(dout, Wo, dcontext);

  TensorView dqkv = tensorFactory_.temp_attn_dqkv(idx_, token_rows);
  ops_.fill(dqkv, 0.0f);
  TensorView dQ = dqkv.subcols(0, model_dim);
  TensorView dK = dqkv.subcols(model_dim, model_dim);
  TensorView dV = dqkv.subcols(2 * model_dim, model_dim);

  TensorView Q = cache_qkv_.subcols(0, model_dim);
  TensorView K = cache_qkv_.subcols(model_dim, model_dim);
  TensorView V = cache_qkv_.subcols(2 * model_dim, model_dim);

  TensorView KhT = tensorFactory_.temp_attn_KhT(idx_, token_rows);
  TensorView scores = tensorFactory_.temp_attn_scores(idx_, token_rows);
  TensorView weights = tensorFactory_.temp_attn_weights(idx_, token_rows);
  TensorView VhT = tensorFactory_.temp_attn_VhT(idx_, token_rows);
  TensorView dweights = tensorFactory_.temp_attn_dweights(idx_, token_rows);
  TensorView weightsT = tensorFactory_.temp_attn_weightsT(idx_, token_rows);
  TensorView dscores = tensorFactory_.temp_attn_dscores(idx_, token_rows);
  TensorView dscoresT = tensorFactory_.temp_attn_dscoresT(idx_, token_rows);

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;
    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);
    TensorView dQh = dQ.subcols(col0, dh);
    TensorView dKh = dK.subcols(col0, dh);
    TensorView dVh = dV.subcols(col0, dh);
    TensorView dhead = dcontext.subcols(col0, dh);

    ops_.transpose(Kh, KhT);
    ops_.matmul(Qh, KhT, scores);
    ops_.mul_scalar(scores, scale, scores);
    ops_.apply_causal_mask_inplace(scores);
    ops_.softmax_rows(scores, weights);

    ops_.transpose(Vh, VhT);
    ops_.matmul(dhead, VhT, dweights);
    ops_.transpose(weights, weightsT);
    ops_.matmul(weightsT, dhead, dVh);

    softmax_backward_rows_f32(weights, dweights, dscores);
    for (int64_t i = 0; i < token_rows; ++i) {
      for (int64_t j = i + 1; j < token_rows; ++j) {
        dscores.set_f32(i, j, 0.0f);
      }
    }

    ops_.matmul(dscores, Kh, dQh);
    ops_.mul_scalar(dQh, scale, dQh);

    ops_.transpose(dscores, dscoresT);
    ops_.matmul(dscoresT, Qh, dKh);
    ops_.mul_scalar(dKh, scale, dKh);
  }

  ops_.matmul_right_transposed(dqkv, Wqkv, dx);

  TensorView dWqkv = tensorFactory_.temp_attn_dWqkv(idx_);
  ops_.matmul_left_transposed(cache_x_, dqkv, dWqkv);
  TensorView dbqkv = tensorFactory_.temp_attn_dbqkv(idx_);
  row_sum_f32(dqkv, dbqkv);

  const std::string layer_prefix = "layer" + std::to_string(idx_) + ".";
  update_param(layer_prefix + "attn_qkv_w", const_cast<TensorView &>(Wqkv), dWqkv,
               true);
  update_param(layer_prefix + "attn_qkv_b", const_cast<TensorView &>(bqkv), dbqkv,
               false);
  update_param(layer_prefix + "attn_out_w", const_cast<TensorView &>(Wo), dWo,
               true);
  update_param(layer_prefix + "attn_out_b", const_cast<TensorView &>(bo), dbo,
               false);
  observer_->on_attention_end(idx_);
}

#undef require
