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
                             TensorFactory &tensors, Ops &ops)
    : idx_(layer_index), cfg_(cfg), tensorFactory_(tensors), ops_(ops) {}

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
  require(x.device() == out.device(), "x/out device mismatch");
  require(x.dtype() == out.dtype(), "x/out dtype mismatch");

  const int64_t T = x.shape().r;
  const int64_t D = x.shape().c;
  require(D == static_cast<int64_t>(cfg_.model.d_model), "x.c != d_model");
  require(out.shape().r == T && out.shape().c == D, "out must be [T, D]");

  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  require(H > 0, "n_heads must be > 0");
  require((D % H) == 0, "d_model must be divisible by n_heads");
  const int64_t dh = D / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView Wqkv = tensorFactory_.param_attn_qkv_w(idx_);
  const TensorView bqkv = tensorFactory_.param_attn_qkv_b(idx_);
  const TensorView Wo = tensorFactory_.param_attn_out_w(idx_);
  const TensorView bo = tensorFactory_.param_attn_out_b(idx_);
  require(Wqkv.shape().r == D && Wqkv.shape().c == 3 * D,
          "Wqkv must be [D, 3D]");
  require(bqkv.shape().r == 1 && bqkv.shape().c == 3 * D,
          "bqkv must be [1, 3D]");
  require(Wo.shape().r == D && Wo.shape().c == D, "Wo must be [D, D]");
  require(bo.shape().r == 1 && bo.shape().c == D, "bo must be [1, D]");

  require(Wqkv.device() == x.device() && Wo.device() == x.device(),
          "param device mismatch");
  require(Wqkv.dtype() == x.dtype() && Wo.dtype() == x.dtype(),
          "param dtype mismatch");

  TensorView qkv = tensorFactory_.temp_attn_qkv(idx_, T);
  ops_.matmul(x, Wqkv, qkv);
  ops_.add_bias_rowwise(qkv, bqkv, qkv);

  TensorView Q = qkv.subcols(0, D);
  TensorView K = qkv.subcols(D, D);
  TensorView V = qkv.subcols(2 * D, D);

  TensorView context = tensorFactory_.temp_attn_context(idx_, T);
  ops_.fill(context, 0.0f);

  TensorView scores = tensorFactory_.temp_attn_scores(idx_, T);
  TensorView weights = tensorFactory_.temp_attn_weights(idx_, T);

  for (int64_t h = 0; h < H; ++h) {
    const int64_t col0 = h * dh;

    TensorView Qh = Q.subcols(col0, dh);
    TensorView Kh = K.subcols(col0, dh);
    TensorView Vh = V.subcols(col0, dh);

    ops_.matmul_transposed(Qh, Kh, scores);

    ops_.mul_scalar(scores, scale, scores);
    ops_.apply_causal_mask_inplace(scores);
    ops_.softmax_rows(scores, weights);

    TensorView head = tensorFactory_.temp_attn_head(idx_, T);
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
}

void SelfAttention::backward(const TensorView &dout, TensorView &dx,
                             const ParamUpdater &update_param) {
  require(has_cache_, "backward called before forward");
  require(dout.shape().r == cache_x_.shape().r && dout.shape().c == cache_x_.shape().c,
          "dout shape mismatch");
  require(dx.shape().r == cache_x_.shape().r && dx.shape().c == cache_x_.shape().c,
          "dx shape mismatch");

  const int64_t T = cache_x_.shape().r;
  const int64_t D = cache_x_.shape().c;
  const int64_t H = static_cast<int64_t>(cfg_.model.n_heads);
  require(H > 0, "n_heads must be > 0");
  require((D % H) == 0, "d_model must be divisible by n_heads");
  const int64_t dh = D / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  const TensorView Wqkv = tensorFactory_.param_attn_qkv_w(idx_);
  const TensorView bqkv = tensorFactory_.param_attn_qkv_b(idx_);
  const TensorView Wo = tensorFactory_.param_attn_out_w(idx_);
  const TensorView bo = tensorFactory_.param_attn_out_b(idx_);
  TensorView contextT = tensorFactory_.temp_attn_contextT(idx_, T);
  ops_.transpose(cache_context_, contextT);
  TensorView dWo = tensorFactory_.temp_attn_dWo(idx_);
  ops_.matmul(contextT, dout, dWo);
  TensorView dbo = tensorFactory_.temp_attn_dbo(idx_);
  row_sum_f32(dout, dbo);

  TensorView WoT = tensorFactory_.temp_attn_WoT(idx_);
  ops_.transpose(Wo, WoT);
  TensorView dcontext = tensorFactory_.temp_attn_dcontext(idx_, T);
  ops_.matmul(dout, WoT, dcontext);

  TensorView dqkv = tensorFactory_.temp_attn_dqkv(idx_, T);
  ops_.fill(dqkv, 0.0f);
  TensorView dQ = dqkv.subcols(0, D);
  TensorView dK = dqkv.subcols(D, D);
  TensorView dV = dqkv.subcols(2 * D, D);

  TensorView Q = cache_qkv_.subcols(0, D);
  TensorView K = cache_qkv_.subcols(D, D);
  TensorView V = cache_qkv_.subcols(2 * D, D);

  TensorView KhT = tensorFactory_.temp_attn_KhT(idx_, T);
  TensorView scores = tensorFactory_.temp_attn_scores(idx_, T);
  TensorView weights = tensorFactory_.temp_attn_weights(idx_, T);
  TensorView VhT = tensorFactory_.temp_attn_VhT(idx_, T);
  TensorView dweights = tensorFactory_.temp_attn_dweights(idx_, T);
  TensorView weightsT = tensorFactory_.temp_attn_weightsT(idx_, T);
  TensorView dscores = tensorFactory_.temp_attn_dscores(idx_, T);
  TensorView dscoresT = tensorFactory_.temp_attn_dscoresT(idx_, T);

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
    for (int64_t i = 0; i < T; ++i) {
      for (int64_t j = i + 1; j < T; ++j) {
        dscores.set_f32(i, j, 0.0f);
      }
    }

    ops_.matmul(dscores, Kh, dQh);
    ops_.mul_scalar(dQh, scale, dQh);

    ops_.transpose(dscores, dscoresT);
    ops_.matmul(dscoresT, Qh, dKh);
    ops_.mul_scalar(dKh, scale, dKh);
  }

  TensorView WqkvT = tensorFactory_.temp_attn_WqkvT(idx_);
  ops_.transpose(Wqkv, WqkvT);
  ops_.matmul(dqkv, WqkvT, dx);

  TensorView xT = tensorFactory_.temp_attn_xT(idx_, T);
  ops_.transpose(cache_x_, xT);
  TensorView dWqkv = tensorFactory_.temp_attn_dWqkv(idx_);
  ops_.matmul(xT, dqkv, dWqkv);
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
}

#undef require
