#include "transformer.hpp"

#include <utils/assert.hpp>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("Transformer: ") + std::string(msg); })

static void report_if(ReportSink *sink, ReportEvent event, uint32_t step,
                      float value, const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TRAINING, event, step, value,
                          message);
}

Transformer::Transformer(const Config &cfg, TensorFactory &tensor_factory, Ops &ops,
                         ReportSink *sink)
    : cfg_(cfg), tensorFactory_(tensor_factory), ops_(ops), sink_(sink) {
  layers_.reserve(cfg_.model.n_layers);
  for (uint32_t i = 0; i < cfg_.model.n_layers; ++i) {
    layers_.emplace_back(static_cast<int>(i), cfg_, tensorFactory_, ops_);
  }
  validate_contract();
}

void Transformer::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
  for (auto &layer : layers_) {
    layer.set_observer(observer);
  }
}

void Transformer::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t window_capacity =
      static_cast<int64_t>(cfg_.model.window_capacity);

  const TensorView &tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView &pos_emb = tensorFactory_.param_pos_embedding();
  const TensorView &lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView &lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView &lm_w = tensorFactory_.param_lm_head_w();

  require(tok_emb.shape().r == vocab_size && tok_emb.shape().c == model_dim,
          "tok_embedding must be [V, D]");
  require(pos_emb.shape().r == window_capacity && pos_emb.shape().c == model_dim,
          "pos_embedding must be [S, D]");
  require(lnf_g.shape().r == 1 && lnf_g.shape().c == model_dim,
          "lnf_gamma must be [1, D]");
  require(lnf_b.shape().r == 1 && lnf_b.shape().c == model_dim,
          "lnf_beta must be [1, D]");
  require(lm_w.shape().r == model_dim && lm_w.shape().c == vocab_size,
          "lm_head_w must be [D, V]");

  require(tok_emb.device() == pos_emb.device() &&
              tok_emb.device() == lnf_g.device() &&
              tok_emb.device() == lnf_b.device() &&
              tok_emb.device() == lm_w.device(),
          "transformer parameter devices must match");
  require(tok_emb.dtype() == pos_emb.dtype() &&
              tok_emb.dtype() == lnf_g.dtype() &&
              tok_emb.dtype() == lnf_b.dtype() &&
              tok_emb.dtype() == lm_w.dtype(),
          "transformer parameter dtypes must match");
}

void Transformer::forward(const TensorView &ids, TensorView &logits,
                          TensorView *last_hidden) {
  observer_->on_forward_start();
  const int64_t token_rows = ids.shape().r;
  const int64_t ids_cols = ids.shape().c;
  require(ids_cols == 1 || ids_cols == 0, "ids must be [T] or [T,1]");

  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t window_capacity = static_cast<int64_t>(cfg_.model.window_capacity);
  require(token_rows <= window_capacity,
          "window length T=" + std::to_string(token_rows) +
              " exceeds window_capacity=" + std::to_string(window_capacity) +
              ". Logic: T=ids.shape().r (row count passed to Transformer). "
              "In training, current loader flattens batches so "
              "T=batch_size*window_training.");

  require(logits.shape().r == token_rows &&
              logits.shape().c == vocab_size,
          "logits must be [T, vocab_size]");

  const TensorView &tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView &pos_emb = tensorFactory_.param_pos_embedding();
  const TensorView &lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView &lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView &lm_w = tensorFactory_.param_lm_head_w();

  require(tok_emb.device() == logits.device(), "params/logits device mismatch");
  require(tok_emb.dtype() == logits.dtype(), "params/logits dtype mismatch");
  require(pos_emb.device() == logits.device(), "pos_emb device mismatch");
  require(lm_w.device() == logits.device(), "lm_head device mismatch");

  TensorView X = tensorFactory_.temp_tr_X(token_rows);

  ops_.embedding_lookup(tok_emb, ids, X);

  TensorView pos_slice = pos_emb.subrows(0, token_rows);
  ops_.add(X, pos_slice, X);
  cache_x0_ = X;

  TensorView Y = tensorFactory_.temp_tr_Y(token_rows);
  observer_->init_tensors_xy_ready(X.shape().r, X.shape().c, Y.shape().r, Y.shape().c,
                                   tok_emb, pos_emb);

  for (size_t l = 0; l < layers_.size(); ++l) {
    observer_->on_layer_start(static_cast<int>(l));
    layers_[l].forward(X, Y);
    observer_->on_layer_end(static_cast<int>(l));

    TensorView tmp = X;
    X = Y;
    Y = tmp;
  }

  TensorView Xn = tensorFactory_.temp_tr_Xn(token_rows);
  ops_.layernorm(X, lnf_g, lnf_b, Xn);
  cache_x_last_ = X;
  cache_xn_ = Xn;

  if (last_hidden != nullptr) {
    require(last_hidden->shape().r == token_rows &&
                last_hidden->shape().c == model_dim,
            "last_hidden must be [T, d_model]");
    require(last_hidden->device() == logits.device(),
            "last_hidden device mismatch");
    require(last_hidden->dtype() == logits.dtype(), "last_hidden dtype mismatch");
    ops_.copy(Xn, *last_hidden);
  }

  ops_.matmul(Xn, lm_w, logits);
  has_cache_ = true;
  observer_->on_forward_end();
}

void Transformer::backward(const TensorView &ids, const TensorView &dlogits,
                           const ParamUpdater &update_param,
                           const RuntimeFlags::ProbeFlags &probe) {
  observer_->on_backward_start();
  require(has_cache_, "backward called before forward");
  require(ids.device() == dlogits.device(), "ids/dlogits device mismatch");
  require(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
          "ids must be I32/F32");

  const int64_t token_rows = dlogits.shape().r;
  const int64_t vocab_size = dlogits.shape().c;
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t window_capacity = static_cast<int64_t>(cfg_.model.window_capacity);
  require(token_rows <= window_capacity,
          "backward window length T=" + std::to_string(token_rows) +
              " exceeds window_capacity=" + std::to_string(window_capacity) +
              ". Logic: T=dlogits.shape().r and must match forward ids rows.");

  const TensorView &lm_w = tensorFactory_.param_lm_head_w();
  const TensorView &lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView &lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView &tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView &pos_emb = tensorFactory_.param_pos_embedding();

  require(dlogits.shape().r == token_rows &&
              dlogits.shape().c == vocab_size,
          "dlogits shape mismatch");

  TensorView XnT = tensorFactory_.temp_bw_XnT(token_rows);
  ops_.transpose(cache_xn_, XnT);
  TensorView d_lm_w = tensorFactory_.temp_bw_d_lm_w(token_rows);
  ops_.matmul(XnT, dlogits, d_lm_w);
  (void)probe;
  observer_->probe_output_head_ready(lm_w, d_lm_w);
  update_param("lm_head_w", const_cast<TensorView &>(lm_w), d_lm_w, true);

  TensorView lm_wT = tensorFactory_.temp_bw_lm_wT();
  ops_.transpose(lm_w, lm_wT);
  TensorView d_xn = tensorFactory_.temp_bw_d_xn(token_rows);
  ops_.matmul(dlogits, lm_wT, d_xn);

  TensorView d_xlast = tensorFactory_.temp_bw_d_xlast(token_rows);
  TensorView d_lnf_g = tensorFactory_.temp_bw_d_lnf_g();
  TensorView d_lnf_b = tensorFactory_.temp_bw_d_lnf_b();
  ops_.layernorm_backward(cache_x_last_, lnf_g, d_xn, d_xlast, d_lnf_g,
                          d_lnf_b);
  update_param("lnf_gamma", const_cast<TensorView &>(lnf_g), d_lnf_g, false);
  update_param("lnf_beta", const_cast<TensorView &>(lnf_b), d_lnf_b, false);

  TensorView d_cur = d_xlast;
  for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
    observer_->on_layer_start(l);
    TensorView d_prev = tensorFactory_.temp_layer_bw_d_prev(l, token_rows);
    layers_[static_cast<size_t>(l)].backward(d_cur, d_prev, update_param);
    observer_->on_layer_end(l);
    d_cur = d_prev;
  }

  TensorView d_tok = tensorFactory_.temp_bw_d_tok();
  ops_.fill(d_tok, 0.0f);
  TensorView d_pos = tensorFactory_.temp_bw_d_pos();
  ops_.fill(d_pos, 0.0f);

  for (int64_t t = 0; t < token_rows; ++t) {
    const int64_t idx = (ids.dtype() == DType::I32)
                            ? static_cast<int64_t>(*reinterpret_cast<const int32_t *>(
                                  reinterpret_cast<const uint8_t *>(ids.data()) +
                                  t * ids.stride_r_bytes()))
                            : static_cast<int64_t>(ids.at_f32(t, 0));
    require(idx >= 0 && idx < vocab_size, "token id out of range");
    for (int64_t d = 0; d < model_dim; ++d) {
      const float g = d_cur.at_f32(t, d);
      d_tok.set_f32(idx, d, d_tok.at_f32(idx, d) + g);
      d_pos.set_f32(t, d, d_pos.at_f32(t, d) + g);
    }
  }

  if (probe.embeddings && sink_ != nullptr) {
    sink_->report_probe_tensor("embeddings", "tok_embedding", tok_emb);
    sink_->report_probe_tensor("embeddings", "tok_embedding.grad", d_tok);
    sink_->report_probe_tensor("embeddings", "pos_embedding", pos_emb);
    sink_->report_probe_tensor("embeddings", "pos_embedding.grad", d_pos);
  }

  update_param("tok_embedding", const_cast<TensorView &>(tok_emb), d_tok, true);
  update_param("pos_embedding", const_cast<TensorView &>(pos_emb), d_pos, true);
  observer_->on_backward_end();
}

#undef require
