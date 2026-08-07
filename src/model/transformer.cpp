#include "transformer.hpp"
#include "training_diagnostics_controller.hpp"
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

Transformer::Transformer(const Config &cfg, TensorFactory &tensor_factory,
                         GradientFactory *gradient_factory, Ops &ops,
                         ReportSink *sink)
    : cfg_(cfg),
      tensorFactory_(tensor_factory),
      gradientFactory_(gradient_factory),
      ops_(ops),
      sink_(sink) {
  layers_.reserve(cfg_.model.n_layers);
  for (uint32_t i = 0; i < cfg_.model.n_layers; ++i) {
    layers_.emplace_back(static_cast<int>(i), cfg_, tensorFactory_,
                         gradientFactory_, ops_);
  }
  validate_contract();
}

void Transformer::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
  for (auto &layer : layers_) {
    layer.set_observer(observer);
  }
}

void Transformer::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  require(diagnostics != nullptr, "diagnostics must be non-null");
  diagnostics_ = diagnostics;
  for (auto &layer : layers_) {
    layer.set_diagnostics(diagnostics);
  }
}

void Transformer::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t max_seq_len =
      static_cast<int64_t>(cfg_.model.max_seq_len);

  const TensorView &tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView &pos_emb = tensorFactory_.param_pos_embedding();
  const TensorView &lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView &lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView &lm_w = tensorFactory_.param_lm_head_w();

  require(tok_emb.dim(0) == vocab_size && tok_emb.dim(1) == model_dim,
          "tok_embedding must be [V, D]");
  require(pos_emb.dim(0) == max_seq_len && pos_emb.dim(1) == model_dim,
          "pos_embedding must be [S, D]");
  require(lnf_g.dim(0) == 1 && lnf_g.dim(1) == model_dim,
          "lnf_gamma must be [1, D]");
  require(lnf_b.dim(0) == 1 && lnf_b.dim(1) == model_dim,
          "lnf_beta must be [1, D]");
  require(lm_w.dim(0) == model_dim && lm_w.dim(1) == vocab_size,
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
  require(ids.rank() == 2, "ids must be semantic [B, S]");
  const int64_t batch_size = ids.dim(0);
  const int64_t seq_len = ids.dim(1);
  const int64_t token_rows = static_cast<int64_t>(ids.numel());
  require(batch_size > 0 && seq_len > 0, "ids must define [B, S] with positive dims");
  require(batch_size * seq_len == token_rows, "ids [B, S] must cover all tokens");

  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t max_seq_len = static_cast<int64_t>(cfg_.model.max_seq_len);
  require(seq_len <= max_seq_len,
          "sequence length S=" + std::to_string(seq_len) +
              " exceeds max_seq_len=" + std::to_string(max_seq_len));

  require(logits.rank() == 3 && logits.dim(0) == batch_size &&
              logits.dim(1) == seq_len && logits.dim(2) == vocab_size,
          "logits must be semantic [B, S, V]");

  const TensorView &tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView &pos_emb = tensorFactory_.param_pos_embedding();
  const TensorView &lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView &lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView &lm_w = tensorFactory_.param_lm_head_w();

  require(tok_emb.device() == logits.device(), "params/logits device mismatch");
  require(tok_emb.dtype() == logits.dtype(), "params/logits dtype mismatch");
  require(pos_emb.device() == logits.device(), "pos_emb device mismatch");
  require(lm_w.device() == logits.device(), "lm_head device mismatch");

  TensorView X = tensorFactory_.temp_tr_X(batch_size, seq_len);

  ops_.embedding_lookup(tok_emb, ids, X);

  TensorView pos_slice = pos_emb.subrows(0, seq_len);
  ops_.add(X, pos_slice, X);
  cache_x0_ = X;

  TensorView report_Y = tensorFactory_.temp_tr_Y(batch_size, seq_len);
  observer_->init_tensors_xy_ready(X.dim(0), X.dim(1) * X.dim(2),
                                   report_Y.dim(0),
                                   report_Y.dim(1) * report_Y.dim(2),
                                   tok_emb, pos_emb);

  for (size_t l = 0; l < layers_.size(); ++l) {
    TensorView Y = tensorFactory_.temp_layer_hidden(static_cast<int>(l),
                                                    batch_size, seq_len);
    observer_->on_layer_start(static_cast<int>(l));
    layers_[l].forward(X, Y);
    observer_->on_layer_end(static_cast<int>(l));
    X = Y;
  }

  TensorView Xn = tensorFactory_.temp_tr_Xn(batch_size, seq_len);
  ops_.layernorm(X, lnf_g, lnf_b, Xn);
  cache_x_last_ = X;
  cache_xn_ = Xn;

  if (last_hidden != nullptr) {
    require(last_hidden->rank() == 3 && last_hidden->dim(0) == batch_size &&
                last_hidden->dim(1) == seq_len && last_hidden->dim(2) == model_dim,
            "last_hidden must be semantic [B, S, D]");
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
                           const RuntimeFlags::ProbeFlags &probe) {
  observer_->on_backward_start();
  require(gradientFactory_ != nullptr, "backward requires gradient factory");
  require(diagnostics_ != nullptr, "backward requires diagnostics controller");
  require(has_cache_, "backward called before forward");
  require(ids.device() == dlogits.device(), "ids/dlogits device mismatch");
  require(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
          "ids must be I32/F32");

  require(ids.rank() == 2, "ids must be semantic [B, S]");
  const int64_t batch_size = ids.dim(0);
  const int64_t seq_len = ids.dim(1);
  const int64_t ids_token_rows = static_cast<int64_t>(ids.numel());
  const int64_t token_rows = static_cast<int64_t>(dlogits.numel() /
      static_cast<uint64_t>(cfg_.model.target_vocab_size));
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t max_seq_len = static_cast<int64_t>(cfg_.model.max_seq_len);
  require(batch_size > 0 && seq_len > 0, "ids must define [B, S] with positive dims");
  require(batch_size * seq_len == ids_token_rows, "ids [B, S] must cover all tokens");
  require(seq_len <= max_seq_len,
          "backward sequence length S=" + std::to_string(seq_len) +
              " exceeds max_seq_len=" + std::to_string(max_seq_len));
  require(ids_token_rows == token_rows,
          "backward ids/dlogits token-row mismatch");

  const TensorView &lm_w = tensorFactory_.param_lm_head_w();
  const TensorView &lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView &lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView &tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView &pos_emb = tensorFactory_.param_pos_embedding();

  require(dlogits.rank() == 3 && dlogits.dim(0) == batch_size &&
              dlogits.dim(1) == seq_len && dlogits.dim(2) == vocab_size,
          "dlogits must be semantic [B, S, V]");

  TensorView d_lm_w = gradientFactory_->grad_for_param(lm_w);
  diagnostics_->bk_transformer_dlogits(dlogits);
  ops_.matmul_left_transposed(cache_xn_, dlogits, d_lm_w);
  diagnostics_->bk_transformer_d_lm_w(d_lm_w);
  (void)probe;
  observer_->probe_output_head_ready(lm_w, d_lm_w);

  TensorView d_xn = tensorFactory_.temp_bw_d_xn(batch_size, seq_len);
  ops_.matmul_right_transposed(dlogits, lm_w, d_xn);
  diagnostics_->bk_transformer_d_xn(d_xn);

  TensorView d_xlast = tensorFactory_.temp_bw_d_xlast(batch_size, seq_len);
  TensorView d_lnf_g = gradientFactory_->grad_for_param(lnf_g);
  TensorView d_lnf_b = gradientFactory_->grad_for_param(lnf_b);
  ops_.layernorm_backward(cache_x_last_, lnf_g, d_xn, d_xlast, d_lnf_g,
                          d_lnf_b);
  diagnostics_->bk_transformer_d_xlast(d_xlast);
  diagnostics_->bk_transformer_d_lnf_g(d_lnf_g);
  diagnostics_->bk_transformer_d_lnf_b(d_lnf_b);

  TensorView d_cur = d_xlast;
  for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
    observer_->on_layer_start(l);
    TensorView d_prev = tensorFactory_.temp_layer_bw_d_prev(l, batch_size, seq_len);
    layers_[static_cast<size_t>(l)].backward(d_cur, d_prev);
    observer_->on_layer_end(l);
    diagnostics_->bk_transformer_layer_d_prev(l, d_prev);
    d_cur = d_prev;
  }

  TensorView d_tok = gradientFactory_->grad_for_param(tok_emb);
  TensorView d_pos = gradientFactory_->grad_for_param(pos_emb);
  diagnostics_->bk_transformer_d_cur_before_embeddings(d_cur);
  ops_.accumulate_embedding_grads(ids, d_cur, d_tok, d_pos);
  diagnostics_->bk_transformer_d_tok(d_tok);
  diagnostics_->bk_transformer_d_pos(d_pos);

  if (probe.embeddings && sink_ != nullptr) {
    sink_->report_probe_tensor("embeddings", "tok_embedding", tok_emb);
    sink_->report_probe_tensor("embeddings", "tok_embedding.grad", d_tok);
    sink_->report_probe_tensor("embeddings", "pos_embedding", pos_emb);
    sink_->report_probe_tensor("embeddings", "pos_embedding.grad", d_pos);
  }
  observer_->on_backward_end();
}

#undef require
