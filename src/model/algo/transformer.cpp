#include "transformer.hpp"
#include "tensor_contracts.hpp"
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

Transformer::Transformer(const Config &cfg, TensorStore &tensor_store,
                         GradientStore *gradient_store, Ops &ops,
                         ReportSink *sink)
    : cfg_(cfg),
      tensorStore_(tensor_store),
      gradientStore_(gradient_store),
      ops_(ops),
      algoConfig_(ModelAlgoConfig::from_config(cfg)),
      algoFactory_(algoConfig_),
      outputHead_(cfg_, tensorStore_, gradientStore_, ops_),
      sink_(sink) {
  layers_.reserve(cfg_.model.n_layers);
  for (uint32_t i = 0; i < cfg_.model.n_layers; ++i) {
    layers_.emplace_back(static_cast<int>(i), cfg_, tensorStore_,
                         gradientStore_, ops_, algoFactory_);
  }
  validate_contract();
}

void Transformer::set_observer(ITrainingObserver *observer) {
  observer_ = observer;
  for (auto &layer : layers_) {
    layer.set_observer(observer);
  }
  outputHead_.set_observer(observer);
}

void Transformer::set_diagnostics(TrainingDiagnosticsController *diagnostics) {
  diagnostics_ = diagnostics;
  for (auto &layer : layers_) {
    layer.set_diagnostics(diagnostics);
  }
  outputHead_.set_diagnostics(diagnostics);
}

void Transformer::validate_contract() const {
  const int64_t model_dim = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t vocab_size =
      static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t max_seq_len =
      static_cast<int64_t>(cfg_.model.max_seq_len);

  const TensorView &tok_emb = tensorStore_.param_tok_embedding();
  const TensorView &pos_emb = tensorStore_.param_pos_embedding();

  TensorContracts::validate_transformer_embedding_params(
      tok_emb, pos_emb, model_dim, vocab_size, max_seq_len, "Transformer");
}

void Transformer::forward(const TensorView &ids, TensorView &logits,
                          TensorView *last_hidden) {
  observer_->on_forward_start();
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t max_seq_len = static_cast<int64_t>(cfg_.model.max_seq_len);
  const TensorContracts::BatchSeqDims dims =
      TensorContracts::validate_ids_bs(ids, max_seq_len, "Transformer");
  const int64_t batch_size = dims.batch_size;
  const int64_t seq_len = dims.seq_len;
  TensorContracts::validate_logits_bsv(logits, batch_size, seq_len,
                                       vocab_size, "Transformer", "logits");

  const TensorView &tok_emb = tensorStore_.param_tok_embedding();
  const TensorView &pos_emb = tensorStore_.param_pos_embedding();

  TensorContracts::validate_same_device_dtype(tok_emb, logits, "Transformer",
                                              "params/logits");
  require(pos_emb.device() == logits.device(), "pos_emb/logits device mismatch");

  TensorView X = tensorStore_.temp_tr_X(batch_size, seq_len);

  ops_.embedding_lookup(tok_emb, ids, X);

  TensorView pos_slice = pos_emb.subrows(0, seq_len);
  ops_.add(X, pos_slice, X);
  cache_x0_ = X;

  TensorView report_Y = tensorStore_.temp_tr_Y(batch_size, seq_len);
  observer_->init_tensors_xy_ready(X.dim(0), X.dim(1) * X.dim(2),
                                   report_Y.dim(0),
                                   report_Y.dim(1) * report_Y.dim(2),
                                   tok_emb, pos_emb);

  for (size_t l = 0; l < layers_.size(); ++l) {
    TensorView Y = tensorStore_.temp_layer_hidden(static_cast<int>(l),
                                                    batch_size, seq_len);
    observer_->on_layer_start(static_cast<int>(l));
    layers_[l].forward(X, Y);
    observer_->on_layer_end(static_cast<int>(l));
    X = Y;
  }

  outputHead_.forward(X, logits, last_hidden);
  observer_->on_forward_end();
}

void Transformer::backward(const TensorView &ids, const TensorView &dlogits,
                           const RuntimeFlags::ProbeFlags &probe) {
  observer_->on_backward_start();
  require(ids.device() == dlogits.device(), "ids/dlogits device mismatch");
  require(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
          "ids must be I32/F32");

  const int64_t max_seq_len = static_cast<int64_t>(cfg_.model.max_seq_len);
  const TensorContracts::BatchSeqDims dims =
      TensorContracts::validate_ids_bs(ids, max_seq_len, "Transformer");
  const int64_t batch_size = dims.batch_size;
  const int64_t seq_len = dims.seq_len;
  const int64_t ids_token_rows = static_cast<int64_t>(ids.numel());
  const int64_t token_rows = static_cast<int64_t>(dlogits.numel() /
      static_cast<uint64_t>(cfg_.model.target_vocab_size));
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  require(ids_token_rows == token_rows,
          "backward ids/dlogits token-row mismatch");

  const TensorView &tok_emb = tensorStore_.param_tok_embedding();
  const TensorView &pos_emb = tensorStore_.param_pos_embedding();

  TensorContracts::validate_logits_bsv(dlogits, batch_size, seq_len,
                                       vocab_size, "Transformer", "dlogits");

  (void)probe;

  TensorView d_xlast = tensorStore_.temp_bw_d_xlast(batch_size, seq_len);
  outputHead_.backward(dlogits, d_xlast);

  TensorView d_cur = d_xlast;
  for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
    observer_->on_layer_start(l);
    TensorView d_prev = tensorStore_.temp_layer_bw_d_prev(l, batch_size, seq_len);
    layers_[static_cast<size_t>(l)].backward(d_cur, d_prev);
    observer_->on_layer_end(l);
    diagnostics_->bk_transformer_layer_d_prev(l, d_prev);
    d_cur = d_prev;
  }

  TensorView d_tok = gradientStore_->grad_for_param(tok_emb);
  TensorView d_pos = gradientStore_->grad_for_param(pos_emb);
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
