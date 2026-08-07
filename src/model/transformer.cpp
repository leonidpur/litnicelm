#include "transformer.hpp"

#include <utils/assert.hpp>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#define require(cond, msg)                                                      \
  REQUIRE_DEBUG((cond),                                                         \
                [&]() { return std::string("Transformer: ") + std::string(msg); })

struct ProbeStats {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
  uint32_t hash = 2166136261u;
};

static void hash_bytes(uint32_t &h, const void *ptr, size_t n) {
  const auto *p = reinterpret_cast<const uint8_t *>(ptr);
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint32_t>(p[i]);
    h *= 16777619u;
  }
}

static void report_if(ReportSink *sink, ReportEvent event, uint32_t step,
                      float value, const std::string &message) {
  report_utils::report_if(sink, ReportPhase::TRAINING, event, step, value,
                          message);
}

static ProbeStats probe_stats_f32(const TensorView &t) {
  require(t.device() == Device::CPU && t.dtype() == DType::F32,
          "probe_stats_f32 requires CPU F32");
  ProbeStats s;
  const int64_t R = t.shape().r;
  const int64_t C = t.shape().c;
  double sum = 0.0;
  double sum_sq = 0.0;
  bool init = false;
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      const float v = t.at_f32(r, c);
      hash_bytes(s.hash, &v, sizeof(float));
      const double x = static_cast<double>(v);
      if (!init) {
        s.min = s.max = x;
        init = true;
      } else {
        if (x < s.min) s.min = x;
        if (x > s.max) s.max = x;
      }
      sum += x;
      sum_sq += x * x;
    }
  }
  const double n = static_cast<double>(R * C);
  if (n > 0.0) {
    s.mean = sum / n;
    double var = (sum_sq / n) - (s.mean * s.mean);
    if (var < 0.0) {
      var = 0.0;
    }
    s.stddev = std::sqrt(var);
  }
  return s;
}

static void layernorm_backward_f32(const TensorView &x, const TensorView &gamma_1xC,
                                   const TensorView &dout, TensorView &dx,
                                   TensorView &dgamma_1xC,
                                   TensorView &dbeta_1xC) {
  require(x.device() == Device::CPU && gamma_1xC.device() == Device::CPU &&
              dout.device() == Device::CPU && dx.device() == Device::CPU &&
              dgamma_1xC.device() == Device::CPU &&
              dbeta_1xC.device() == Device::CPU,
          "layernorm_backward CPU only");
  require(x.dtype() == DType::F32 && gamma_1xC.dtype() == DType::F32 &&
              dout.dtype() == DType::F32 && dx.dtype() == DType::F32 &&
              dgamma_1xC.dtype() == DType::F32 && dbeta_1xC.dtype() == DType::F32,
          "layernorm_backward F32 only");
  require(gamma_1xC.shape().r == 1 && gamma_1xC.shape().c == x.shape().c,
          "gamma shape mismatch");
  require(dout.shape().r == x.shape().r && dout.shape().c == x.shape().c,
          "dout shape mismatch");
  require(dx.shape().r == x.shape().r && dx.shape().c == x.shape().c, "dx shape mismatch");
  require(dgamma_1xC.shape().r == 1 && dgamma_1xC.shape().c == x.shape().c,
          "dgamma shape mismatch");
  require(dbeta_1xC.shape().r == 1 && dbeta_1xC.shape().c == x.shape().c,
          "dbeta shape mismatch");

  const int64_t T = x.shape().r;
  const int64_t C = x.shape().c;
  const float eps = 1e-5f;
  for (int64_t c = 0; c < C; ++c) {
    dgamma_1xC.set_f32(0, c, 0.0f);
    dbeta_1xC.set_f32(0, c, 0.0f);
  }
  for (int64_t r = 0; r < T; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      mean += x.at_f32(r, c);
    }
    mean /= static_cast<double>(C);
    double var = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double d = static_cast<double>(x.at_f32(r, c)) - mean;
      var += d * d;
    }
    var /= static_cast<double>(C);
    const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

    double sum_dxhat = 0.0;
    double sum_dxhat_xhat = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double xhat = (static_cast<double>(x.at_f32(r, c)) - mean) * inv_std;
      const double g = static_cast<double>(gamma_1xC.at_f32(0, c));
      const double dyi = static_cast<double>(dout.at_f32(r, c));
      const double dxhat = dyi * g;
      sum_dxhat += dxhat;
      sum_dxhat_xhat += dxhat * xhat;
      dgamma_1xC.set_f32(0, c, dgamma_1xC.at_f32(0, c) + static_cast<float>(dyi * xhat));
      dbeta_1xC.set_f32(0, c, dbeta_1xC.at_f32(0, c) + static_cast<float>(dyi));
    }

    for (int64_t c = 0; c < C; ++c) {
      const double xhat = (static_cast<double>(x.at_f32(r, c)) - mean) * inv_std;
      const double g = static_cast<double>(gamma_1xC.at_f32(0, c));
      const double dyi = static_cast<double>(dout.at_f32(r, c));
      const double dxhat = dyi * g;
      const double n = static_cast<double>(C);
      const double dxi =
          (inv_std / n) * (n * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
      dx.set_f32(r, c, static_cast<float>(dxi));
    }
  }
}

Transformer::Transformer(const Config &cfg, TensorFactory &tensors, Ops &ops,
                         ReportSink *sink)
    : cfg_(cfg), tensorFactory_(tensors), ops_(ops), sink_(sink) {
  layers_.reserve(cfg_.model.n_layers);
  for (uint32_t i = 0; i < cfg_.model.n_layers; ++i) {
    layers_.emplace_back(static_cast<int>(i), cfg_, tensorFactory_, ops_);
  }
}

void Transformer::forward(const TensorView &ids, TensorView &logits,
                          TensorView *last_hidden) {
  const int64_t T = ids.shape().r;
  const int64_t ids_cols = ids.shape().c;
  require(ids_cols == 1 || ids_cols == 0, "ids must be [T] or [T,1]");

  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t V = static_cast<int64_t>(cfg_.model.target_vocab_size);
  const int64_t S = static_cast<int64_t>(cfg_.model.window_capacity);
  require(T <= S,
          "window length T=" + std::to_string(T) +
              " exceeds window_capacity=" + std::to_string(S) +
              ". Logic: T=ids.shape().r (row count passed to Transformer). "
              "In training, current loader flattens batches so "
              "T=batch_size*window_training.");

  require(logits.shape().r == T && logits.shape().c == V,
          "logits must be [T, vocab_size]");

  const TensorView tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView pos_emb = tensorFactory_.param_pos_embedding();
  const TensorView lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView lm_w = tensorFactory_.param_lm_head_w();

  require(tok_emb.shape().r == V && tok_emb.shape().c == D,
          "tok_embedding must be [V, D]");
  require(pos_emb.shape().r == S && pos_emb.shape().c == D,
          "pos_embedding must be [S, D]");
  require(lnf_g.shape().r == 1 && lnf_g.shape().c == D,
          "lnf_gamma must be [1, D]");
  require(lnf_b.shape().r == 1 && lnf_b.shape().c == D,
          "lnf_beta must be [1, D]");
  require(lm_w.shape().r == D && lm_w.shape().c == V,
          "lm_head_w must be [D, V]");

  require(tok_emb.device() == logits.device(), "params/logits device mismatch");
  require(tok_emb.dtype() == logits.dtype(), "params/logits dtype mismatch");
  require(pos_emb.device() == logits.device(), "pos_emb device mismatch");
  require(lm_w.device() == logits.device(), "lm_head device mismatch");

  TensorView X = tensorFactory_.temp_tr_X(T);

  ops_.embedding_lookup(tok_emb, ids, X);

  TensorView pos_slice = pos_emb.subrows(0, T);
  ops_.add(X, pos_slice, X);
  cache_x0_ = X;

  TensorView Y = tensorFactory_.temp_tr_Y(T);
  if (sink_ != nullptr) {
    sink_->init_tensors_X_Y(X.shape().r, X.shape().c, Y.shape().r, Y.shape().c,
                            tok_emb, pos_emb);
  }

  for (size_t l = 0; l < layers_.size(); ++l) {
    layers_[l].forward(X, Y);

    TensorView tmp = X;
    X = Y;
    Y = tmp;
  }

  TensorView Xn = tensorFactory_.temp_tr_Xn(T);
  ops_.layernorm(X, lnf_g, lnf_b, Xn);
  cache_x_last_ = X;
  cache_xn_ = Xn;

  if (last_hidden != nullptr) {
    require(last_hidden->shape().r == T && last_hidden->shape().c == D,
            "last_hidden must be [T, d_model]");
    require(last_hidden->device() == logits.device(),
            "last_hidden device mismatch");
    require(last_hidden->dtype() == logits.dtype(), "last_hidden dtype mismatch");
    ops_.copy(Xn, *last_hidden);
  }

  ops_.matmul(Xn, lm_w, logits);
  has_cache_ = true;
}

void Transformer::backward(const TensorView &ids, const TensorView &dlogits,
                           const ParamUpdater &update_param, bool do_probe) {
  require(has_cache_, "backward called before forward");
  require(ids.device() == dlogits.device(), "ids/dlogits device mismatch");
  require(ids.dtype() == DType::I32 || ids.dtype() == DType::F32,
          "ids must be I32/F32");

  const int64_t T = dlogits.shape().r;
  const int64_t V = dlogits.shape().c;
  const int64_t D = static_cast<int64_t>(cfg_.model.d_model);
  const int64_t S = static_cast<int64_t>(cfg_.model.window_capacity);
  require(T <= S,
          "backward window length T=" + std::to_string(T) +
              " exceeds window_capacity=" + std::to_string(S) +
              ". Logic: T=dlogits.shape().r and must match forward ids rows.");

  const TensorView lm_w = tensorFactory_.param_lm_head_w();
  const TensorView lnf_g = tensorFactory_.param_lnf_gamma();
  const TensorView lnf_b = tensorFactory_.param_lnf_beta();
  const TensorView tok_emb = tensorFactory_.param_tok_embedding();
  const TensorView pos_emb = tensorFactory_.param_pos_embedding();

  require(dlogits.shape().r == T && dlogits.shape().c == V, "dlogits shape mismatch");

  TensorView XnT = tensorFactory_.temp_bw_XnT(T);
  ops_.transpose(cache_xn_, XnT);
  TensorView d_lm_w = tensorFactory_.temp_bw_d_lm_w(T);
  ops_.matmul(XnT, dlogits, d_lm_w);
  update_param("lm_head_w", const_cast<TensorView &>(lm_w), d_lm_w, true);

  TensorView lm_wT = tensorFactory_.temp_bw_lm_wT();
  ops_.transpose(lm_w, lm_wT);
  TensorView d_xn = tensorFactory_.temp_bw_d_xn(T);
  ops_.matmul(dlogits, lm_wT, d_xn);

  TensorView d_xlast = tensorFactory_.temp_bw_d_xlast(T);
  TensorView d_lnf_g = tensorFactory_.temp_bw_d_lnf_g();
  TensorView d_lnf_b = tensorFactory_.temp_bw_d_lnf_b();
  layernorm_backward_f32(cache_x_last_, lnf_g, d_xn, d_xlast, d_lnf_g, d_lnf_b);
  update_param("lnf_gamma", const_cast<TensorView &>(lnf_g), d_lnf_g, false);
  update_param("lnf_beta", const_cast<TensorView &>(lnf_b), d_lnf_b, false);

  TensorView d_cur = d_xlast;
  for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) {
    TensorView d_prev = tensorFactory_.temp_layer_bw_d_prev(l, T);
    layers_[static_cast<size_t>(l)].backward(d_cur, d_prev, update_param);
    d_cur = d_prev;
  }

  TensorView d_tok = tensorFactory_.temp_bw_d_tok();
  ops_.fill(d_tok, 0.0f);
  TensorView d_pos = tensorFactory_.temp_bw_d_pos();
  ops_.fill(d_pos, 0.0f);

  for (int64_t t = 0; t < T; ++t) {
    const int64_t idx = (ids.dtype() == DType::I32)
                            ? static_cast<int64_t>(*reinterpret_cast<const int32_t *>(
                                  reinterpret_cast<const uint8_t *>(ids.data()) +
                                  t * ids.stride_r_bytes()))
                            : static_cast<int64_t>(ids.at_f32(t, 0));
    require(idx >= 0 && idx < V, "token id out of range");
    for (int64_t d = 0; d < D; ++d) {
      const float g = d_cur.at_f32(t, d);
      d_tok.set_f32(idx, d, d_tok.at_f32(idx, d) + g);
      d_pos.set_f32(t, d, d_pos.at_f32(t, d) + g);
    }
  }

  if (do_probe) {
    const ProbeStats s = probe_stats_f32(d_tok);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "[probe] tok_embedding.grad shape=[" << d_tok.shape().r << ","
        << d_tok.shape().c << "]"
        << " mean=" << s.mean
        << " std=" << s.stddev
        << " min=" << s.min
        << " max=" << s.max
        << " hash=0x" << std::hex << s.hash << std::dec;
    report_if(sink_, ReportEvent::STEP_COMPLETE, 0, 0.0f, oss.str());
  }

  update_param("tok_embedding", const_cast<TensorView &>(tok_emb), d_tok, true);
  update_param("pos_embedding", const_cast<TensorView &>(pos_emb), d_pos, true);
}

#undef require
