#include <inference.hpp>

#include "arena.hpp"
#include "backend/device_backend.hpp"
#include "checkpoint.hpp"
#include <config.hpp>
#include "dataset.hpp"
#include "ops.hpp"
#include "optimizer_adamw.hpp"
#include "named_layout.hpp"
#include "tensor_factory.hpp"
#include <tokenizer_factory.hpp>
#include "transformer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void validate_vocab_contract_or_throw(const Config &cfg,
                                      const Tokenizer &tokenizer) {
  const uint32_t cfg_vocab = cfg.model.target_vocab_size;
  const uint32_t tok_vocab = static_cast<uint32_t>(tokenizer.vocab_size());
  const DatasetHeader header =
      TextDataset::read_header_or_throw(cfg.tokenization.output_binary);
  const uint32_t ds_vocab = header.vocab_size;
  if (cfg_vocab != tok_vocab || cfg_vocab != ds_vocab || tok_vocab != ds_vocab) {
    throw std::runtime_error("Mismatched Vocab: Config(" +
                             std::to_string(cfg_vocab) + ") != File(" +
                             std::to_string(tok_vocab) + ") != Header(" +
                             std::to_string(ds_vocab) + ")");
  }
}

void report_if(ReportSink *sink, ReportEvent event, uint32_t step, float value,
               const std::string &message) {
  report_utils::report_if(sink, ReportPhase::INFERENCE, event, step, value,
                          message);
}

struct InferRuntime {
  Config cfg;
  std::unique_ptr<DeviceBackend> backend;
  std::unique_ptr<Tokenizer> tokenizer;
  NamedLayout param_layout;
  NamedLayout temp_layout;
  Arena param_arena;
  Arena adam_arena;
  Arena temp_arena;
  ArenaView data_view;
  AdamStateView adam_view;
  TensorFactory tensor_factory;
  Ops ops;
  OptimizerAdamW opt;
  Transformer model;
  ReportSink *sink = nullptr;

  InferRuntime(const std::string &config_path, ReportSink *sink_in)
      : cfg([&]() {
          Config loaded = Config::load_from_file(config_path);
          auto resolved_tokenizer = TokenizerFactory::create(loaded, sink_in);
          loaded.model.target_vocab_size =
              static_cast<uint32_t>(resolved_tokenizer->vocab_size());
          return loaded;
        }()),
        backend(make_device_backend(cfg)),
        tokenizer(TokenizerFactory::create(cfg, sink_in)),
        param_layout(build_param_layout(cfg)),
        temp_layout(build_temp_layout(cfg)),
        param_arena(*backend, cfg.device, param_layout.total_bytes(),
                    cfg.memory.alignment_bytes),
        adam_arena(*backend, cfg.device, param_layout.total_bytes() * 2,
                   cfg.memory.alignment_bytes),
        temp_arena(*backend, cfg.device, temp_layout.total_bytes(),
                   cfg.memory.alignment_bytes),
        data_view{param_arena.ptr(), param_arena.size_bytes(), cfg.device},
        adam_view{adam_arena.ptr(), adam_arena.size_bytes(), cfg.device},
        tensor_factory(cfg, param_layout, data_view.base, data_view.bytes,
                       data_view.device, temp_layout, temp_arena.ptr(),
                       temp_arena.size_bytes()),
        ops(cfg.device, *backend),
        opt(cfg.device),
        model(cfg, tensor_factory, ops, sink_in),
        sink(sink_in) {
    validate_vocab_contract_or_throw(cfg, *tokenizer);
  }
};

void load_weights_or_throw(InferRuntime &rt) {
  uint64_t restored_step = 0;
  uint32_t restored_epoch = 0;
  std::string ckpt_error;
  const bool ok =
      load_checkpoint(rt.cfg.paths.model_file, rt.cfg.model,
                      rt.cfg.conf_version, rt.cfg.memory.alignment_bytes,
                      *rt.backend, rt.data_view, rt.adam_view,
                      restored_step, restored_epoch, &ckpt_error);
  if (!ok) {
    std::string msg = "Failed to load checkpoint: " + rt.cfg.paths.model_file;
    if (!ckpt_error.empty()) {
      msg += " | mismatch: " + ckpt_error;
    }
    throw std::runtime_error(msg);
  }
}

int32_t argmax_last_row(const TensorView &logits) {
  const int64_t r = logits.shape().r - 1;
  const int64_t C = logits.shape().c;

  int32_t best = 0;
  float bestv = logits.at_f32(r, 0);
  for (int64_t c = 1; c < C; ++c) {
    const float v = logits.at_f32(r, c);
    if (v > bestv) {
      bestv = v;
      best = static_cast<int32_t>(c);
    }
  }
  return best;
}

std::vector<int32_t> prompt_ids(const Tokenizer &tokenizer,
                                const std::string &prompt) {
  std::vector<int32_t> ids = tokenizer.encode(prompt);
  if (ids.empty()) {
    ids.push_back(0);
  }
  return ids;
}

TensorView forward_prompt(InferRuntime &rt, const std::vector<int32_t> &ids) {
  const int64_t prompt_token_rows = static_cast<int64_t>(ids.size());
  TensorView ids_t = rt.tensor_factory.temp_infer_ids(prompt_token_rows);
  auto *p = reinterpret_cast<int32_t *>(ids_t.data());
  for (size_t i = 0; i < ids.size(); ++i) {
    p[i] = ids[i];
  }

  TensorView logits = rt.tensor_factory.temp_infer_logits(prompt_token_rows);
  rt.model.forward(ids_t, logits);
  return logits;
}

std::string decode_token_safe(const Tokenizer &tokenizer, int32_t id) {
  try {
    return tokenizer.decode({id});
  } catch (...) {
    return "<invalid>";
  }
}

std::string generate_text(InferRuntime &rt, const std::string &prompt) {
  std::vector<int32_t> ids = prompt_ids(*rt.tokenizer, prompt);

  const uint32_t max_seq = rt.cfg.model.window_capacity;
  const uint32_t max_new = rt.cfg.inference.window_inference;
  report_if(rt.sink, ReportEvent::START, 0, 0.0f,
            "Inference started: max_new=" + std::to_string(max_new));
  for (uint32_t step = 0; step < max_new && ids.size() < max_seq; ++step) {
    TensorView logits = forward_prompt(rt, ids);
    const int32_t next = argmax_last_row(logits);
    ids.push_back(next);
    report_if(rt.sink, ReportEvent::TOKEN_GEN, step + 1, static_cast<float>(next),
              "Generated token id=" + std::to_string(next));
  }

  report_if(rt.sink, ReportEvent::END, static_cast<uint32_t>(ids.size()), 0.0f,
            "Inference complete");
  return rt.tokenizer->decode(ids);
}

std::string inspect_distribution(InferRuntime &rt, const std::string &prompt) {
  constexpr size_t kTopK = 5;

  const std::vector<int32_t> ids = prompt_ids(*rt.tokenizer, prompt);
  const TensorView logits = forward_prompt(rt, ids);
  const int64_t row = logits.shape().r - 1;
  const int64_t cols = logits.shape().c;

  std::vector<float> row_logits(static_cast<size_t>(cols), 0.0f);
  float max_logit = logits.at_f32(row, 0);
  for (int64_t c = 0; c < cols; ++c) {
    const float v = logits.at_f32(row, c);
    row_logits[static_cast<size_t>(c)] = v;
    if (v > max_logit) {
      max_logit = v;
    }
  }

  std::vector<double> probs(static_cast<size_t>(cols), 0.0);
  double sum = 0.0;
  for (int64_t c = 0; c < cols; ++c) {
    const double w = std::exp(static_cast<double>(row_logits[static_cast<size_t>(c)] -
                                                  max_logit));
    probs[static_cast<size_t>(c)] = w;
    sum += w;
  }
  for (double &p : probs) {
    p /= sum;
  }

  std::vector<int32_t> order(static_cast<size_t>(cols), 0);
  for (int64_t c = 0; c < cols; ++c) {
    order[static_cast<size_t>(c)] = static_cast<int32_t>(c);
  }
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
  });

  double entropy = 0.0;
  for (double p : probs) {
    if (p > 0.0) {
      entropy -= p * std::log(p);
    }
  }

  std::ostringstream oss;
  oss << "[INFERENCE][INSPECT] Prompt=\"" << prompt << "\"\n";
  oss << "Argmax: id=" << order.front() << " token=\""
      << decode_token_safe(*rt.tokenizer, order.front()) << "\"\n";
  oss << "Entropy: " << std::fixed << std::setprecision(6) << entropy << "\n";
  oss << "Top-" << std::min(kTopK, order.size()) << " next-token distribution:\n";
  for (size_t i = 0; i < std::min(kTopK, order.size()); ++i) {
    const int32_t id = order[i];
    oss << (i + 1) << ". id=" << id << " token=\""
        << decode_token_safe(*rt.tokenizer, id) << "\""
        << " logit=" << std::fixed << std::setprecision(6)
        << row_logits[static_cast<size_t>(id)]
        << " prob=" << probs[static_cast<size_t>(id)] << "\n";
  }
  return oss.str();
}
} // namespace

int run_infer_mode(const std::string &config_path, const std::string &prompt,
                   ReportSink *sink) {
  InferRuntime rt(config_path, sink);
  load_weights_or_throw(rt);
  const std::string out = generate_text(rt, prompt);
  report_if(sink, ReportEvent::TOKEN_GEN, 0, 0.0f, out);
  return 0;
}

int run_inspect_mode(const std::string &config_path, const std::string &prompt,
                     ReportSink *sink) {
  InferRuntime rt(config_path, sink);
  load_weights_or_throw(rt);
  report_if(sink, ReportEvent::TOKEN_GEN, 0, 0.0f,
            inspect_distribution(rt, prompt));
  return 0;
}

int run_inferloop_mode(const std::string &config_path, ReportSink *sink) {
  InferRuntime rt(config_path, sink);
  load_weights_or_throw(rt);

  report_if(sink, ReportEvent::START, 0, 0.0f,
            "inferloop ready. Type a prompt (or :q to quit).");
  std::string prompt;
  while (true) {
    report_if(sink, ReportEvent::PROGRESS, 0, 0.0f, ">");
    if (!std::getline(std::cin, prompt)) {
      break;
    }
    if (prompt == ":q" || prompt == "exit" || prompt == "quit") {
      break;
    }
    report_if(sink, ReportEvent::TOKEN_GEN, 0, 0.0f, generate_text(rt, prompt));
  }
  report_if(sink, ReportEvent::END, 0, 0.0f, "inferloop ended");
  return 0;
}
