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
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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
  TensorFactory tensors;
  Ops ops;
  OptimizerAdamW opt;
  Transformer model;
  ReportSink *sink = nullptr;

  InferRuntime(const std::string &config_path, ReportSink *sink_in)
      : cfg(Config::load_from_file(config_path)),
        backend(make_device_backend(cfg.device)),
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
        tensors(cfg, param_layout, data_view.base, data_view.bytes,
                data_view.device, temp_layout, temp_arena.ptr(),
                temp_arena.size_bytes()),
        ops(cfg.device),
        opt(cfg.device),
        model(cfg, tensors, ops, sink_in),
        sink(sink_in) {
    validate_vocab_contract_or_throw(cfg, *tokenizer);
  }
};

void load_weights_or_throw(InferRuntime &rt) {
  uint64_t restored_step = 0;
  std::string ckpt_error;
  const bool ok =
      load_checkpoint(rt.cfg.paths.model_file, rt.cfg.model,
                      rt.cfg.conf_version,
                      rt.cfg.memory.alignment_bytes, rt.data_view, rt.adam_view,
                      restored_step, &ckpt_error);
  if (!ok) {
    std::string msg = "Failed to load checkpoint: " + rt.cfg.paths.model_file;
    if (!ckpt_error.empty()) {
      msg += " | mismatch: " + ckpt_error;
    }
    throw std::runtime_error(msg);
  }
}

bool is_forbidden_special_id(int64_t id) {
  return id < 3;
}

int32_t argmax_last_row(const TensorView &logits) {
  const int64_t r = logits.shape().r - 1;
  const int64_t C = logits.shape().c;
  if (C <= 3) {
    throw std::runtime_error("Inference: vocab size must be > 3 to mask special ids");
  }

  int32_t best = -1;
  float bestv = 0.0f;
  for (int64_t c = 0; c < C; ++c) {
    if (is_forbidden_special_id(c)) {
      continue;
    }
    const float v = logits.at_f32(r, c);
    if (best < 0 || v > bestv) {
      bestv = v;
      best = static_cast<int32_t>(c);
    }
  }
  return best;
}

std::string generate_text(InferRuntime &rt, const std::string &prompt) {
  std::vector<int32_t> ids = rt.tokenizer->encode(prompt);
  if (ids.empty()) {
    ids.push_back(0);
  }

  const uint32_t max_seq = rt.cfg.model.window_capacity;
  const uint32_t max_new = rt.cfg.inference.window_inference;
  report_if(rt.sink, ReportEvent::START, 0, 0.0f,
            "Inference started: max_new=" + std::to_string(max_new));
  for (uint32_t step = 0; step < max_new && ids.size() < max_seq; ++step) {
    const int64_t T = static_cast<int64_t>(ids.size());
    TensorView ids_t = rt.tensors.temp_infer_ids(T);
    auto *p = reinterpret_cast<int32_t *>(ids_t.data());
    for (size_t i = 0; i < ids.size(); ++i) {
      p[i] = ids[i];
    }

    TensorView logits = rt.tensors.temp_infer_logits(T);
    rt.model.forward(ids_t, logits);

    const int32_t next = argmax_last_row(logits);
    ids.push_back(next);
    report_if(rt.sink, ReportEvent::TOKEN_GEN, step + 1, static_cast<float>(next),
              "Generated token id=" + std::to_string(next));
  }

  report_if(rt.sink, ReportEvent::END, static_cast<uint32_t>(ids.size()), 0.0f,
            "Inference complete");
  return rt.tokenizer->decode(ids);
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
