#include "trainer.hpp"

#include "arena.hpp"
#include "dataset.hpp"
#include "named_layout.hpp"
#include "training_report_sink.hpp"
#include "training_session_controller.hpp"
#include "trainer_validation_utils.hpp"
#include <tokenizer_factory.hpp>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
Trainer::Trainer(const Config &cfg, TensorFactory &tensor_factory, Ops &ops,
                 OptimizerAdamW &opt, Transformer &model,
                 const ArenaView &data_arena, const AdamStateView &adam_state,
                 DeviceBackend &device_backend, const Command &cmd,
                 const RuntimeFlags &runtime_flags, TrainingReportSink *sink)
    : cfg_(cfg),
      tensorFactory_(tensor_factory),
      ops_(ops),
      opt_(opt),
      model_(model),
      data_arena_(data_arena),
      adam_state_(adam_state),
      runtime_flags_(runtime_flags),
      sink_(sink),
      device_backend_(device_backend),
      session_controller_(cfg_, cmd) {
  runtime_flags_.epoch_report_every =
      (runtime_flags_.epoch_report_every == 0)
          ? cfg_.logging.epoch_report_every
          : runtime_flags_.epoch_report_every;
  runtime_flags_.epoch_report_every =
      std::max<uint32_t>(1, runtime_flags_.epoch_report_every);
}

void Trainer::on_gradient_ready(const std::string &name, TensorView &param,
                                const TensorView &grad, bool is_row_sparse,
                                uint64_t step) {
  (void)is_row_sparse;

  const NamedLayout param_layout = build_param_layout(cfg_);
  const LayoutSlice *slice = param_layout.find(name);
  if (slice == nullptr) {
    throw std::runtime_error("Trainer::on_gradient_ready: missing parameter slice for " +
                             name);
  }

  uint8_t *adam_base = reinterpret_cast<uint8_t *>(adam_state_.base);
  float *m_ptr = reinterpret_cast<float *>(adam_base + slice->offset);
  float *v_ptr =
      reinterpret_cast<float *>(adam_base + param_layout.total_bytes() + slice->offset);
  TensorView m(param.device(), param.dtype(), m_ptr, param.shape());
  TensorView v(param.device(), param.dtype(), v_ptr, param.shape());

  bool use_wd = (name.find("_b") == std::string::npos &&
                 name.find("_beta") == std::string::npos &&
                 name.find("_gamma") == std::string::npos);
  /*
  if (sink_ != nullptr && name == "lm_head_w") {
    sink_->report_optimizer_state(0, name, param, grad, m, v, step, use_wd);
  }
    */
  opt_.step(cfg_.training, param, grad, m, v, step, use_wd);
  /*
  if (sink_ != nullptr && name == "lm_head_w") {
    sink_->report_optimizer_state(1, name, param, grad, m, v, step, use_wd);
  }*/
}

double Trainer::train_one_batch(const TrainBatch &batch, TrainingState &state) {
  const int64_t token_rows = batch.ids.shape().r;
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  TensorView logits = tensorFactory_.temp_tr_logits(token_rows);
  if (sink_ != nullptr) {
    sink_->report_batch_step(
        static_cast<uint32_t>(cfg_.training.batch_size),
        static_cast<uint32_t>(cfg_.training.window_training),
        static_cast<uint32_t>(token_rows),
        static_cast<uint32_t>(vocab_size));
  }
  model_.forward(batch.ids, logits);
  TensorView loss_scalar = tensorFactory_.temp_tr_loss();
  ops_.cross_entropy_mean(logits, batch.targets, loss_scalar);
  if (runtime_flags_.probe.loss && sink_ != nullptr) {
    sink_->report_probe_loss(loss_scalar, logits, batch.targets);
  }
  const double loss = ops_.read_scalar_f32(loss_scalar);

  ops_.backward_from_logits_targets(logits, batch.targets);

  auto updater = [&](const std::string &name, TensorView &param,
                     const TensorView &grad, bool sparse) {
    this->on_gradient_ready(name, param, grad, sparse, state.global_step + 1);
  };
  model_.backward(batch.ids, logits, updater, runtime_flags_.probe);

  return loss;
}

void Trainer::train(IDataLoader &loader) {
  TrainingState state{};
  session_controller_.training_loop_start(state, tensorFactory_,
                                          loader.steps_per_epoch(),
                                          device_backend_, sink_,
                                          data_arena_, adam_state_);

  for (uint32_t e = state.epoch + 1; session_controller_.should_continue(e); ++e) {
    session_controller_.on_epoch_start(e);
    loader.reset_epoch();
    TrainBatch batch{};
    uint32_t n = 0;
    double sum_loss = 0.0;
    //std::cout << "\n===\n" << std::flush;
    while (loader.next(batch)) {
      //if(state.global_step%33==0)
      //  std::cout << "@" << std::flush;
      const double loss = train_one_batch(batch, state);
      sum_loss += loss;
      n += 1;
      //if(state.global_step%33==0)
      //  std::cout << "#" << std::flush;
      state.global_step += 1;
    }

    const float mean_loss =
        static_cast<float>(sum_loss / std::max<uint32_t>(1, n));
    
    session_controller_.on_epoch_end(e, mean_loss, state,
                                     runtime_flags_.epoch_report_every,
                                     device_backend_, sink_,
                                     data_arena_, adam_state_);
  }

  session_controller_.training_loop_end(state, sink_);
}

int train_entry_point(const Config &cfg, const Command &cmd) {
  Config runtime_cfg = cfg;
  TrainerValidationUtils::validate_training_context(runtime_cfg);
  std::cout << "[Trainer] Training context validated.\n";
  auto tokenizer = TokenizerFactory::create(runtime_cfg, nullptr);
  runtime_cfg.model.target_vocab_size = static_cast<uint32_t>(tokenizer->vocab_size());
  TrainerValidationUtils::validate_vocab_contract_or_throw(runtime_cfg);
  TrainingReportSink training_sink(runtime_cfg.logging);

  training_sink.report_training_start(runtime_cfg);

  const std::string selected_input_path = runtime_cfg.tokenization.output_binary;
  std::cout << "[Trainer] Dataset file to load: " << selected_input_path << "\n";
  if (selected_input_path.empty()) {
    throw std::runtime_error("run_train_mode: tokenization.output_binary is required");
  }

  std::cout << "[Trainer] Training runtime initialization done.\n";
  //////////////////////////
  // Asset construction
  //////////////////////////
  std::unique_ptr<DeviceBackend> backend = make_device_backend(runtime_cfg);

  NamedLayout param_layout = build_param_layout(runtime_cfg);
  NamedLayout temp_layout = build_temp_layout(runtime_cfg);
  training_sink.report_init_config(runtime_cfg, param_layout, temp_layout);
  
  const uint64_t param_bytes = param_layout.total_bytes();
  Arena param_arena(*backend, runtime_cfg.device, param_bytes,
                    runtime_cfg.memory.alignment_bytes);
  
  Arena adam_arena(*backend, runtime_cfg.device, param_bytes * 2,
                   runtime_cfg.memory.alignment_bytes);
  
  const uint64_t temp_bytes = temp_layout.total_bytes();
  Arena temp_arena(*backend, runtime_cfg.device, temp_bytes,
                   runtime_cfg.memory.alignment_bytes);
  
  
  ArenaView data_view{param_arena.ptr(), param_arena.size_bytes(), runtime_cfg.device};
  AdamStateView adam_view{adam_arena.ptr(), adam_arena.size_bytes(), runtime_cfg.device};
  
  training_sink.report_init_topology(param_layout, data_view.base, data_view.bytes,
                                     adam_view.base, adam_view.bytes,
                                     temp_arena.ptr(), temp_arena.size_bytes());

  TensorFactory tensor_factory(runtime_cfg, param_layout, data_view.base,
                               data_view.bytes, data_view.device, temp_layout,
                               temp_arena.ptr(), temp_arena.size_bytes());
  
  training_sink.report_tensor_factory_topology(runtime_cfg, tensor_factory);
  Ops ops(runtime_cfg.device, *backend);
  OptimizerAdamW opt(runtime_cfg.device);
  Transformer model(runtime_cfg, tensor_factory, ops, &training_sink);

  TextDataset loader(tensor_factory, selected_input_path, Device::CPU,
                     runtime_cfg.training.window_training,
                     runtime_cfg.training.batch_size,
                     /*shuffle_blocks=*/true, &training_sink);
  TrainerValidationUtils::print_dataset_stats(runtime_cfg, loader);
  std::cout << "\n[Trainer] Engine, memory arenas, tensor factory, model, optimizer, and dataset are initialized.\n\n";

  Trainer trainer(runtime_cfg, tensor_factory, ops, opt, model, data_view, adam_view, *backend,
                  cmd, cmd.runtime_flags, &training_sink);
  trainer.train(loader);

  return 0;
}
