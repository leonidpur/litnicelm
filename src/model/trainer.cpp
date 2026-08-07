#include "trainer.hpp"

#include "dataset.hpp"
#include "memory/training_memory_manager.hpp"
#include "training_report_sink.hpp"
#include "training_session_controller.hpp"
#include "trainer_validation_utils.hpp"
#include <tokenizer_factory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace {
TensorView make_flat_zone_view(const ArenaView &arena, uint64_t offset_bytes,
                               uint64_t zone_bytes) {
  if ((offset_bytes % sizeof(float)) != 0 || (zone_bytes % sizeof(float)) != 0) {
    throw std::runtime_error("make_flat_zone_view: non-float-aligned zone");
  }
  if (offset_bytes + zone_bytes > arena.bytes) {
    throw std::runtime_error("make_flat_zone_view: zone out of bounds");
  }
  uint8_t *base = reinterpret_cast<uint8_t *>(arena.base);
  void *ptr = base + offset_bytes;
  const int64_t elems = static_cast<int64_t>(zone_bytes / sizeof(float));
  return TensorView(arena.device, DType::F32, ptr, Shape{1, elems});
}
} // namespace

Trainer::Trainer(const Config &cfg, TensorStore &tensor_store, Ops &ops,
                 OptimizerAdamW &opt, Transformer &transformer,
                 const ArenaView &data_arena, const ArenaView &grad_arena,
                 GradientStore &gradient_store, uint64_t decay_bytes,
                 const AdamStateView &adam_state,
                 DeviceBackend &device_backend,
                 TrainingSessionController &session_controller,
                 const RuntimeFlags &runtime_flags, TrainingReportSink *sink)
    : cfg_(cfg),
      tensorStore_(tensor_store),
      ops_(ops),
      opt_(opt),
      transformer_(transformer),
      data_arena_(data_arena),
      grad_arena_(grad_arena),
      gradientStore_(gradient_store),
      decay_bytes_(decay_bytes),
      adam_state_(adam_state),
      runtime_flags_(runtime_flags),
      sink_(sink),
      device_backend_(device_backend),
      session_controller_(session_controller),
      diagnostics_(tensor_store, ops, transformer_, gradient_store,
                   device_backend, runtime_flags_, cfg) {
  if (decay_bytes_ > data_arena_.bytes || decay_bytes_ > grad_arena_.bytes ||
      decay_bytes_ > (adam_state_.bytes / 2)) {
    throw std::runtime_error("Trainer: decay zone exceeds arena bounds");
  }
  runtime_flags_.epoch_report_every =
      (runtime_flags_.epoch_report_every == 0)
          ? cfg_.logging.epoch_report_every
          : runtime_flags_.epoch_report_every;
  runtime_flags_.epoch_report_every =
      std::max<uint32_t>(1, runtime_flags_.epoch_report_every);
  transformer_.set_observer(&session_controller_);
  transformer_.set_diagnostics(&diagnostics_);
}

void Trainer::zero_gradients() {
  TensorView grad_view = gradientStore_.full_gradient_view();
  ops_.fill(grad_view, 0.0f);
}

double Trainer::compute_global_grad_norm() const {
  return diagnostics_.compute_global_grad_norm();
}

void Trainer::clip_gradients() {
  if (cfg_.training.grad_clip <= 0.0f) {
    return;
  }
  const double grad_norm = compute_global_grad_norm();
  if (grad_norm <= 0.0) {
    return;
  }
  if (grad_norm <= static_cast<double>(cfg_.training.grad_clip)) {
    return;
  }
  const float scale = static_cast<float>(
      static_cast<double>(cfg_.training.grad_clip) / (grad_norm + 1e-12));
  TensorView grad_view = gradientStore_.full_gradient_view();
  ops_.mul_scalar(grad_view, scale, grad_view);
}

void Trainer::apply_decay_zone_gradients(uint64_t step) {
  if (decay_bytes_ == 0) {
    return;
  }
  TensorView params = make_flat_zone_view(data_arena_, 0, decay_bytes_);
  TensorView grads = make_flat_zone_view(grad_arena_, 0, decay_bytes_);
  TensorView m = make_flat_zone_view(
      ArenaView{adam_state_.base, adam_state_.bytes / 2, adam_state_.device}, 0,
      decay_bytes_);
  TensorView v = make_flat_zone_view(
      ArenaView{reinterpret_cast<uint8_t *>(adam_state_.base) + (adam_state_.bytes / 2),
                adam_state_.bytes / 2, adam_state_.device},
      0, decay_bytes_);
  opt_.step(cfg_.training, params, grads, m, v, step, true);
}

void Trainer::apply_no_decay_zone_gradients(uint64_t step) {
  const uint64_t no_decay_bytes = data_arena_.bytes - decay_bytes_;
  if (no_decay_bytes == 0) {
    return;
  }
  TensorView params = make_flat_zone_view(data_arena_, decay_bytes_, no_decay_bytes);
  TensorView grads = make_flat_zone_view(grad_arena_, decay_bytes_, no_decay_bytes);
  TensorView m = make_flat_zone_view(
      ArenaView{adam_state_.base, adam_state_.bytes / 2, adam_state_.device},
      decay_bytes_, no_decay_bytes);
  TensorView v = make_flat_zone_view(
      ArenaView{reinterpret_cast<uint8_t *>(adam_state_.base) + (adam_state_.bytes / 2),
                adam_state_.bytes / 2, adam_state_.device},
      decay_bytes_, no_decay_bytes);
  opt_.step(cfg_.training, params, grads, m, v, step, false);
}

double Trainer::train_one_batch(const TrainBatch &batch, TrainingState &state) {
  session_controller_.on_train_step_start(state.global_step);
  const int64_t token_rows = batch.token_count();
  const int64_t batch_size = batch.batch_size();
  const int64_t seq_len = batch.seq_len();
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.target_vocab_size);
  TensorView logits = tensorStore_.temp_tr_logits(batch_size, seq_len);
  session_controller_.batch_step_ready(
      static_cast<uint32_t>(batch_size),
      static_cast<uint32_t>(seq_len),
      static_cast<uint32_t>(token_rows),
      static_cast<uint32_t>(vocab_size));
  transformer_.forward(batch.ids, logits);
  diagnostics_.after_forward(logits);
  TensorView loss_scalar = tensorStore_.temp_tr_loss();
  double loss = 0.0;
  if (runtime_flags_.probe.loss) {
    ops_.cross_entropy_mean(logits, batch.targets, loss_scalar);
    diagnostics_.after_loss_scalar(loss_scalar);
    session_controller_.probe_loss_ready(loss_scalar, logits, batch.targets);
    loss = ops_.read_scalar_f32(loss_scalar);
    ops_.backward_from_logits_targets(logits, batch.targets);
    diagnostics_.after_logits_targets_backward(logits);
  } else {
    ops_.cross_entropy_mean_backward_inplace(logits, batch.targets, loss_scalar);
    diagnostics_.after_loss_scalar(loss_scalar);
    diagnostics_.after_cross_entropy_backward(logits);
    loss = ops_.read_scalar_f32(loss_scalar);
  }
  if (!std::isfinite(loss)) {
    throw std::runtime_error("Trainer::train_one_batch produced non-finite loss");
  }

  zero_gradients();
  transformer_.backward(batch.ids, logits, runtime_flags_.probe);
  const uint64_t step = state.global_step + 1;
  try {
    clip_gradients();
  } catch (const std::runtime_error &err) {
    diagnostics_.handle_nonfinite_grad_norm(
        batch, logits, err, [this]() { zero_gradients(); });
  }
  apply_decay_zone_gradients(step);
  apply_no_decay_zone_gradients(step);

  session_controller_.on_train_step_end(state.global_step, loss);

  return loss;
}

void Trainer::train(IDataLoader &loader) {
  TrainingState state{};
  session_controller_.on_training_start(state, tensorStore_,
                                        loader.steps_per_epoch(),
                                        device_backend_, sink_,
                                        data_arena_, adam_state_);

  const uint32_t total_epochs = session_controller_.total_epochs();
  for (uint32_t e = state.epoch + 1; e <= total_epochs; ++e) {
    session_controller_.on_epoch_start(e);
    loader.reset_epoch();
    TrainBatch batch{};
    uint32_t n = 0;
    double sum_loss = 0.0;
    const uint64_t steps_this_epoch = loader.steps_per_epoch();
    for (uint64_t batch_idx = 0; batch_idx < steps_this_epoch; ++batch_idx) {
      session_controller_.on_batch_start(state.global_step);
      const bool has_batch = loader.next(batch, state.global_step);
      if (!has_batch) {
        session_controller_.on_batch_end(state.global_step, 0.0);
        break;
      }
      const double loss = train_one_batch(batch, state);
      session_controller_.on_batch_end(state.global_step, loss);
      sum_loss += loss;
      n += 1;
      state.global_step += 1;
    }

    const float mean_loss =
        static_cast<float>(sum_loss / std::max<uint32_t>(1, n));

    const bool continue_training =
        session_controller_.on_epoch_end(e, mean_loss, state, device_backend_,
                                         sink_, data_arena_, adam_state_);
    if (!continue_training) {
      std::cout << "[Trainer] Early stop at epoch " << e << ": "
                << session_controller_.early_stop_message() << "\n";
      break;
    }
  }

  session_controller_.on_training_end(state, sink_);
}

void Trainer::import_vocab_size(Config &cfg, const Tokenizer &tokenizer) {
  cfg.model.target_vocab_size = static_cast<uint32_t>(tokenizer.vocab_size());
  std::cout << "[Trainer] vocab_size " << cfg.model.target_vocab_size
            << " imported from tokenizer\n";
}

int Trainer::train_entry_point(const Config &cfg, const Command &cmd) {
  //////////////////////////////
  // Validate training context and runtime initialization
  //////////////////////////////
  Config runtime_cfg = cfg;
  TrainerValidationUtils::validate_training_context(runtime_cfg);
  std::cout << "[Trainer] Training context validated.\n";
  auto tokenizer = TokenizerFactory::create(runtime_cfg, nullptr);
  Trainer::import_vocab_size(runtime_cfg, *tokenizer);
  TrainerValidationUtils::validate_vocab_contract_or_throw(runtime_cfg);
  TrainingReportSink training_sink(runtime_cfg.logging);
  TrainingSessionController session_controller(runtime_cfg, cmd, training_sink);

  session_controller.runtime_cfg_ready(runtime_cfg);

  TextDataset::early_evaluate_input(runtime_cfg);

  std::cout << "[Trainer] Training runtime initialization done.\n";
  //////////////////////////
  // Asset construction
  //////////////////////////
  std::unique_ptr<DeviceBackend> backend = DeviceBackend::create_instance(runtime_cfg);
  TrainingMemoryManager memory_manager(runtime_cfg, *backend, session_controller);
  Ops ops(*backend);
  OptimizerAdamW opt(*backend);
  Transformer transformer(runtime_cfg, memory_manager.tensor_store(),
                    &memory_manager.gradient_store(), ops, &training_sink);

  TextDataset loader(memory_manager.tensor_store(), *backend, runtime_cfg,
                     /*shuffle_blocks=*/true, &training_sink, &session_controller);
  TrainerValidationUtils::print_dataset_stats(runtime_cfg, loader);
  
  std::cout << "\n[Trainer] Engine, memory arenas, tensor store, model, optimizer, and dataset are initialized.\n\n";

  Trainer trainer(runtime_cfg, memory_manager.tensor_store(), ops, opt, transformer,
                  memory_manager.data_arena(),
                  memory_manager.grad_arena(),
                  memory_manager.gradient_store(),
                  memory_manager.param_layout().decay_bytes(),
                  memory_manager.adam_state(), *backend, session_controller,
                  cmd.runtime_flags, &training_sink);
  
  // Do it! Train the model using the dataset loader
  trainer.train(loader);
  return 0;
}
