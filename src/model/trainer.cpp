#include "trainer.hpp"

#include "mean_loss.hpp"

#include "dataset.hpp"
#include <litnicelm_utils/log.hpp>
#include "model_events.hpp"
#include "memory/training_memory_manager.hpp"
#include "model_algo_factory.hpp"
#include "report_renderer.hpp"
#include "runtime_config.hpp"
#include "training_session_controller.hpp"
#include "trainer_validation_utils.hpp"
#include <tokenizer_factory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>


Trainer::Trainer(const Config &cfg, TensorStore &tensor_store, const Ops &ops,
                 OptimizerAdamW &opt, Transformer &transformer,
                 const ArenaView &data_arena, const ArenaView &grad_sum_arena,
                 GradientStore &gradient_store, AdamStateStore &adam_state_store,
                 const AdamStateView &adam_state,
                 DeviceBackend &device_backend,
                 TrainingMemoryManager &memory_manager,
                 TrainingSessionController &session_controller,
                 const RuntimeFlags &runtime_flags)
    : cfg_(cfg),
      tensorStore_(tensor_store),
      ops_(ops),
      opt_(opt),
      transformer_(transformer),
      data_arena_(data_arena),
      gradientStore_(gradient_store),
      adamStateStore_(adam_state_store),
      adam_state_(adam_state),
      runtime_flags_(runtime_flags),
      device_backend_(device_backend),
      memory_manager_(memory_manager),
      session_controller_(session_controller),
      diagnostics_(tensor_store, ops, transformer_, gradient_store,
                   device_backend, runtime_flags_, cfg),
      accumulator_(ops, cfg.training.batches_per_step,
                   gradient_store.full_gradient_view(), grad_sum_arena) {
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

void Trainer::apply_zone_gradients(DecayZone zone, uint64_t step,
                                   float learning_rate) {
  std::optional<TensorView> params = tensorStore_.param_zone_view(zone);
  if (!params) {
    return;
  }
  std::optional<TensorView> grads = gradientStore_.zone_view(zone);
  std::optional<TensorView> m = adamStateStore_.m_zone_view(zone);
  std::optional<TensorView> v = adamStateStore_.v_zone_view(zone);
  if (!grads || !m || !v) {
    throw std::runtime_error("Trainer: decay zones of parameters and their state differ");
  }
  opt_.step(cfg_.training, learning_rate, *params, *grads, *m, *v, step,
            /*apply_weight_decay=*/zone == DecayZone::Decay);
}

double Trainer::train_one_batch(const TrainBatch &batch, BatchInEpoch in_epoch,
                                TrainingPosition &training_position) {
  const bool first_of_step = session_controller_.step_starts_at(in_epoch.index);
  if (first_of_step) {
    session_controller_.on_train_step_start(training_position.optimizer_steps);
  }
  const double batch_loss = forward_backward(batch);
  accumulator_.add_batch(first_of_step, batch_loss);
  if (session_controller_.step_ends_at(in_epoch.index, in_epoch.count)) {
    const double step_loss = accumulator_.finish_step();
    apply_optimizer_step(batch, training_position);
    session_controller_.on_train_step_end(training_position.optimizer_steps, step_loss);
    training_position.optimizer_steps += 1;
  }
  return batch_loss;
}

double Trainer::forward_backward(const TrainBatch &batch) {
  const int64_t token_rows = batch.token_count();
  const int64_t batch_size = batch.batch_size();
  const int64_t seq_len = batch.seq_len();
  const int64_t vocab_size = static_cast<int64_t>(cfg_.model.vocab_size);
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
    throw std::runtime_error("Trainer::forward_backward produced non-finite loss");
  }

  zero_gradients();
  transformer_.backward(batch.ids, logits, runtime_flags_.probe);
  return loss;
}

void Trainer::apply_optimizer_step(const TrainBatch &batch,
                                   const TrainingPosition &training_position) {
  const uint64_t step = training_position.optimizer_steps + 1;
  const float learning_rate =
      session_controller_.learning_rate(training_position.optimizer_steps);
  try {
    clip_gradients();
  } catch (const std::runtime_error &err) {
    TensorView logits =
        tensorStore_.temp_tr_logits(batch.batch_size(), batch.seq_len());
    diagnostics_.handle_nonfinite_grad_norm(
        batch, logits, err, [this]() { zero_gradients(); });
  }
  apply_zone_gradients(DecayZone::Decay, step, learning_rate);
  apply_zone_gradients(DecayZone::NoDecay, step, learning_rate);
  memory_manager_.sync_compute_weights();
}

ValidationResult Trainer::evaluate(IDataLoader &val_dataset) {
  val_dataset.reset_epoch();
  TrainBatch batch{};
  MeanLoss val_loss;
  uint64_t tokens = 0;
  while (val_dataset.next(batch)) {
    tokens += static_cast<uint64_t>(batch.batch_size() * batch.seq_len());
    TensorView logits =
        tensorStore_.temp_tr_logits(batch.batch_size(), batch.seq_len());
    transformer_.forward(batch.ids, logits);
    TensorView loss_scalar = tensorStore_.temp_tr_loss();
    ops_.cross_entropy_mean(logits, batch.targets, loss_scalar);
    val_loss.add(ops_.read_scalar_f32(loss_scalar));
  }
  ValidationResult result;
  result.loss = val_loss.mean();
  result.batches = val_loss.count();
  result.tokens = tokens;
  return result;
}

void Trainer::train(IDataLoader &training_dataset, IDataLoader &val_dataset) {
  const uint64_t batches_per_epoch = training_dataset.batches_per_epoch();
  TrainingPosition training_position =
      session_controller_.restore_or_initialize(
          tensorStore_, device_backend_, data_arena_, adam_state_,
          batches_per_epoch);
  memory_manager_.sync_compute_weights();
  // Fixes the run's plan (incl. the learning-rate schedule) and notifies the
  // reporting observers.
  session_controller_.on_training_start(training_position, batches_per_epoch);

  // The controller decides when training ends: plan complete or early stop.
  if (!session_controller_.has_epochs_left(training_position)) {
    session_controller_.nothing_to_do(training_position);
  } else {
    while (true) {
      const uint32_t current_epoch = training_position.epoch + 1;
      session_controller_.on_epoch_start(current_epoch);
      training_dataset.reset_epoch();
      TrainBatch batch{};
      MeanLoss train_loss;
      const uint64_t batches_this_epoch = training_dataset.batches_per_epoch();
      for (uint64_t batch_idx = 0; batch_idx < batches_this_epoch; ++batch_idx) {
        //prepare batch and notify observers
        session_controller_.on_batch_start(batch_idx);
        const bool has_batch = training_dataset.next(batch, batch_idx);
        if (!has_batch) {
          session_controller_.on_batch_end(batch_idx, 0.0);
          break;
        }
        //train on the batch
        const double loss =
            train_one_batch(batch, {batch_idx, batches_this_epoch}, training_position);
        session_controller_.on_batch_end(batch_idx, loss);
        train_loss.add(loss);
      }
      //consolidate epoch metrics
      EpochMetrics metrics;
      metrics.train_loss = static_cast<float>(train_loss.mean());
      if (session_controller_.validation_due(current_epoch,
                                             val_dataset.batches_per_epoch())) {
        const ValidationResult validation = evaluate(val_dataset);
        metrics.val_loss = static_cast<float>(validation.loss);
        session_controller_.on_evaluate_end(current_epoch, validation,
                                            metrics.train_loss);
      }
      //notify observers; the controller decides whether training continues
      const ContinueTrainingDecision decision = session_controller_.on_epoch_end(
          current_epoch, metrics, training_position, device_backend_, data_arena_,
          adam_state_);
      if (!decision.continue_training) {
        break;
      }
    }
  }

  session_controller_.on_training_end(training_position);
}

int Trainer::train_entry_point(const Config &cfg, const Command &cmd) {
  //////////////////////////////
  // Validate config, tokenizer and dataset, and runtime initialization
  //////////////////////////////
  auto tokenizer = TokenizerFactory::create(cfg);
  const Config runtime_cfg = with_tokenizer_vocab_size(cfg, *tokenizer);
  TrainerValidationUtils::validate_config_tokenizer_and_dataset(runtime_cfg,
                                                                *tokenizer);
  ReportRenderer report_renderer;
  const ModelAlgoFactory algo(ModelAlgoConfig::from_config(runtime_cfg));
  TrainingSessionController session_controller(runtime_cfg, cmd, algo,
                                               report_renderer);

  session_controller.runtime_cfg_ready(runtime_cfg);

  const std::string dataset_path =
      TextDataset::early_evaluate_input(runtime_cfg);

  logger().log(model_events::kTrainingRuntimeInitialized);
  //////////////////////////
  // Asset construction
  //////////////////////////
  const std::unique_ptr<DeviceBackend> backend =
      DeviceBackend::create_instance(runtime_cfg);
  const PrecisionPolicy precision =
      PrecisionPolicy::create(runtime_cfg, backend->precision_variants());
  TrainingMemoryManager memory_manager(runtime_cfg, algo, *backend, precision,
                                       session_controller);
  const Ops ops(*backend);
  OptimizerAdamW opt(*backend);
  Transformer transformer(runtime_cfg, algo, memory_manager.tensor_store(),
                          &memory_manager.gradient_store(), ops);

  const DatasetHeader header = TextDataset::read_header_or_throw(dataset_path);
  const DatasetSplit split = DatasetSplit::from_config(runtime_cfg, header.num_tokens);
  TextDataset training_dataset(dataset_path, memory_manager.tensor_store(), *backend,
                     runtime_cfg, split, DatasetSplit::Side::Train,
                     /*shuffle_blocks=*/true, /*report_first_fetch=*/true,
                     &session_controller);
  TextDataset val_dataset(dataset_path, memory_manager.tensor_store(), *backend,
                       runtime_cfg, split, DatasetSplit::Side::Validation);
  TrainerValidationUtils::validate_validation_split(runtime_cfg, val_dataset);
  TrainerValidationUtils::print_dataset_stats(runtime_cfg, training_dataset);
  
  logger().log(model_events::kTrainingEngineInitialized);

  Trainer trainer(runtime_cfg, memory_manager.tensor_store(), ops, opt, transformer,
                  memory_manager.data_arena(),
                  memory_manager.grad_sum_arena(),
                  memory_manager.gradient_store(),
                  memory_manager.adam_state_store(),
                  memory_manager.adam_state(), *backend, memory_manager,
                  session_controller,
                  cmd.runtime_flags);
  
  // Do it! Train the model using the dataset loader
  trainer.train(training_dataset, val_dataset);
  logger().log(model_events::kTrainingConfigFile, cmd.config_path,
               LogFields().str("path", cmd.config_path));
  return 0;
}
