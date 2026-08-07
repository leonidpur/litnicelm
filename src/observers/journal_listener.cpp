#include "journal_listener.hpp"

#include "model_convergence_and_checkpoint_listener.hpp"

#include <chrono>
#include <ctime>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
std::string now_timestamp_local() {
  const auto now = std::time(nullptr);
  std::tm tmv{};
#if defined(_WIN32)
  localtime_s(&tmv, &now);
#else
  localtime_r(&now, &tmv);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

void write_journal_entry(const std::string &journal_path,
                         const std::string &op_name,
                         const std::string &details) {
  if (journal_path.empty()) {
    return;
  }

  const std::filesystem::path path(journal_path);
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream journal(path, std::ios::app);
  if (!journal) {
    throw std::runtime_error("JournalListener: failed to open journal: " +
                             path.string());
  }

  journal << "[" << now_timestamp_local() << "] OPERATION: " << op_name
          << "\n\n";
  journal << details << "\n\n";
  if (!journal) {
    throw std::runtime_error("JournalListener: failed to write journal: " +
                             path.string());
  }
}

int64_t elapsed_ms(std::chrono::steady_clock::time_point start,
                   std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

std::string format_duration_ms(int64_t ms) {
  if (ms < 0) {
    ms = 0;
  }
  const int64_t total_seconds = ms / 1000;
  const int64_t hours = total_seconds / 3600;
  const int64_t minutes = (total_seconds % 3600) / 60;
  const int64_t seconds = total_seconds % 60;
  const int64_t millis = ms % 1000;

  std::ostringstream oss;
  if (hours > 0) {
    oss << hours << "h ";
  }
  if (hours > 0 || minutes > 0) {
    oss << minutes << "m ";
  }
  oss << seconds << "." << std::setw(3) << std::setfill('0') << millis << "s";
  return oss.str();
}

void append_model_config(std::ostringstream &oss, const ModelConfig &model) {
  oss << "model.max_seq_len=" << model.max_seq_len << "\n"
      << "model.n_layers=" << model.n_layers << "\n"
      << "model.n_heads=" << model.n_heads << "\n"
      << "model.d_model=" << model.d_model << "\n"
      << "model.d_ff=" << model.d_ff << "\n"
      << "model.target_vocab_size=" << model.target_vocab_size;
}

void append_training_config(std::ostringstream &oss,
                            const TrainingConfig &training) {
  oss << "training.learning_rate=" << training.learning_rate << "\n"
      << "training.beta1=" << training.beta1 << "\n"
      << "training.beta2=" << training.beta2 << "\n"
      << "training.eps=" << training.eps << "\n"
      << "training.weight_decay=" << training.weight_decay << "\n"
      << "training.incremental=" << (training.incremental ? "true" : "false") << "\n"
      << "training.dry_run=" << (training.dry_run ? "true" : "false") << "\n"
      << "training.num_epochs_train=" << training.num_epochs_train << "\n"
      << "training.num_epochs_dry_run=" << training.num_epochs_dry_run << "\n"
      << "training.save_interval_epochs=" << training.save_interval_epochs << "\n"
      << "training.grad_clip=" << training.grad_clip << "\n"
      << "training.train_seq_len=" << training.train_seq_len << "\n"
      << "training.window_stride=" << training.window_stride << "\n"
      << "training.batch_size=" << training.batch_size << "\n"
      << "training.target_loss=" << training.target_loss << "\n"
      << "training.min_delta=" << training.min_delta << "\n"
      << "training.patience_epochs=" << training.patience_epochs << "\n"
      << "training.min_epochs=" << training.min_epochs << "\n"
      << "training.stop_on_nonfinite_loss="
      << (training.stop_on_nonfinite_loss ? "true" : "false");
}
} // namespace

JournalListener::JournalListener(
    const Config &cfg, const Command &cmd,
    const ModelConvergenceAndCheckpointListener &convergence)
    : cfg_(cfg), convergence_(convergence) {
  (void)cmd;
}

void JournalListener::on_training_start(TrainingState &state,
                                        TensorStore &tensor_store,
                                        uint64_t steps_per_epoch,
                                        DeviceBackend &device_backend,
                                        ReportSink *sink,
                                        const ArenaView &data_arena,
                                        const AdamStateView &adam_state) {
  (void)state;
  (void)tensor_store;
  (void)steps_per_epoch;
  (void)device_backend;
  (void)sink;
  (void)data_arena;
  (void)adam_state;
  training_started_at_ = Clock::now();
  total_epoch_ms_ = 0;
  measured_epochs_ = 0;
  training_started_ = true;
  epoch_started_ = false;
}

void JournalListener::on_epoch_start(uint32_t epoch) {
  (void)epoch;
  epoch_started_at_ = Clock::now();
  epoch_started_ = true;
}

bool JournalListener::on_epoch_end(uint32_t epoch, float mean_loss,
                                   TrainingState &state,
                                   DeviceBackend &device_backend,
                                   ReportSink *sink,
                                   const ArenaView &data_arena,
                                   const AdamStateView &adam_state) {
  (void)epoch;
  (void)mean_loss;
  (void)state;
  (void)device_backend;
  (void)sink;
  (void)data_arena;
  (void)adam_state;
  if (epoch_started_) {
    total_epoch_ms_ += elapsed_ms(epoch_started_at_, Clock::now());
    measured_epochs_ += 1;
    epoch_started_ = false;
  }
  return true;
}

void JournalListener::on_training_end(const TrainingState &state,
                                      ReportSink *sink) {
  (void)sink;
  const bool estimate = convergence_.is_estimation_mode();
  const bool early_stopped = convergence_.should_stop();
  const int64_t total_process_ms =
      training_started_ ? elapsed_ms(training_started_at_, Clock::now()) : 0;
  const int64_t avg_epoch_ms =
      measured_epochs_ == 0 ? 0 : total_epoch_ms_ / measured_epochs_;

  std::ostringstream oss;
  oss << "Status: SUCCESS\n\n"
      << "Mode: " << (estimate ? "DRY_RUN" : "TRAIN") << "\n\n"
      << "Input corpus: " << cfg_.tokenization.input_corpus << "\n\n"
      << "Dataset: " << cfg_.tokenization.output_binary << "\n\n"
      << "Time total: " << format_duration_ms(total_process_ms)
      << " (" << total_process_ms << " ms)\n\n"
      << "Average epoch time: " << format_duration_ms(avg_epoch_ms)
      << " (" << avg_epoch_ms << " ms over " << measured_epochs_
      << " measured epoch(s))\n\n"
      << "Epochs completed: " << state.epoch << "\n\n"
      << "Global step: " << state.global_step;

  if (!estimate) {
    oss << "\n\nCheckpoint latest: " << cfg_.paths.model_file_latest
        << "\nBest checkpoint: " << convergence_.best_checkpoint_path();
  }
  if (convergence_.has_best()) {
    oss << "\nBest loss: " << convergence_.best_loss()
        << "\nBest epoch: " << convergence_.best_epoch();
  }
  if (early_stopped) {
    oss << "\nStop reason: " << convergence_.early_stop_message();
  }

  oss << "\n\nModel config:\n";
  append_model_config(oss, cfg_.model);
  oss << "\n\nTraining config:\n";
  append_training_config(oss, cfg_.training);

  write_journal_entry(cfg_.paths.journal_file,
                      estimate ? "DRY_RUN" : "TRAIN_RUN", oss.str());
}
