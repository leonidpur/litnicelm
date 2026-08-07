#pragma once

#include <report_interface.hpp>
#include "report_sink.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Config;
struct LoggingConfig;
class NamedLayout;
class TensorFactory;

struct TrainingFetchReportData {
  uint64_t start_index = 0;
  uint32_t batch_size = 0;
  uint32_t seq_len = 0;
  uint64_t total_tokens = 0;
};

class TrainingReportSink final : public ReportSink {
public:
  explicit TrainingReportSink(const LoggingConfig &logging);

  void report_training_start(const Config &cfg);
  void report_epoch_complete(uint32_t epoch, float mean_loss);
  void report_training_end(uint32_t global_step);
  void report_error(const std::string &message);
  void report_fetch(const TrainingFetchReportData &data);
  void report_batch_step(uint32_t batch_cfg, uint32_t window_cfg,
                         uint32_t batch_flat_t, uint32_t logit_dim_v);
  void report_init_config(const Config &cfg, const NamedLayout &layout);
  void report_init_topology(const NamedLayout &layout, void *param_base,
                            uint64_t param_size, void *adam_base,
                            uint64_t adam_size);
  void report_tensor_factory_topology(const Config &cfg,
                                      const TensorFactory &tensors);

  void report(ReportEvent event, const std::string &message) override;
  void init_tensors_X_Y(int64_t x_rows, int64_t x_cols, int64_t y_rows,
                        int64_t y_cols, const TensorView &tok_emb,
                        const TensorView &pos_emb) override;

private:
  ConsoleSink console_;
  bool verbose_init_enabled_ = false;
  uint64_t alignment_bytes_ = 0;
  uint64_t batch_step_report_count_ = 0;
  uint64_t batch_step_report_limit_ = 1;
  uint64_t init_tensors_report_count_ = 0;
  uint64_t init_tensors_report_limit_ = 1;
};
