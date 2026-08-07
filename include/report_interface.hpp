#pragma once

#include <cstdint>
#include <string>

class TensorView;

enum class ReportPhase {
  TOKENIZER,
  TRAINING,
  INFERENCE,
  VALIDATION,
};

enum class ReportEvent {
  START,
  PROGRESS,
  STEP_COMPLETE,
  TOKEN_GEN,
  ERROR,
  END,
};

class ReportSink {
public:
  virtual ~ReportSink() = default;
  virtual void report(ReportEvent event, const std::string &message) = 0;
  virtual void init_tensors_X_Y(int64_t x_rows, int64_t x_cols, int64_t y_rows,
                                int64_t y_cols, const TensorView &tok_emb,
                                const TensorView &pos_emb) {
    (void)x_rows;
    (void)x_cols;
    (void)y_rows;
    (void)y_cols;
    (void)tok_emb;
    (void)pos_emb;
  }
};

namespace report_utils {
inline const char *phase_name(ReportPhase phase) {
  switch (phase) {
  case ReportPhase::TOKENIZER:
    return "TOKENIZER";
  case ReportPhase::TRAINING:
    return "TRAINING";
  case ReportPhase::INFERENCE:
    return "INFERENCE";
  case ReportPhase::VALIDATION:
    return "VALIDATION";
  default:
    return "UNKNOWN";
  }
}

inline const char *event_name(ReportEvent event) {
  switch (event) {
  case ReportEvent::START:
    return "START";
  case ReportEvent::PROGRESS:
    return "PROGRESS";
  case ReportEvent::STEP_COMPLETE:
    return "STEP_COMPLETE";
  case ReportEvent::TOKEN_GEN:
    return "TOKEN_GEN";
  case ReportEvent::ERROR:
    return "ERROR";
  case ReportEvent::END:
    return "END";
  default:
    return "UNKNOWN";
  }
}

inline void report_if(ReportSink *sink, ReportPhase phase, ReportEvent event,
                      uint32_t step, float value, const std::string &message) {
  if (sink == nullptr) {
    return;
  }
  sink->report(event, std::string("[") + phase_name(phase) + "][" +
                         event_name(event) + "] step=" +
                         std::to_string(step) + " val=" +
                         std::to_string(value) + " " + message);
}
} // namespace report_utils
