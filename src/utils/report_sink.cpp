#include "report_sink.hpp"

#include <config.hpp>

#include <iostream>

ConsoleSink::ConsoleSink(const LoggingConfig &logging) : logging_(logging) {}

void ConsoleSink::report(ReportEvent event, const std::string &message) {
  const bool is_bpe_line =
      message.find("[TOKENIZER]") == 0 || message.find("[VALIDATION]") == 0;
  if (is_bpe_line && !logging_.show_bpe) {
    return;
  }
  const bool is_training_line =
      message.find("[TRAINING]") == 0 ||
      message.find("[TrainingReportSink]") == 0 ||
      message.find("[EtaObserver]") == 0 ||
      message.find("[MC&CListener]") == 0;
  if (is_training_line && !logging_.show_train) {
    return;
  }
  const bool is_inference_line = message.find("[INFERENCE]") == 0;
  if (is_inference_line && !logging_.show_inference) {
    return;
  }

  if (event == ReportEvent::PROGRESS && logging_.report_every_n_steps > 1) {
    const std::string marker = " step=";
    const size_t pos = message.find(marker);
    if (pos != std::string::npos) {
      const size_t start = pos + marker.size();
      const size_t end = message.find(' ', start);
      const std::string step_s = message.substr(start, end - start);
      try {
        const uint32_t step = static_cast<uint32_t>(std::stoul(step_s));
        if (step > 0 &&
            (step % static_cast<uint32_t>(logging_.report_every_n_steps)) != 0) {
          return;
        }
      } catch (...) {
      }
    }
  }

  std::cout << message << "\n";
}
