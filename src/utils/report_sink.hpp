#ifndef REPORT_SINK_HPP
#define REPORT_SINK_HPP

#include <report_interface.hpp>

struct Config;
struct LoggingConfig;

class ConsoleSink final : public ReportSink {
public:
  explicit ConsoleSink(const LoggingConfig &logging);
  void report(ReportEvent event, const std::string &message) override;

private:
  const LoggingConfig &logging_;
};

#endif
