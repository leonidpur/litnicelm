#pragma once

#include <report_interface.hpp>
#include <string>

int run_infer_mode(const std::string &config_path, const std::string &prompt,
                   ReportSink *sink = nullptr);
int run_inferloop_mode(const std::string &config_path, ReportSink *sink = nullptr);
