#pragma once

#include <config.hpp>
#include <report_interface.hpp>

int run_infer_mode(const Config &cfg, ReportSink *sink = nullptr);
int run_inspect_mode(const Config &cfg, ReportSink *sink = nullptr);
int run_inferloop_mode(const Config &cfg, ReportSink *sink = nullptr);
