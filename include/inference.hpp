#pragma once

#include <report_interface.hpp>
#include <string>

int run_infer_mode(const std::string &config_path, const std::string &prompt,
                   ReportSink *sink = nullptr);
int run_inspect_mode(const std::string &config_path, const std::string &prompt,
                     ReportSink *sink = nullptr);
int run_inferloop_mode(const std::string &config_path, ReportSink *sink = nullptr);
int run_backup_mode(const std::string &config_path, const std::string &input_path,
                    const std::string &backup_root_path);
