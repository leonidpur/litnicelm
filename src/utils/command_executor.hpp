#pragma once

#include <report_interface.hpp>

#include "cli_parser.hpp"

struct Config;

class CommandExecutor final {
public:
  static int run(const Command &cmd, const Config &cfg);
};
