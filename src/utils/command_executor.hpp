// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once


#include "cli_parser.hpp"

struct Config;

class CommandExecutor final {
public:
  static int run(const Command &cmd, const Config &cfg);
  // Commands that need no config.
  static int run_without_config(const Command &cmd);
};
