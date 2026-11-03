// Copyright 2026 Leonid Purgalin
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// https://github.com/leonidpur/litnicelm

#pragma once

#include <config.hpp>

class ProgramTerminal;

int run_infer_mode(const Config &cfg, ProgramTerminal &terminal);
int run_inspect_mode(const Config &cfg, ProgramTerminal &terminal);
int run_inferloop_mode(const Config &cfg, ProgramTerminal &terminal);
