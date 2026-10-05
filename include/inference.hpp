#pragma once

#include <config.hpp>

class ProgramTerminal;

int run_infer_mode(const Config &cfg, ProgramTerminal &terminal);
int run_inspect_mode(const Config &cfg, ProgramTerminal &terminal);
int run_inferloop_mode(const Config &cfg, ProgramTerminal &terminal);
