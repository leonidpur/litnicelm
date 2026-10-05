#include "program_terminal.hpp"

#include <iostream>

ProgramTerminal::ProgramTerminal(std::ostream &out) : out_(out) {}

ProgramTerminal::ProgramTerminal() : out_(std::cout) {}

void ProgramTerminal::usage(const std::string &text) { out_ << text; }

void ProgramTerminal::generated_text(const std::string &text) {
  out_ << text << "\n";
}

void ProgramTerminal::token_distribution(const std::string &table) {
  out_ << table;
}

void ProgramTerminal::loop_ready() {
  out_ << "inferloop ready. Type a prompt (or :q to quit).\n";
}

void ProgramTerminal::prompt() { out_ << "> " << std::flush; }

void ProgramTerminal::loop_ended() { out_ << "inferloop ended\n"; }
