#pragma once

#include <iosfwd>
#include <string>

// What the user asked a command for (its result) and the interaction with
// the user, written as is: no prefix, no filtering. Logs about the run go to
// logger(). One method per kind of output.
class ProgramTerminal {
public:
  explicit ProgramTerminal(std::ostream &out);
  ProgramTerminal();  // writes to std::cout

  void usage(const std::string &text);

  void generated_text(const std::string &text);
  void token_distribution(const std::string &table);
  void loop_ready();
  void prompt();  // no newline; the user types on the same line
  void loop_ended();

private:
  std::ostream &out_;
};
