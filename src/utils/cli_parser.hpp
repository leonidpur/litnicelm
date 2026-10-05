#pragma once

#include <types.hpp>

#include <string>

class CliParser final {
public:
  static bool wants_help(int argc, char **argv);
  static std::string usage_text();
  static Command parse(int argc, char **argv);
};
