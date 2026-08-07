#pragma once

#include <types.hpp>

class CliParser final {
public:
  static bool wants_help(int argc, char **argv);
  static void print_usage();
  static Command parse(int argc, char **argv);
};
