#include <config.hpp>
#include "cli_parser.hpp"
#include "command_executor.hpp"
#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
  try {
    // 1. Path from CLI
    const Command cmd = CliParser::parse(argc, argv);
    if (cmd.config_path.empty()) {
      throw std::runtime_error("--config <config.yaml> is required");
    }
    std::cout << "[Step 1] CLI parsed. Config path: " << cmd.config_path << "\n";

    // 2. YAML Parse
    Config cfg = Config::load_from_file(cmd.config_path);
    std::cout << "[Step 2] Base YAML loaded.\n";

    // 3. ENV Overrides (Explicit check)
    if (const char *p = std::getenv("ENV_PREFIX")) {
      std::cout << "[Step 3] Applying ENV overrides with prefix: " << p << "\n";
      cfg.apply_env_overrides(p);
    } else {
      std::cout << "[Step 3] ENV_PREFIX not set. No ENV overrides applied.\n";
    }

    // 4. CLI Overrides
    std::cout << cfg.apply_command_overrides(cmd) << "\n";

    // 5. Basic Integrity only
    cfg.validate_base();

    // Hand over to the executor - expertise starts here
    return CommandExecutor::run(cmd, cfg);
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << "\n";
    return 1;
  }
}
