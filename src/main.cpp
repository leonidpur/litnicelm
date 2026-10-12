#include <config.hpp>
#include "cli_parser.hpp"
#include "command_executor.hpp"
#include <litnicelm_utils/log.hpp>
#include "main_events.hpp"
#include "program_terminal.hpp"
#include <cstdlib>
#include <exception>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    if (CliParser::wants_help(argc, argv)) {
      ProgramTerminal().usage(CliParser::usage_text());
      return 0;
    }

    // 1. Path from CLI
    const Command cmd = CliParser::parse(argc, argv);
    if (cmd.config_path.empty()) {
      throw std::runtime_error("--config <config.yaml> is required");
    }
    logger().log(main_events::kCliParsed, cmd.config_path);

    // 2. YAML Parse
    Config cfg = Config::load_from_file(cmd.config_path);
    logger().log(main_events::kConfigLoaded);

    // 3. ENV Overrides (Explicit check)
    if (const char *p = std::getenv("ENV_PREFIX")) {
      logger().log(main_events::kEnvOverridesApplied, p);
      cfg.apply_env_overrides(p);
    } else {
      logger().log(main_events::kEnvOverridesSkipped);
    }

    // 4. CLI Overrides
    logger().log(main_events::kCommandOverridesApplied,
                 cfg.apply_command_overrides(cmd));

    // 5. Basic Integrity only
    cfg.validate_base();

    // Hand over to the executor - expertise starts here
    return CommandExecutor::run(cmd, cfg);
  } catch (const std::exception &e) {
    logger().log(main_events::kFatalError, e.what());
    return 1;
  }
}
