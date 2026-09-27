// Checks that config_to_yaml output loads back to the same config:
// load(path) -> dump -> load -> dump must reproduce the first dump.
//
// Usage: litnice_config_yaml_test <tmp dir> <config.yaml>...

#include "config_yaml.hpp"

#include <config.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <tmp dir> <config.yaml>...\n";
    return 2;
  }
  const std::filesystem::path tmp = argv[1];
  std::filesystem::create_directories(tmp);
  int failures = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string path = argv[i];
    try {
      Config cfg = Config::load_from_file(path);
      // Exercise fields that the checked-in configs leave at defaults.
      cfg.tracking.sinks = {"./build/liblitnice_tracking_journal.so",
                            "./build/liblitnice_tracking_runs.so"};
      cfg.tracking.metrics_every_n_steps = 50;
      cfg.tracking.options = {{"runs.dir", "runs"},
                              {"journal.path", "journal/x.txt"}};
      const std::string first = config_to_yaml(cfg);
      const std::filesystem::path dumped =
          tmp / std::filesystem::path(path).filename();
      std::ofstream(dumped) << first;
      const std::string second =
          config_to_yaml(Config::load_from_file(dumped.string()));
      const bool ok = first == second;
      std::cout << (ok ? "[PASS] " : "[FAIL] ") << path << "\n";
      if (!ok) {
        ++failures;
        std::cout << "--- first dump ---\n" << first
                  << "--- second dump ---\n" << second;
      }
    } catch (const std::exception &e) {
      std::cout << "[FAIL] " << path << ": " << e.what() << "\n";
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
