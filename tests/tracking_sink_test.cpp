// Checks ExperimentTrackingSink plugin loading and event fan-out using the
// fake tracking plugins.
//
// Usage: litnice_tracking_sink_test <fake.so> <fake2.so> <fake_bad_abi.so> <tmp dir>

#include "experiment_tracking_sink.hpp"

#include <config.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

std::vector<std::string> read_lines(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  for (std::string line; std::getline(in, line);) {
    lines.push_back(line);
  }
  return lines;
}

bool contains(const std::vector<std::string> &lines, const std::string &s) {
  for (const auto &line : lines) {
    if (line == s) {
      return true;
    }
  }
  return false;
}

size_t count_prefix(const std::vector<std::string> &lines,
                    const std::string &prefix) {
  size_t n = 0;
  for (const auto &line : lines) {
    n += line.rfind(prefix, 0) == 0 ? 1 : 0;
  }
  return n;
}

// Runs fn and returns the exception message, or "" when nothing was thrown.
std::string thrown_message(const std::function<void()> &fn) {
  try {
    fn();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

Config base_config() {
  Config cfg{};
  cfg.paths.journal_file = "";
  return cfg;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " <fake.so> <fake2.so> <fake_bad_abi.so> <tmp dir>\n";
    return 2;
  }
  const std::string fake = argv[1];
  const std::string fake2 = argv[2];
  const std::string bad_abi = argv[3];
  const std::filesystem::path tmp = argv[4];
  std::filesystem::create_directories(tmp);
  const std::string log1 = (tmp / "fake.log").string();
  const std::string log2 = (tmp / "fake2.log").string();
  std::filesystem::remove(log1);
  std::filesystem::remove(log2);

  {
    Config cfg = base_config();
    cfg.tracking.sinks = {fake, fake2};
    cfg.tracking.options = {{"fake.log", log1},
                            {"fake.fail_on", "2"},
                            {"fake2.log", log2}};
    ExperimentTrackingSink tracking(cfg, "test");
    for (int i = 0; i < 4; ++i) {
      tracking.emit(TrackingEventBuilder(TRACKING_EVENT_METRICS, "epoch")
                        .epoch(static_cast<uint32_t>(i))
                        .f64("train_loss", 1.0 / (i + 1)));
    }
    check(tracking.active_sinks() == 1,
          "failing sink is disabled, the other stays active");
  }
  const auto lines1 = read_lines(log1);
  const auto lines2 = read_lines(log2);
  check(count_prefix(lines2, "EVENT epoch fields=1") == 4,
        "healthy sink receives every event");
  check(count_prefix(lines1, "EVENT ") == 1,
        "failing sink stops receiving events after its failure");
  check(contains(lines1, "OPT log") && contains(lines1, "OPT fail_on") &&
            contains(lines1, "OPT context.journal_file"),
        "sink receives its own options (prefix stripped) and context.*");
  check(!contains(lines1, "OPT fake2.log") && !contains(lines2, "OPT fail_on"),
        "sink does not receive other sinks' options");

  {
    Config cfg = base_config();
    cfg.tracking.sinks = {bad_abi};
    const std::string msg =
        thrown_message([&] { ExperimentTrackingSink t(cfg, "test"); });
    check(msg.find("ABI mismatch") != std::string::npos,
          "wrong ABI version is rejected: " + msg);
  }
  {
    Config cfg = base_config();
    cfg.tracking.sinks = {(tmp / "missing.so").string()};
    const std::string msg =
        thrown_message([&] { ExperimentTrackingSink t(cfg, "test"); });
    check(msg.find("failed to load") != std::string::npos,
          "missing library is rejected");
  }
  {
    Config cfg = base_config();
    cfg.tracking.sinks = {fake2, fake};
    cfg.tracking.options = {{"fake.fail_create", "1"}};
    const std::string msg =
        thrown_message([&] { ExperimentTrackingSink t(cfg, "test"); });
    check(msg.find("create failed") != std::string::npos,
          "create failure is rejected (after another sink loaded)");
  }
  {
    Config cfg = base_config();
    ExperimentTrackingSink tracking(cfg, "test");
    check(tracking.active_sinks() == 0,
          "no sinks and no journal_file loads nothing");
  }

  std::cout << (g_failures == 0 ? "All tracking sink tests passed.\n"
                                : "Tracking sink tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
