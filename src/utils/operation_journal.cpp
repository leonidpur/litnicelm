#include "operation_journal.hpp"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string now_timestamp_local() {
  const auto now = std::time(nullptr);
  std::tm tmv{};
#if defined(_WIN32)
  localtime_s(&tmv, &now);
#else
  localtime_r(&now, &tmv);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}
} // namespace

void log_operation(const std::string &journal_path, const std::string &op_name,
                   const std::string &details) {
  if (journal_path.empty()) {
    return;
  }

  const std::filesystem::path path(journal_path);
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream journal(path, std::ios::app);
  if (!journal) {
    throw std::runtime_error("log_operation: failed to open journal: " +
                             path.string());
  }

  journal << "[" << now_timestamp_local() << "] OPERATION: " << op_name << "\n\n";
  journal << details << "\n\n";
  if (!journal) {
    throw std::runtime_error("log_operation: failed to write journal: " +
                             path.string());
  }
}
