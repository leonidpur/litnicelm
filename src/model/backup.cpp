#include "backup.hpp"

#include <config.hpp>

#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
namespace fs = std::filesystem;

std::string now_timestamp_local() {
  const auto now = std::time(nullptr);
  std::tm tmv{};
#if defined(_WIN32)
  localtime_s(&tmv, &now);
#else
  localtime_r(&now, &tmv);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tmv, "%Y%m%d_%H%M%S");
  return oss.str();
}

void require_file_exists(const fs::path &p, const char *label) {
  if (!fs::exists(p)) {
    throw std::runtime_error(std::string("Backup: missing ") + label + ": " +
                             p.string());
  }
  if (!fs::is_regular_file(p)) {
    throw std::runtime_error(std::string("Backup: not a regular file for ") +
                             label + ": " + p.string());
  }
}
} // namespace

int run_backup_mode(const std::string &config_path, const std::string &input_path,
                    const std::string &backup_root_path) {
  const Config cfg = Config::load_from_file(config_path);

  const fs::path conf_p(config_path);
  const fs::path input_p(input_path);
  const fs::path ckpt_p(cfg.paths.model_file);
  const fs::path backup_root(backup_root_path);

  require_file_exists(conf_p, "config file");
  require_file_exists(input_p, "input file");
  require_file_exists(ckpt_p, "checkpoint file");

  fs::create_directories(backup_root);

  fs::path dst_dir = backup_root / now_timestamp_local();
  int suffix = 1;
  while (fs::exists(dst_dir)) {
    dst_dir = backup_root / (now_timestamp_local() + "_" + std::to_string(suffix++));
  }
  fs::create_directories(dst_dir);

  const fs::path dst_conf = dst_dir / conf_p.filename();
  const fs::path dst_input = dst_dir / input_p.filename();
  const fs::path dst_ckpt = dst_dir / ckpt_p.filename();

  fs::copy_file(conf_p, dst_conf, fs::copy_options::overwrite_existing);
  fs::copy_file(input_p, dst_input, fs::copy_options::overwrite_existing);
  fs::copy_file(ckpt_p, dst_ckpt, fs::copy_options::overwrite_existing);

  std::cout << "Backup created: " << dst_dir.string() << "\n";
  std::cout << "  config:     " << dst_conf.string() << "\n";
  std::cout << "  input:      " << dst_input.string() << "\n";
  std::cout << "  checkpoint: " << dst_ckpt.string() << "\n";
  return 0;
}
