#pragma once

#include "memory/memory_resource_info.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

struct Config;
class NamedLayout;
class TensorStore;

// Draws the training setup tables (config, memory topology, memory usage,
// tensor store) and prints them. One method per table.
class ReportRenderer final {
public:
  explicit ReportRenderer(std::ostream &out);
  ReportRenderer();  // prints to std::cout

  void report_init_config(const Config &cfg, const NamedLayout &param_layout,
                          const NamedLayout &temp_layout);
  void report_init_topology(const NamedLayout &param_layout, void *param_base,
                            uint64_t param_size, void *grad_base,
                            uint64_t grad_size, void *adam_base,
                            uint64_t adam_size, void *temp_base,
                            uint64_t temp_size);
  void report_memory_usage(const TrainingMemoryUsage &usage);
  void report_tensor_store_topology(const Config &cfg,
                                      const TensorStore &tensor_store);

private:
  void print(const std::string &text);

  std::ostream &out_;
  bool verbose_init_enabled_ = false;
  uint64_t alignment_bytes_ = 0;
};
