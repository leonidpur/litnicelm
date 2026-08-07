#pragma once

#include <string>

void log_operation(const std::string &journal_path, const std::string &op_name,
                   const std::string &details);
