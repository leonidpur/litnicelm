#pragma once

#include <types.hpp>

#include <string>

// Serializes the fully resolved config (after file, env and CLI overrides)
// as YAML that Config::load_from_file reads back to the same values.
std::string config_to_yaml(const Config &cfg);
