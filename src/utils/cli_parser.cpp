#include "cli_parser.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
uint32_t parse_u32_or_throw(const std::string &s, const std::string &flag_name) {
  try {
    const unsigned long v = std::stoul(s);
    return static_cast<uint32_t>(v);
  } catch (...) {
    throw std::runtime_error(flag_name + " requires a valid unsigned integer");
  }
}

bool parse_bool_or_throw(const std::string &s, const std::string &flag_name) {
  if (s == "true" || s == "1") {
    return true;
  }
  if (s == "false" || s == "0") {
    return false;
  }
  throw std::runtime_error(flag_name + " requires true/false or 1/0");
}

void set_probe_flag_or_throw(const std::string &token, Command &cmd) {
  if (token == "embeddings") {
    cmd.runtime_flags.probe.embeddings = true;
    return;
  }
  if (token == "output_head") {
    cmd.runtime_flags.probe.output_head = true;
    return;
  }
  if (token == "loss") {
    cmd.runtime_flags.probe.loss = true;
    return;
  }
  if (token == "backward") {
    cmd.runtime_flags.probe.backward = true;
    return;
  }
  if (token == "attention") {
    cmd.runtime_flags.probe.attention = true;
    return;
  }
  if (token == "ffn") {
    cmd.runtime_flags.probe.ffn = true;
    return;
  }
  if (token == "layernorm") {
    cmd.runtime_flags.probe.layernorm = true;
    return;
  }
  throw std::runtime_error("unknown probe selector: " + token);
}

void parse_probe_list_or_throw(const std::string &value, Command &cmd) {
  std::stringstream ss(value);
  std::string token;
  bool saw_any = false;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    saw_any = true;
    set_probe_flag_or_throw(token, cmd);
  }
  if (!saw_any) {
    throw std::runtime_error("--probe requires a non-empty selector list");
  }
}

void parse_train_args(const std::vector<std::string> &args, Command &cmd) {
  std::vector<std::string> positional;
  positional.reserve(args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &arg = args[i];
    if (arg == "--do-probe") {
      cmd.runtime_flags.probe.embeddings = true;
      cmd.runtime_flags.probe.output_head = true;
      cmd.runtime_flags.probe.loss = true;
      cmd.runtime_flags.probe.backward = true;
      cmd.runtime_flags.probe.attention = true;
      cmd.runtime_flags.probe.ffn = true;
      cmd.runtime_flags.probe.layernorm = true;
      continue;
    }
    if (arg == "--probe") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--probe requires a selector list");
      }
      parse_probe_list_or_throw(args[++i], cmd);
      continue;
    }
    constexpr const char *kProbePrefix = "--probe=";
    if (arg.rfind(kProbePrefix, 0) == 0) {
      parse_probe_list_or_throw(arg.substr(std::string(kProbePrefix).size()), cmd);
      continue;
    }
    if (arg == "--logit" || arg == "--logits") {
      cmd.runtime_flags.logit = true;
      continue;
    }
    if (arg == "--incremental") {
      cmd.has_incremental_override = true;
      cmd.incremental_override = true;
      continue;
    }
    if (arg == "--no-incremental") {
      cmd.has_incremental_override = true;
      cmd.incremental_override = false;
      continue;
    }
    constexpr const char *kIncrementalPrefix = "--incremental=";
    if (arg.rfind(kIncrementalPrefix, 0) == 0) {
      cmd.has_incremental_override = true;
      cmd.incremental_override = parse_bool_or_throw(
          arg.substr(std::string(kIncrementalPrefix).size()), "--incremental");
      continue;
    }
    if (arg == "--epochs") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--epochs requires a value");
      }
      cmd.num_epochs_override = parse_u32_or_throw(args[++i], "--epochs");
      continue;
    }
    constexpr const char *kEpochPrefix = "--epochs=";
    if (arg.rfind(kEpochPrefix, 0) == 0) {
      cmd.num_epochs_override =
          parse_u32_or_throw(arg.substr(std::string(kEpochPrefix).size()), "--epochs");
      continue;
    }
    if (arg == "--epoch_report_every") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--epoch_report_every requires a value");
      }
      cmd.runtime_flags.epoch_report_every =
          parse_u32_or_throw(args[++i], "--epoch_report_every");
      if (cmd.runtime_flags.epoch_report_every == 0) {
        throw std::runtime_error("--epoch_report_every must be >= 1");
      }
      continue;
    }
    constexpr const char *kEpochReportEveryPrefix = "--epoch_report_every=";
    if (arg.rfind(kEpochReportEveryPrefix, 0) == 0) {
      cmd.runtime_flags.epoch_report_every = parse_u32_or_throw(
          arg.substr(std::string(kEpochReportEveryPrefix).size()),
          "--epoch_report_every");
      if (cmd.runtime_flags.epoch_report_every == 0) {
        throw std::runtime_error("--epoch_report_every must be >= 1");
      }
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      throw std::runtime_error("Unknown flag: " + arg);
    }
    positional.push_back(arg);
  }

  if (positional.size() > 2) {
    throw std::runtime_error(
        "Too many train arguments. Usage: ./build/litnicegpt train --config <config.yaml> "
        "[window_training] [batch_size] [--probe a,b] [--epochs N] [--logit] "
        "[--incremental|--no-incremental] [--epoch_report_every N]");
  }
  if (!positional.empty()) {
    cmd.window_training_override = parse_u32_or_throw(positional[0], "window_training");
  }
  if (positional.size() > 1) {
    cmd.batch_size_override = parse_u32_or_throw(positional[1], "batch_size");
  }
}
} // namespace

bool CliParser::wants_help(int argc, char **argv) {
  return argc > 1 &&
         (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

void CliParser::print_usage() {
  std::cout
      << "Usage:\n"
      << "  Train (default):\n"
      << "    ./build/litnicegpt train --config <config.yaml> [window_training] [batch_size] "
         "[--probe embeddings,output_head|--do-probe] [--epochs N|--epochs=N] [--logit] "
         "[--incremental|--no-incremental] "
         "[--epoch_report_every N|--epoch_report_every=N]\n"
      << "    ./build/litnicegpt --config <config.yaml> [window_training] [batch_size] "
         "[--probe embeddings,output_head|--do-probe] [--epochs N|--epochs=N] [--logit] "
         "[--incremental|--no-incremental] "
         "[--epoch_report_every N|--epoch_report_every=N]\n"
      << "  Dry run:\n"
      << "    ./build/litnicegpt dry_run --config <config.yaml> [window_training] [batch_size] "
         "[--probe embeddings,output_head|--do-probe] [--epochs N|--epochs=N] [--logit] "
         "[--incremental|--no-incremental] "
         "[--epoch_report_every N|--epoch_report_every=N]\n"
      << "  Inference:\n"
      << "    ./build/litnicegpt infer --config <config.yaml> [prompt]\n"
      << "  Inspect next-token distribution:\n"
      << "    ./build/litnicegpt inspect --config <config.yaml> [prompt]\n"
      << "  Interactive inference:\n"
      << "    ./build/litnicegpt inferloop --config <config.yaml>\n"
      << "  Backup:\n"
      << "    ./build/litnicegpt --backup --config <config.yaml> [input] [backup_root]\n"
      << "  Train tokenizer artifacts:\n"
      << "    ./build/litnicegpt tokenizer_training --config <config.yaml>\n"
      << "    ./build/litnicegpt --tokenizer_training --config <config.yaml>\n"
      << "  Tokenize corpus:\n"
      << "    ./build/litnicegpt encode --config <config.yaml>\n"
      << "    ./build/litnicegpt --encode --config <config.yaml>\n"
      << "    ./build/litnicegpt --tokenize --config <config.yaml>\n"
      << "  Help:\n"
      << "    ./build/litnicegpt --help\n";
}

Command CliParser::parse(int argc, char **argv) {
  Command cmd{};
  cmd.target = Command::Target::TRAIN;
  cmd.config_path.clear();
  cmd.window_training_override = 0;
  cmd.batch_size_override = 0;
  cmd.has_incremental_override = false;
  cmd.incremental_override = false;
  cmd.runtime_flags.probe.embeddings = false;
  cmd.runtime_flags.probe.output_head = false;
  cmd.runtime_flags.probe.loss = false;
  cmd.runtime_flags.probe.backward = false;
  cmd.runtime_flags.probe.attention = false;
  cmd.runtime_flags.probe.ffn = false;
  cmd.runtime_flags.probe.layernorm = false;
  cmd.runtime_flags.logit = false;
  cmd.runtime_flags.epoch_report_every = 0;
  cmd.num_epochs_override = 0;
  cmd.prompt.clear();
  cmd.backup_input.clear();
  cmd.backup_root.clear();

  size_t arg_start = 1;
  if (argc > 1) {
    const std::string candidate = argv[1];
    if (candidate == "tokenizer_training") {
      cmd.target = Command::Target::TOKENIZER_TRAINING;
      arg_start = 2;
    } else if (candidate == "--tokenizer_training" ||
               candidate == "--tokenizer_trainig") {
      cmd.target = Command::Target::TOKENIZER_TRAINING;
      arg_start = 2;
    } else if (candidate == "encode" || candidate == "--encode" ||
               candidate == "--tokenize") {
      cmd.target = Command::Target::TOKENIZE;
      arg_start = 2;
    } else if (candidate == "infer") {
      cmd.target = Command::Target::INFER;
      arg_start = 2;
    } else if (candidate == "inspect") {
      cmd.target = Command::Target::INSPECT;
      arg_start = 2;
    } else if (candidate == "inferloop") {
      cmd.target = Command::Target::INFERLOOP;
      arg_start = 2;
    } else if (candidate == "--backup") {
      cmd.target = Command::Target::BACKUP;
      arg_start = 2;
    } else if (candidate == "dry_run" || candidate == "--dry_run") {
      cmd.target = Command::Target::DRY_RUN;
      arg_start = 2;
    } else if (candidate == "train" || candidate == "--train") {
      cmd.target = Command::Target::TRAIN;
      arg_start = 2;
    }
  }

  std::vector<std::string> args;
  for (size_t i = arg_start; i < static_cast<size_t>(argc); ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= static_cast<size_t>(argc)) {
        throw std::runtime_error("--config requires a yaml path");
      }
      cmd.config_path = argv[++i];
      continue;
    }
    constexpr const char *kConfigPrefix = "--config=";
    if (arg.rfind(kConfigPrefix, 0) == 0) {
      cmd.config_path = arg.substr(std::string(kConfigPrefix).size());
      continue;
    }
    args.push_back(arg);
  }

  switch (cmd.target) {
  case Command::Target::TOKENIZER_TRAINING:
  case Command::Target::TOKENIZE:
  case Command::Target::INFERLOOP:
    if (!args.empty()) {
      throw std::runtime_error("This target does not accept positional arguments");
    }
    break;
  case Command::Target::INFER:
  case Command::Target::INSPECT:
    if (args.size() > 1) {
      throw std::runtime_error("infer/inspect accepts at most one prompt argument");
    }
    cmd.prompt = args.empty() ? "" : args[0];
    break;
  case Command::Target::BACKUP:
    if (args.size() != 2) {
      throw std::runtime_error(
          "Usage: ./build/litnicegpt --backup --config <config.yaml> [input] [backup_root]");
    }
    cmd.backup_input = args[0];
    cmd.backup_root = args[1];
    break;
  case Command::Target::DRY_RUN:
  case Command::Target::TRAIN:
    parse_train_args(args, cmd);
    break;
  }

  return cmd;
}
