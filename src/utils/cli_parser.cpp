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

float parse_f32_or_throw(const std::string &s, const std::string &flag_name) {
  try {
    return std::stof(s);
  } catch (...) {
    throw std::runtime_error(flag_name + " requires a valid float");
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
        "Too many train arguments. Usage: ./build/litnicelm train --config <config.yaml> "
        "[train_seq_len] [batch_size] [--probe a,b] [--epochs N] [--logit] "
        "[--incremental|--no-incremental] [--epoch_report_every N]");
  }
  if (!positional.empty()) {
    cmd.train_seq_len_override = parse_u32_or_throw(positional[0], "train_seq_len");
  }
  if (positional.size() > 1) {
    cmd.batch_size_override = parse_u32_or_throw(positional[1], "batch_size");
  }
}

void parse_infer_args(const std::vector<std::string> &args, Command &cmd,
                      bool allow_prompt_positional) {
  std::vector<std::string> positional;
  positional.reserve(args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &arg = args[i];
    if (arg == "--prompt") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--prompt requires a value");
      }
      cmd.has_prompt_override = true;
      cmd.prompt = args[++i];
      continue;
    }
    constexpr const char *kPromptPrefix = "--prompt=";
    if (arg.rfind(kPromptPrefix, 0) == 0) {
      cmd.has_prompt_override = true;
      cmd.prompt = arg.substr(std::string(kPromptPrefix).size());
      continue;
    }
    if (arg == "--max_new") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--max_new requires a value");
      }
      cmd.has_max_new_override = true;
      cmd.max_new_override = parse_u32_or_throw(args[++i], "--max_new");
      continue;
    }
    constexpr const char *kMaxNewPrefix = "--max_new=";
    if (arg.rfind(kMaxNewPrefix, 0) == 0) {
      cmd.has_max_new_override = true;
      cmd.max_new_override =
          parse_u32_or_throw(arg.substr(std::string(kMaxNewPrefix).size()),
                             "--max_new");
      continue;
    }
    if (arg == "--temp") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--temp requires a value");
      }
      cmd.has_temp_override = true;
      cmd.temp_override = parse_f32_or_throw(args[++i], "--temp");
      continue;
    }
    constexpr const char *kTempPrefix = "--temp=";
    if (arg.rfind(kTempPrefix, 0) == 0) {
      cmd.has_temp_override = true;
      cmd.temp_override =
          parse_f32_or_throw(arg.substr(std::string(kTempPrefix).size()),
                             "--temp");
      continue;
    }
    if (arg == "--top_k") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--top_k requires a value");
      }
      cmd.has_top_k_override = true;
      cmd.top_k_override = parse_u32_or_throw(args[++i], "--top_k");
      continue;
    }
    constexpr const char *kTopKPrefix = "--top_k=";
    if (arg.rfind(kTopKPrefix, 0) == 0) {
      cmd.has_top_k_override = true;
      cmd.top_k_override =
          parse_u32_or_throw(arg.substr(std::string(kTopKPrefix).size()),
                             "--top_k");
      continue;
    }
    if (arg == "--top_p") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--top_p requires a value");
      }
      cmd.has_top_p_override = true;
      cmd.top_p_override = parse_f32_or_throw(args[++i], "--top_p");
      continue;
    }
    constexpr const char *kTopPPrefix = "--top_p=";
    if (arg.rfind(kTopPPrefix, 0) == 0) {
      cmd.has_top_p_override = true;
      cmd.top_p_override =
          parse_f32_or_throw(arg.substr(std::string(kTopPPrefix).size()),
                             "--top_p");
      continue;
    }
    if (arg == "--seed") {
      if (i + 1 >= args.size()) {
        throw std::runtime_error("--seed requires a value");
      }
      cmd.has_seed_override = true;
      cmd.seed_override = parse_u32_or_throw(args[++i], "--seed");
      continue;
    }
    constexpr const char *kSeedPrefix = "--seed=";
    if (arg.rfind(kSeedPrefix, 0) == 0) {
      cmd.has_seed_override = true;
      cmd.seed_override =
          parse_u32_or_throw(arg.substr(std::string(kSeedPrefix).size()),
                             "--seed");
      continue;
    }
    if (arg.rfind("--", 0) == 0) {
      throw std::runtime_error("Unknown flag: " + arg);
    }
    positional.push_back(arg);
  }

  if (!allow_prompt_positional) {
    if (!positional.empty()) {
      throw std::runtime_error("This target does not accept positional arguments");
    }
    return;
  }
  if (positional.size() > 1) {
    throw std::runtime_error("infer/inspect accepts at most one positional prompt argument");
  }
  if (!positional.empty()) {
    if (cmd.has_prompt_override) {
      throw std::runtime_error("Use either --prompt or one positional prompt, not both");
    }
    cmd.has_prompt_override = true;
    cmd.prompt = positional[0];
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
      << "    ./build/litnicelm train --config <config.yaml> [train_seq_len] [batch_size] "
         "[--probe embeddings,output_head|--do-probe] [--epochs N|--epochs=N] [--logit] "
         "[--incremental|--no-incremental] "
         "[--epoch_report_every N|--epoch_report_every=N]\n"
      << "    ./build/litnicelm --config <config.yaml> [train_seq_len] [batch_size] "
         "[--probe embeddings,output_head|--do-probe] [--epochs N|--epochs=N] [--logit] "
         "[--incremental|--no-incremental] "
         "[--epoch_report_every N|--epoch_report_every=N]\n"
      << "  Dry run:\n"
      << "    ./build/litnicelm dry_run --config <config.yaml> [train_seq_len] [batch_size] "
         "[--probe embeddings,output_head|--do-probe] [--epochs N|--epochs=N] [--logit] "
         "[--incremental|--no-incremental] "
         "[--epoch_report_every N|--epoch_report_every=N]\n"
      << "  Inference:\n"
      << "    ./build/litnicelm infer --config <config.yaml> "
         "[--prompt TEXT] [--max_new N] [--temp X] [--top_k K] [--top_p P] [--seed N]\n"
      << "  Inspect next-token distribution:\n"
      << "    ./build/litnicelm inspect --config <config.yaml> "
         "[--prompt TEXT] [--max_new N] [--temp X] [--top_k K] [--top_p P] [--seed N]\n"
      << "  Interactive inference:\n"
      << "    ./build/litnicelm inferloop --config <config.yaml>\n"
      << "  Train tokenizer artifacts:\n"
      << "    ./build/litnicelm tokenizer_training --config <config.yaml>\n"
      << "    ./build/litnicelm --tokenizer_training --config <config.yaml>\n"
      << "  Tokenize corpus:\n"
      << "    ./build/litnicelm encode --config <config.yaml>\n"
      << "    ./build/litnicelm --encode --config <config.yaml>\n"
      << "    ./build/litnicelm --tokenize --config <config.yaml>\n"
      << "  Help:\n"
      << "    ./build/litnicelm --help\n";
}

Command CliParser::parse(int argc, char **argv) {
  Command cmd{};
  cmd.target = Command::Target::TRAIN;
  cmd.config_path.clear();
  cmd.train_seq_len_override = 0;
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
  cmd.has_prompt_override = false;
  cmd.prompt.clear();
  cmd.has_max_new_override = false;
  cmd.max_new_override = 0;
  cmd.has_temp_override = false;
  cmd.temp_override = 0.0f;
  cmd.has_top_k_override = false;
  cmd.top_k_override = 0;
  cmd.has_top_p_override = false;
  cmd.top_p_override = 0.0f;
  cmd.has_seed_override = false;
  cmd.seed_override = 0;

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
    if (!args.empty()) {
      throw std::runtime_error("This target does not accept positional arguments");
    }
    break;
  case Command::Target::INFER:
  case Command::Target::INSPECT:
    parse_infer_args(args, cmd, /*allow_prompt_positional=*/true);
    break;
  case Command::Target::INFERLOOP:
    parse_infer_args(args, cmd, /*allow_prompt_positional=*/false);
    break;
  case Command::Target::DRY_RUN:
  case Command::Target::TRAIN:
    parse_train_args(args, cmd);
    break;
  }

  return cmd;
}
