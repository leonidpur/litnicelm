#include "litnicelm_utils/log.hpp"
#include "litnicelm_utils/plugin_log.h"

#include <iostream>
#include <type_traits>

#include <unistd.h>

namespace {

void write_value(std::ostream &out, const LogFields::Value &value) {
  std::visit(
      [&out](const auto &v) {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, bool>) {
          out << (v ? "true" : "false");
        } else {
          out << v;
        }
      },
      value);
}

// An event's own color, or the default of its type.
const char *line_color(const EventDef &def) {
  if (def.color != nullptr) {
    return def.color;
  }
  switch (def.type) {
  case EventType::Warning:
    return kAnsiYellow;
  case EventType::Error:
    return kAnsiRed;
  default:
    return nullptr;
  }
}

ILogger *&installed_logger() {
  static Logger console = Logger::console();
  static ILogger *installed = &console;
  return installed;
}

} // namespace

void StreamDestination::write(const LogRecord &record) {
  const char *color = line_color(record.def);
  const bool colored = colors_ && color != nullptr;
  if (colored) {
    out_ << color;
  }
  std::string_view prefix = record.def.prefix;
  out_ << prefix;
  bool separate = !prefix.empty();
  if (!record.appendix.empty()) {
    // An appendix that starts on a new line is a block under the prefix.
    if (separate && record.appendix.front() != '\n') {
      out_ << " ";
    }
    out_ << record.appendix;
  } else {
    for (const auto &field : record.fields.items()) {
      out_ << (separate ? " " : "") << field.key << "=";
      write_value(out_, field.value);
      separate = true;
    }
  }
  if (colored) {
    out_ << kAnsiReset;
  }
  out_ << "\n";
}

ILogDestination &
Logger::add_destination(std::unique_ptr<ILogDestination> destination) {
  destinations_.push_back(std::move(destination));
  return *destinations_.back();
}

void Logger::route(EventType type, ILogDestination &destination) {
  routes_[static_cast<std::size_t>(type)].push_back(&destination);
}

void Logger::log(EventRef event, std::string_view appendix,
                 const LogFields &fields) {
  const EventDef &def = event.def();
  const LogRecord record{def, appendix, fields};
  for (ILogDestination *destination :
       routes_[static_cast<std::size_t>(def.type)]) {
    destination->write(record);
  }
}

Logger Logger::console() {
  Logger logger;
  // Colors only on a terminal; files and pipes get plain text.
  ILogDestination &out = logger.add_destination(std::make_unique<StreamDestination>(
      std::cout, /*colors=*/isatty(STDOUT_FILENO) != 0));
  ILogDestination &err = logger.add_destination(std::make_unique<StreamDestination>(
      std::cerr, /*colors=*/isatty(STDERR_FILENO) != 0));
  logger.route(EventType::Info, out);
  logger.route(EventType::History, out);
  logger.route(EventType::Probe, out);
  logger.route(EventType::Warning, err);
  logger.route(EventType::Error, err);
  return logger;
}

ILogger &logger() { return *installed_logger(); }

void set_logger(ILogger &logger) { installed_logger() = &logger; }

ScopedLogger::ScopedLogger(ILogger &logger) : previous_(::logger()) {
  set_logger(logger);
}

ScopedLogger::~ScopedLogger() { set_logger(previous_); }

extern "C" void litnicelm_log(uint8_t type, const char *name, const char *prefix,
                              const char *color, const char *appendix) {
  try {
    const EventDef def{name != nullptr ? name : "", static_cast<EventType>(type),
                       prefix != nullptr ? prefix : "", color};
    logger().log(def, appendix != nullptr ? appendix : "");
  } catch (...) {
    // Logging must not throw across the plugin boundary.
  }
}
