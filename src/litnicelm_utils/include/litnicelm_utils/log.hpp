#pragma once

// The logging mechanism shared by the program and its plugins: events,
// fields, destinations, routing by type and console colors. What is logged
// (the events themselves) belongs to the components.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// What the output is for. The logger routes by type.
enum class EventType : uint8_t {
  Info,     // what the program is doing: stages, loads, saves, ETA
  Warning,
  Error,
  Probe,    // tensor and memory diagnostics
  History,  // the run's record: metrics, checkpoints, decisions
  Count
};

// One kind of logged event, defined by the component that logs it.
struct EventDef {
  const char *name;    // stable machine name, for structured destinations
  EventType type;
  const char *prefix;  // text written before the component's appendix
  const char *color = nullptr;  // ANSI color of the whole line on a console;
                                // unset: Warning yellow, Error red
};

inline constexpr const char *kAnsiRed = "\033[31m";
inline constexpr const char *kAnsiGreen = "\033[32m";
inline constexpr const char *kAnsiYellow = "\033[33m";
inline constexpr const char *kAnsiCyan = "\033[36m";
inline constexpr const char *kAnsiReset = "\033[0m";

// The event to log: an EventDef, or an id that an event_def(id) function
// (found by argument-dependent lookup) turns into one. The id form serves
// the central catalog until events move to their components.
class EventRef {
public:
  EventRef(const EventDef &def) : def_(&def) {}
  template <typename Id,
            typename = decltype(event_def(std::declval<Id>()))>
  EventRef(Id id) : def_(&event_def(id)) {}

  const EventDef &def() const { return *def_; }

private:
  const EventDef *def_;
};

// Typed key/value data of one event, for destinations that keep numbers
// (journal, TensorBoard). Text destinations write them as key=value.
class LogFields {
public:
  using Value = std::variant<bool, int64_t, uint64_t, double, std::string>;
  struct Field {
    std::string key;
    Value value;
  };

  LogFields &boolean(std::string key, bool v) { return add(std::move(key), v); }
  LogFields &i64(std::string key, int64_t v) { return add(std::move(key), v); }
  LogFields &u64(std::string key, uint64_t v) { return add(std::move(key), v); }
  LogFields &f64(std::string key, double v) { return add(std::move(key), v); }
  LogFields &str(std::string key, std::string v) {
    return add(std::move(key), std::move(v));
  }

  const std::vector<Field> &items() const { return fields_; }
  bool empty() const { return fields_.empty(); }

private:
  LogFields &add(std::string key, Value v) {
    fields_.push_back({std::move(key), std::move(v)});
    return *this;
  }
  std::vector<Field> fields_;
};

// One logged event as the destinations see it.
struct LogRecord {
  const EventDef &def;
  std::string_view appendix;
  const LogFields &fields;
};

// Where records end up: a stream, a file, a tracking plugin.
class ILogDestination {
public:
  virtual ~ILogDestination() = default;
  virtual void write(const LogRecord &record) = 0;
};

// What components use. The event decides the prefix text and the type; the
// type decides the destinations.
class ILogger {
public:
  virtual ~ILogger() = default;
  virtual void log(EventRef event, std::string_view appendix,
                   const LogFields &fields) = 0;

  void log(EventRef event, std::string_view appendix = {}) {
    log(event, appendix, LogFields{});
  }
  void log(EventRef event, const LogFields &fields) { log(event, {}, fields); }
};

// Writes "prefix appendix" to a stream; without an appendix, the fields as
// key=value. With colors, the whole line in the event's color.
class StreamDestination final : public ILogDestination {
public:
  explicit StreamDestination(std::ostream &out, bool colors = false)
      : out_(out), colors_(colors) {}
  void write(const LogRecord &record) override;

private:
  std::ostream &out_;
  bool colors_;
};

// Sends each event to every destination routed for its type.
class Logger final : public ILogger {
public:
  using ILogger::log;

  // The logger keeps the destination; route it with route().
  ILogDestination &add_destination(std::unique_ptr<ILogDestination> destination);
  void route(EventType type, ILogDestination &destination);

  void log(EventRef event, std::string_view appendix,
           const LogFields &fields) override;

  // Info, History and Probe to stdout, Warning and Error to stderr.
  static Logger console();

private:
  std::vector<std::unique_ptr<ILogDestination>> destinations_;
  std::array<std::vector<ILogDestination *>,
             static_cast<std::size_t>(EventType::Count)>
      routes_;
};

// The process-wide logger components log to. Until set_logger() is called it
// is a Logger::console().
ILogger &logger();
// Installs logger, which must stay alive while installed. Call at startup,
// not while other threads log.
void set_logger(ILogger &logger);

// Installs a logger for the lifetime of the scope, then restores the previous
// one. For tests.
class ScopedLogger {
public:
  explicit ScopedLogger(ILogger &logger);
  ~ScopedLogger();
  ScopedLogger(const ScopedLogger &) = delete;
  ScopedLogger &operator=(const ScopedLogger &) = delete;

private:
  ILogger &previous_;
};
