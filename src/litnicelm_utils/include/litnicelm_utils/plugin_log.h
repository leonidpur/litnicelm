#pragma once

// Logging for plugins: one C function that logs through the program's logger
// (a single instance in liblitnicelm_utils.so), so no C++ types cross the
// plugin boundary. It never throws.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// type: an EventType value. name, prefix and appendix: NUL-terminated text.
// color: an ANSI color code, or NULL for the type's default.
void litnicelm_log(uint8_t type, const char *name, const char *prefix,
                   const char *color, const char *appendix);

#ifdef __cplusplus
}

#include "litnicelm_utils/log.hpp"

#include <string>

// For C++ plugins: logs one of the plugin's own events.
inline void plugin_log(const EventDef &event, const std::string &appendix) {
  litnicelm_log(static_cast<uint8_t>(event.type), event.name, event.prefix,
                event.color, appendix.c_str());
}
#endif
