#pragma once

// Events logged by the program entry (src/main.cpp).
// Names are "main.<event>".

#include <litnicelm_utils/log.hpp>

namespace main_events {

// main.cpp
inline constexpr EventDef kCliParsed{
    "main.cli_parsed",
    EventType::Info,
    "[Main: Step 1] CLI parsed. Config path:"};
inline constexpr EventDef kConfigLoaded{
    "main.config_loaded",
    EventType::Info,
    "[Main: Step 2] Base YAML loaded."};
inline constexpr EventDef kEnvOverridesApplied{
    "main.env_overrides_applied",
    EventType::Info,
    "[Main: Step 3] Applying ENV overrides with prefix:"};
inline constexpr EventDef kEnvOverridesSkipped{
    "main.env_overrides_skipped",
    EventType::Info,
    "[Main: Step 3] ENV_PREFIX not set. No ENV overrides applied."};
inline constexpr EventDef kCommandOverridesApplied{
    "main.command_overrides_applied",
    EventType::Info,
    ""};
inline constexpr EventDef kFatalError{"main.fatal_error", EventType::Error, "Fatal error:"};

} // namespace main_events
