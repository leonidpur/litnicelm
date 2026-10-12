#pragma once

// Events logged by the config component (src/config).
// Names are "config.<event>".

#include <litnicelm_utils/log.hpp>

namespace config_events {

// config_mapper.cpp
inline constexpr EventDef kEnvOverride{
    "config.env_override",
    EventType::Info,
    "  -> ENV Override:"};

} // namespace config_events
