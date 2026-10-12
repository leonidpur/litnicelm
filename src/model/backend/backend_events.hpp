#pragma once

// Events logged by the backend component (src/model/backend): device backends
// and the plugin loader.
// Names are "backend.<event>".

#include <litnicelm_utils/log.hpp>

namespace backend_events {

// device_backend.cpp
inline constexpr EventDef kBackendLoaded{
    "backend.backend_loaded",
    EventType::Info,
    "[DeviceBackend] Loaded backend library:",
    kAnsiGreen};

} // namespace backend_events
