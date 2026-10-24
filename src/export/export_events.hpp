#pragma once

// Events logged by the export component (src/export).
// Names are "export.<event>".

#include <litnicelm_utils/log.hpp>

namespace export_events {

inline constexpr EventDef kExportStart{"export.start", EventType::Info, "[Export][Start]"};
inline constexpr EventDef kExportWrote{"export.wrote", EventType::Info, "[Export]"};
inline constexpr EventDef kExportEnd{"export.end", EventType::Info, "[Export][End]"};

} // namespace export_events
