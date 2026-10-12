// A plugin that logs one of its own events through liblitnicelm_utils.so.

#include <litnicelm_utils/plugin_log.h>

namespace {
inline constexpr EventDef kSays{"fake_plugin_says", EventType::Warning,
                                "[FakePlugin] says:"};
} // namespace

extern "C" void fake_plugin_emit(const char *text) { plugin_log(kSays, text); }
