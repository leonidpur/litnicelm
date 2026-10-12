// Checks that a plugin's log calls reach the program's installed logger, i.e.
// that host and plugin share one logger instance in liblitnicelm_utils.so.
//
// Usage: litnicelm_plugin_log_test <fake_log_plugin.so>

#include <litnicelm_utils/log.hpp>

#include <dlfcn.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
  if (!ok) {
    ++g_failures;
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <fake_log_plugin.so>\n";
    return 2;
  }
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::cerr << "dlopen failed: " << dlerror() << "\n";
    return 2;
  }
  using EmitFn = void (*)(const char *);
  auto emit = reinterpret_cast<EmitFn>(dlsym(handle, "fake_plugin_emit"));
  check(emit != nullptr, "plugin exports fake_plugin_emit");

  std::ostringstream captured;
  Logger host_logger;
  ILogDestination &out =
      host_logger.add_destination(std::make_unique<StreamDestination>(captured));
  host_logger.route(EventType::Warning, out);
  {
    ScopedLogger installed(host_logger);
    if (emit != nullptr) {
      emit("hello");
    }
  }
  check(captured.str() == "[FakePlugin] says: hello\n",
        "plugin event reaches the host's logger: '" + captured.str() + "'");

  dlclose(handle);
  std::cout << (g_failures == 0 ? "All plugin log tests passed.\n"
                                : "Plugin log tests FAILED.\n");
  return g_failures == 0 ? 0 : 1;
}
