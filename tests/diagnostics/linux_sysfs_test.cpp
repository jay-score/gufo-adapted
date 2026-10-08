#include "src/core/diagnostics/linux_sysfs.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void TestCustomProcRoot() {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("gufo-sysfs-test-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root / "sys");
  std::filesystem::create_directories(root / "proc");
  gufo::diagnostics::LinuxSysfs sysfs(root / "sys", root / "proc");
  const bool missing_cpu = !sysfs.QueryCpuInfo().has_value();
  const bool missing_memory = !sysfs.QueryMemInfo().has_value();
  {
    std::ofstream cpu(root / "proc" / "cpuinfo");
    cpu << "processor : 0\nmodel name : Fixture CPU\nsiblings : 8\n"
           "cpu cores : 4\n";
    std::ofstream memory(root / "proc" / "meminfo");
    memory << "MemTotal: 4096 kB\nMemAvailable: 2048 kB\nMemFree: 1024 kB\n";
  }
  const auto cpu = sysfs.QueryCpuInfo();
  const auto memory = sysfs.QueryMemInfo();
  std::filesystem::remove_all(root);

  Expect(missing_cpu && missing_memory,
         "Missing fixture files must not fall back to host hardware");
  Expect(cpu && cpu->model_name == "Fixture CPU" && cpu->logical_cores == 8 &&
             cpu->physical_cores == 4,
         "Custom proc root supplies CPU data");
  Expect(memory && memory->total_bytes == 4096 * 1024 &&
             memory->available_bytes == 2048 * 1024 &&
             memory->free_bytes == 1024 * 1024,
         "Custom proc root supplies memory data");
}

#ifdef _WIN32
void TestWindowsHostFallback() {
  const gufo::diagnostics::LinuxSysfs sysfs;
  const auto cpu = sysfs.QueryCpuInfo();
  const auto memory = sysfs.QueryMemInfo();
  Expect(cpu && !cpu->model_name.empty() && cpu->logical_cores > 0,
         "Default Windows proc root supplies native CPU data");
  Expect(memory && memory->total_bytes > 0 &&
             memory->available_bytes <= memory->total_bytes &&
             memory->free_bytes == memory->available_bytes,
         "Default Windows proc root supplies native memory data");
}
#endif

}  // namespace

int main() {
  TestCustomProcRoot();
#ifdef _WIN32
  TestWindowsHostFallback();
#endif
  std::cout << "LinuxSysfs tests passed\n";
}
