#include "src/storage/ssd_stats.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#endif

namespace gufo::storage {
namespace {

constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kBytesPerMiB = 1024.0 * 1024.0;

struct MemorySample {
  std::uint64_t working_set_bytes{0};
  std::uint64_t swap_used_bytes{0};
};

#ifdef _WIN32
MemorySample ReadMemorySample() noexcept {
  MemorySample sample;
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    sample.working_set_bytes = counters.WorkingSetSize;
  }
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    sample.swap_used_bytes =
        status.ullTotalPageFile > status.ullAvailPageFile
            ? status.ullTotalPageFile - status.ullAvailPageFile
            : 0;
  }
  return sample;
}
#else
std::uint64_t ReadStatusValue(std::string_view key,
                              std::uint64_t multiplier) noexcept {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (!line.starts_with(key))
      continue;
    std::istringstream value_stream(line.substr(key.size()));
    std::uint64_t value = 0;
    if (value_stream >> value)
      return value * multiplier;
  }
  return 0;
}

std::uint64_t ReadSystemSwapUsed() noexcept {
  std::ifstream memory("/proc/meminfo");
  std::string line;
  std::uint64_t total_kib = 0;
  std::uint64_t free_kib = 0;
  while (std::getline(memory, line)) {
    std::istringstream value_stream(line);
    std::string name;
    std::uint64_t value = 0;
    std::string unit;
    if (!(value_stream >> name >> value >> unit))
      continue;
    if (name == "SwapTotal:") {
      total_kib = value;
    } else if (name == "SwapFree:") {
      free_kib = value;
    }
  }
  return total_kib > free_kib ? (total_kib - free_kib) * 1024 : 0;
}

MemorySample ReadMemorySample() noexcept {
  MemorySample sample;
  sample.working_set_bytes = ReadStatusValue("VmRSS:", 1024);
  if (sample.working_set_bytes == 0) {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0)
      sample.working_set_bytes =
          static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
  }
  sample.swap_used_bytes = ReadSystemSwapUsed();
  return sample;
}
#endif

double Percent(std::uint64_t numerator, std::uint64_t denominator) {
  return denominator == 0 ? 0.0
                          : 100.0 * static_cast<double>(numerator) /
                                static_cast<double>(denominator);
}

std::int64_t SignedDelta(std::uint64_t current,
                         std::uint64_t baseline) noexcept {
  constexpr auto max =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (current >= baseline)
    return static_cast<std::int64_t>(std::min(current - baseline, max));
  return -static_cast<std::int64_t>(std::min(baseline - current, max));
}

}  // namespace

SsdStats::~SsdStats() {
  stop_monitor_.store(true, std::memory_order_release);
  if (monitor_.joinable())
    monitor_.join();
}

void SsdStats::Start(std::size_t budget_bytes) {
  bool expected = false;
  if (!enabled_.compare_exchange_strong(expected, true,
                                        std::memory_order_acq_rel))
    return;
  budget_bytes_ = budget_bytes;
  started_ = std::chrono::steady_clock::now();
  const auto initial = ReadMemorySample();
  swap_at_start_bytes_ = initial.swap_used_bytes;
  peak_working_set_bytes_.store(initial.working_set_bytes,
                                std::memory_order_relaxed);
  swap_now_bytes_.store(initial.swap_used_bytes, std::memory_order_relaxed);
  monitor_ = std::thread([this] { MonitorMemory(); });
}

void SsdStats::RecordExpertDemand(std::uint64_t requests,
                                  std::uint64_t resident_hits,
                                  std::uint64_t prefetched_hits_in_time,
                                  std::uint64_t cold_misses,
                                  std::uint64_t late_prefetch_requests,
                                  std::chrono::nanoseconds hard_stall,
                                  bool prefill) noexcept {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  auto& phase = phase_counters_[prefill ? 0 : 1];
  expert_requests_.fetch_add(requests, std::memory_order_relaxed);
  resident_hits_.fetch_add(resident_hits, std::memory_order_relaxed);
  prefetched_hits_in_time_.fetch_add(prefetched_hits_in_time,
                                     std::memory_order_relaxed);
  cold_misses_.fetch_add(cold_misses, std::memory_order_relaxed);
  late_prefetch_requests_.fetch_add(late_prefetch_requests,
                                    std::memory_order_relaxed);
  phase.expert_requests.fetch_add(requests, std::memory_order_relaxed);
  phase.cache_hits.fetch_add(resident_hits + prefetched_hits_in_time,
                             std::memory_order_relaxed);
  phase.cold_misses.fetch_add(cold_misses, std::memory_order_relaxed);
  phase.late_prefetch_requests.fetch_add(late_prefetch_requests,
                                         std::memory_order_relaxed);
  if (hard_stall.count() > 0) {
    const auto ns = static_cast<std::uint64_t>(hard_stall.count());
    hard_stall_nanoseconds_.fetch_add(ns, std::memory_order_relaxed);
    phase.hard_stall_nanoseconds.fetch_add(ns, std::memory_order_relaxed);
  }
}

void SsdStats::RecordPrefetchRequests(std::uint64_t requests,
                                      bool prefill) noexcept {
  if (enabled_.load(std::memory_order_relaxed)) {
    prefetch_requests_.fetch_add(requests, std::memory_order_relaxed);
    phase_counters_[prefill ? 0 : 1].prefetch_requests.fetch_add(
        requests, std::memory_order_relaxed);
  }
}

void SsdStats::RecordPleLookup(bool cache_hit, bool prefill) noexcept {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  ple_requests_.fetch_add(1, std::memory_order_relaxed);
  auto& phase = phase_counters_[prefill ? 0 : 1];
  phase.ple_requests.fetch_add(1, std::memory_order_relaxed);
  if (cache_hit) {
    ple_hits_.fetch_add(1, std::memory_order_relaxed);
    phase.ple_hits.fetch_add(1, std::memory_order_relaxed);
  }
}

void SsdStats::RecordStorageRead(std::uint64_t bytes,
                                 std::chrono::nanoseconds elapsed, bool prefill,
                                 bool prefetch) noexcept {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  RecordStorageBytes(bytes, prefill, prefetch);
  RecordStorageElapsed(elapsed, prefill, prefetch);
}

void SsdStats::RecordStorageBytes(std::uint64_t bytes, bool prefill,
                                  bool prefetch) noexcept {
  if (enabled_.load(std::memory_order_relaxed)) {
    ssd_bytes_read_.fetch_add(bytes, std::memory_order_relaxed);
    auto& phase = phase_counters_[prefill ? 0 : 1];
    (prefetch ? phase.prefetch_bytes_read : phase.demand_bytes_read)
        .fetch_add(bytes, std::memory_order_relaxed);
  }
}

void SsdStats::RecordStorageElapsed(std::chrono::nanoseconds elapsed,
                                    bool prefill, bool prefetch) noexcept {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  if (elapsed.count() > 0) {
    const auto ns = static_cast<std::uint64_t>(elapsed.count());
    ssd_read_nanoseconds_.fetch_add(ns, std::memory_order_relaxed);
    auto& phase = phase_counters_[prefill ? 0 : 1];
    (prefetch ? phase.prefetch_read_nanoseconds : phase.demand_read_nanoseconds)
        .fetch_add(ns, std::memory_order_relaxed);
  }
}

void SsdStats::RecordPleStorageBytes(std::uint64_t bytes,
                                     bool prefill) noexcept {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  ssd_bytes_read_.fetch_add(bytes, std::memory_order_relaxed);
  ple_bytes_read_.fetch_add(bytes, std::memory_order_relaxed);
  phase_counters_[prefill ? 0 : 1].ple_bytes_read.fetch_add(
      bytes, std::memory_order_relaxed);
}

void SsdStats::RecordPleStorageElapsed(std::chrono::nanoseconds elapsed,
                                       bool prefill) noexcept {
  if (!enabled_.load(std::memory_order_relaxed) || elapsed.count() <= 0)
    return;
  const auto ns = static_cast<std::uint64_t>(elapsed.count());
  ssd_read_nanoseconds_.fetch_add(ns, std::memory_order_relaxed);
  ple_read_nanoseconds_.fetch_add(ns, std::memory_order_relaxed);
  phase_counters_[prefill ? 0 : 1].ple_read_nanoseconds.fetch_add(
      ns, std::memory_order_relaxed);
}

SsdPhaseStatsSnapshot SsdStats::PhaseSnapshot(bool prefill) const noexcept {
  const auto& phase = phase_counters_[prefill ? 0 : 1];
  return {
      .expert_requests = phase.expert_requests.load(std::memory_order_relaxed),
      .cache_hits = phase.cache_hits.load(std::memory_order_relaxed),
      .cold_misses = phase.cold_misses.load(std::memory_order_relaxed),
      .prefetch_requests =
          phase.prefetch_requests.load(std::memory_order_relaxed),
      .late_prefetch_requests =
          phase.late_prefetch_requests.load(std::memory_order_relaxed),
      .hard_stall_nanoseconds =
          phase.hard_stall_nanoseconds.load(std::memory_order_relaxed),
      .demand_bytes_read =
          phase.demand_bytes_read.load(std::memory_order_relaxed),
      .prefetch_bytes_read =
          phase.prefetch_bytes_read.load(std::memory_order_relaxed),
      .ple_bytes_read = phase.ple_bytes_read.load(std::memory_order_relaxed),
      .demand_read_nanoseconds =
          phase.demand_read_nanoseconds.load(std::memory_order_relaxed),
      .prefetch_read_nanoseconds =
          phase.prefetch_read_nanoseconds.load(std::memory_order_relaxed),
      .ple_read_nanoseconds =
          phase.ple_read_nanoseconds.load(std::memory_order_relaxed),
      .ple_requests = phase.ple_requests.load(std::memory_order_relaxed),
      .ple_hits = phase.ple_hits.load(std::memory_order_relaxed),
  };
}

void SsdStats::RecordSessionWrite(std::uint64_t bytes) noexcept {
  if (enabled_.load(std::memory_order_relaxed))
    session_bytes_written_.fetch_add(bytes, std::memory_order_relaxed);
}

SsdStatsSnapshot SsdStats::Snapshot() const noexcept {
  if (enabled_.load(std::memory_order_relaxed))
    const_cast<SsdStats*>(this)->SampleMemory();
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      enabled_.load(std::memory_order_relaxed)
          ? std::chrono::duration_cast<std::chrono::nanoseconds>(now - started_)
                .count()
          : 0;
  const std::uint64_t resident_hits =
      resident_hits_.load(std::memory_order_relaxed);
  const std::uint64_t prefetched_hits =
      prefetched_hits_in_time_.load(std::memory_order_relaxed);
  return {
      .budget_bytes = budget_bytes_,
      .peak_working_set_bytes =
          peak_working_set_bytes_.load(std::memory_order_relaxed),
      .swap_delta_bytes =
          SignedDelta(swap_now_bytes_.load(std::memory_order_relaxed),
                      swap_at_start_bytes_),
      .expert_requests = expert_requests_.load(std::memory_order_relaxed),
      .resident_hits = resident_hits,
      .prefetched_hits_in_time = prefetched_hits,
      .cache_hits = resident_hits + prefetched_hits,
      .prefetch_requests = prefetch_requests_.load(std::memory_order_relaxed),
      .cold_misses = cold_misses_.load(std::memory_order_relaxed),
      .late_prefetch_requests =
          late_prefetch_requests_.load(std::memory_order_relaxed),
      .hard_stall_nanoseconds =
          hard_stall_nanoseconds_.load(std::memory_order_relaxed),
      .ple_requests = ple_requests_.load(std::memory_order_relaxed),
      .ple_hits = ple_hits_.load(std::memory_order_relaxed),
      .ssd_bytes_read = ssd_bytes_read_.load(std::memory_order_relaxed),
      .ple_bytes_read = ple_bytes_read_.load(std::memory_order_relaxed),
      .ssd_read_nanoseconds =
          ssd_read_nanoseconds_.load(std::memory_order_relaxed),
      .ple_read_nanoseconds =
          ple_read_nanoseconds_.load(std::memory_order_relaxed),
      .session_bytes_written =
          session_bytes_written_.load(std::memory_order_relaxed),
      .elapsed_nanoseconds = static_cast<std::uint64_t>(elapsed),
      .prefill = PhaseSnapshot(true),
      .decode = PhaseSnapshot(false),
  };
}

std::string SsdStats::Format() const {
  const auto stats = Snapshot();
  const double bandwidth =
      stats.ssd_read_nanoseconds == 0
          ? 0.0
          : static_cast<double>(stats.ssd_bytes_read) * 1.0e9 /
                static_cast<double>(stats.ssd_read_nanoseconds) / kBytesPerGiB;
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << "ssd_stats\n"
      << "  target_memory_budget_gib: "
      << static_cast<double>(stats.budget_bytes) / kBytesPerGiB << '\n'
      << "  peak_working_set_gib:     "
      << static_cast<double>(stats.peak_working_set_bytes) / kBytesPerGiB
      << '\n'
      << "  os_swap_delta_mib:        "
      << static_cast<double>(stats.swap_delta_bytes) / kBytesPerMiB << '\n'
      << "  expert_requests:          " << stats.expert_requests << '\n'
      << "  cache_hit_pct:            "
      << Percent(stats.cache_hits, stats.expert_requests) << '\n'
      << "  cache_hit_resident:       " << stats.resident_hits << '\n'
      << "  prefetch_requests:        " << stats.prefetch_requests << '\n'
      << "  prefetched_in_time_pct:   "
      << Percent(stats.prefetched_hits_in_time, stats.prefetch_requests) << '\n'
      << "  prefetched_hit_async:     " << stats.prefetched_hits_in_time << '\n'
      << "  prefetch_late_waits:      " << stats.late_prefetch_requests << '\n'
      << "  cold_misses_sync:         " << stats.cold_misses << '\n'
      << "  hard_stall_pct:           "
      << Percent(stats.cold_misses + stats.late_prefetch_requests,
                 stats.expert_requests)
      << '\n'
      << "  hard_stall_ms:            "
      << static_cast<double>(stats.hard_stall_nanoseconds) / 1.0e6 << '\n'
      << "  ple_shelf_requests:       " << stats.ple_requests << '\n'
      << "  ple_shelf_bytes_read:     " << stats.ple_bytes_read << '\n'
      << "  ple_shelf_read_ms:        "
      << static_cast<double>(stats.ple_read_nanoseconds) / 1.0e6 << '\n'
      << "  ple_shelf_hit_pct:        "
      << Percent(stats.ple_hits, stats.ple_requests) << '\n'
      << "  ssd_bytes_read:           " << stats.ssd_bytes_read << '\n'
      << "  effective_nvme_gib_s:     " << bandwidth << '\n'
      << "  total_session_bytes_written: " << stats.session_bytes_written
      << '\n'
      << "  inference_window_ms:      "
      << static_cast<double>(stats.elapsed_nanoseconds) / 1.0e6 << '\n';
  const auto write_phase = [&](const char* name,
                               const SsdPhaseStatsSnapshot& phase) {
    out << "  " << name << "_phase:\n"
        << "    expert_requests:        " << phase.expert_requests << '\n'
        << "    cache_hits:             " << phase.cache_hits << '\n'
        << "    cold_misses_sync:       " << phase.cold_misses << '\n'
        << "    prefetch_requests:      " << phase.prefetch_requests << '\n'
        << "    prefetch_late_waits:    " << phase.late_prefetch_requests
        << '\n'
        << "    hard_stall_ms:          "
        << static_cast<double>(phase.hard_stall_nanoseconds) / 1.0e6 << '\n'
        << "    demand_bytes_read:      " << phase.demand_bytes_read << '\n'
        << "    prefetch_bytes_read:    " << phase.prefetch_bytes_read << '\n'
        << "    ple_bytes_read:         " << phase.ple_bytes_read << '\n'
        << "    demand_read_ms:         "
        << static_cast<double>(phase.demand_read_nanoseconds) / 1.0e6 << '\n'
        << "    prefetch_read_ms:       "
        << static_cast<double>(phase.prefetch_read_nanoseconds) / 1.0e6 << '\n'
        << "    ple_read_ms:            "
        << static_cast<double>(phase.ple_read_nanoseconds) / 1.0e6 << '\n'
        << "    ple_shelf_requests:     " << phase.ple_requests << '\n'
        << "    ple_shelf_hit_pct:      "
        << Percent(phase.ple_hits, phase.ple_requests) << '\n';
  };
  write_phase("prefill", stats.prefill);
  write_phase("decode", stats.decode);
  return out.str();
}

void SsdStats::SampleMemory() noexcept {
  const auto sample = ReadMemorySample();
  auto previous = peak_working_set_bytes_.load(std::memory_order_relaxed);
  while (previous < sample.working_set_bytes &&
         !peak_working_set_bytes_.compare_exchange_weak(
             previous, sample.working_set_bytes, std::memory_order_relaxed)) {
  }
  swap_now_bytes_.store(sample.swap_used_bytes, std::memory_order_relaxed);
}

void SsdStats::MonitorMemory() noexcept {
  while (!stop_monitor_.load(std::memory_order_acquire)) {
    SampleMemory();
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  SampleMemory();
}

}  // namespace gufo::storage
