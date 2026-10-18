#include "src/storage/ssd_stats.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Run() {
  gufo::storage::SsdStats stats;
  constexpr std::size_t gib = std::size_t{1} << 30;
  stats.Start(32 * gib);
  stats.RecordExpertDemand(10, 4, 2, 3, 1, std::chrono::milliseconds(8));
  stats.RecordPrefetchRequests(4);
  stats.RecordPleLookup(true);
  stats.RecordPleLookup(false);
  stats.RecordStorageRead(gib, std::chrono::seconds(2));
  stats.RecordSessionWrite(1234);

  const auto snapshot = stats.Snapshot();
  Require(snapshot.budget_bytes == 32 * gib, "budget counter");
  Require(snapshot.expert_requests == 10 && snapshot.cache_hits == 6 &&
              snapshot.resident_hits == 4 &&
              snapshot.prefetched_hits_in_time == 2,
          "expert demand counters");
  Require(
      snapshot.prefetch_requests == 4 && snapshot.prefetched_hits_in_time == 2,
      "prefetch counters");
  Require(snapshot.cold_misses == 3 && snapshot.late_prefetch_requests == 1 &&
              snapshot.hard_stall_nanoseconds == 8'000'000,
          "stall counters");
  Require(snapshot.ple_requests == 2 && snapshot.ple_hits == 1, "PLE counters");
  Require(snapshot.ssd_bytes_read == gib &&
              snapshot.ssd_read_nanoseconds == 2'000'000'000,
          "read counters");
  Require(snapshot.session_bytes_written == 1234, "session write counter");

  const std::string report = stats.Format();
  Require(report.find("cache_hit_pct:            60.000") != std::string::npos,
          "cache hit percentage");
  Require(report.find("prefetched_in_time_pct:   50.000") != std::string::npos,
          "prefetch in-time percentage");
  Require(report.find("hard_stall_pct:           40.000") != std::string::npos,
          "hard stall percentage");
  Require(report.find("ple_shelf_hit_pct:        50.000") != std::string::npos,
          "PLE shelf hit percentage");
  Require(report.find("effective_nvme_gib_s:     0.500") != std::string::npos,
          "effective read bandwidth");
  std::cout << "ssd_stats_test: telemetry counters and formatting passed\n";
}

}  // namespace

int main() {
  try {
    Run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ssd_stats_test: " << error.what() << '\n';
    return 1;
  }
}
