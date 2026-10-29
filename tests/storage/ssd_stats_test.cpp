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
  stats.RecordPleStorageBytes(4096);
  stats.RecordPleStorageElapsed(std::chrono::milliseconds(100));
  stats.RecordExpertDemand(100, 80, 5, 15, 2, std::chrono::milliseconds(12),
                           true);
  stats.RecordPrefetchRequests(20, true);
  stats.RecordPleLookup(true, true);
  stats.RecordStorageRead(3 * gib, std::chrono::seconds(4), true);
  stats.RecordStorageRead(gib / 2, std::chrono::milliseconds(500), true, true);
  stats.RecordPleStorageBytes(8192, true);
  stats.RecordPleStorageElapsed(std::chrono::milliseconds(200), true);
  stats.RecordSessionWrite(1234);
  stats.RecordLayerDemand(0, 2, 1);
  stats.RecordLayerDemand(5, 7, 3);
  stats.RecordLayerDemand(1, 9, 0, true);

  const auto snapshot = stats.Snapshot();
  Require(snapshot.budget_bytes == 32 * gib, "budget counter");
  Require(snapshot.expert_requests == 110 && snapshot.cache_hits == 91 &&
              snapshot.resident_hits == 84 &&
              snapshot.prefetched_hits_in_time == 7,
          "expert demand counters");
  Require(
      snapshot.prefetch_requests == 24 && snapshot.prefetched_hits_in_time == 7,
      "prefetch counters");
  Require(snapshot.cold_misses == 18 && snapshot.late_prefetch_requests == 3 &&
              snapshot.hard_stall_nanoseconds == 20'000'000,
          "stall counters");
  Require(snapshot.ple_requests == 3 && snapshot.ple_hits == 2, "PLE counters");
  Require(snapshot.ssd_bytes_read == (9 * gib) / 2 + 12'288 &&
              snapshot.ple_bytes_read == 12'288 &&
              snapshot.ssd_read_nanoseconds == 6'800'000'000 &&
              snapshot.ple_read_nanoseconds == 300'000'000,
          "read counters");
  Require(snapshot.prefill.expert_requests == 100 &&
              snapshot.prefill.cache_hits == 85 &&
              snapshot.prefill.cold_misses == 15 &&
              snapshot.prefill.hard_stall_nanoseconds == 12'000'000 &&
              snapshot.prefill.demand_bytes_read == 3 * gib &&
              snapshot.prefill.prefetch_bytes_read == gib / 2 &&
              snapshot.prefill.ple_bytes_read == 8192 &&
              snapshot.prefill.ple_read_nanoseconds == 200'000'000,
          "prefill phase counters distinguish demand and speculative bytes");
  Require(snapshot.decode.expert_requests == 10 &&
              snapshot.decode.cache_hits == 6 &&
              snapshot.decode.cold_misses == 3 &&
              snapshot.decode.hard_stall_nanoseconds == 8'000'000 &&
              snapshot.decode.demand_bytes_read == gib &&
              snapshot.decode.prefetch_bytes_read == 0 &&
              snapshot.decode.ple_bytes_read == 4096 &&
              snapshot.decode.ple_read_nanoseconds == 100'000'000,
          "decode phase counters are independent of prefill");
  Require(snapshot.session_bytes_written == 1234, "session write counter");
  Require(snapshot.decode_layer_count == 6 &&
              snapshot.decode_layer_cold_misses[0] == 1 &&
              snapshot.decode_layer_resident_hits[0] == 2 &&
              snapshot.decode_layer_cold_misses[1] == 0 &&
              snapshot.decode_layer_resident_hits[1] == 0 &&
              snapshot.decode_layer_cold_misses[5] == 3 &&
              snapshot.decode_layer_resident_hits[5] == 7,
          "per-layer decode demand counters");

  const std::string report = stats.Format();
  Require(report.find("cache_hit_pct:            82.727") != std::string::npos,
          "cache hit percentage");
  Require(report.find("prefetched_in_time_pct:   29.167") != std::string::npos,
          "prefetch in-time percentage");
  Require(report.find("hard_stall_pct:           19.091") != std::string::npos,
          "hard stall percentage");
  Require(report.find("ple_shelf_hit_pct:        66.667") != std::string::npos,
          "PLE shelf hit percentage");
  Require(report.find("effective_nvme_gib_s:     0.662") != std::string::npos,
          "effective read bandwidth");
  Require(
      report.find("ple_shelf_bytes_read:     12288") != std::string::npos &&
          report.find("ple_shelf_read_ms:        300.000") != std::string::npos,
      "PLE disk telemetry is reported separately");
  Require(report.find("prefill_phase:") != std::string::npos &&
              report.find("decode_phase:") != std::string::npos,
          "report includes per-phase summaries");
  Require(report.find("decode_layer_cold_miss_skew:") != std::string::npos &&
              report.find("L5=3/7") != std::string::npos,
          "report includes per-layer decode skew");
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
