#ifndef GUFO_STORAGE_SSD_STATS_HPP_
#define GUFO_STORAGE_SSD_STATS_HPP_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

namespace gufo::storage {

struct SsdStatsSnapshot {
  std::size_t budget_bytes{0};
  std::uint64_t peak_working_set_bytes{0};
  std::int64_t swap_delta_bytes{0};
  std::uint64_t expert_requests{0};
  std::uint64_t resident_hits{0};
  std::uint64_t prefetched_hits_in_time{0};
  std::uint64_t cache_hits{0};
  std::uint64_t prefetch_requests{0};
  std::uint64_t cold_misses{0};
  std::uint64_t late_prefetch_requests{0};
  std::uint64_t hard_stall_nanoseconds{0};
  std::uint64_t ple_requests{0};
  std::uint64_t ple_hits{0};
  std::uint64_t ssd_bytes_read{0};
  std::uint64_t ssd_read_nanoseconds{0};
  std::uint64_t session_bytes_written{0};
  std::uint64_t elapsed_nanoseconds{0};
};

/// Optional process-wide storage telemetry for one loaded model. Counters are
/// safe for the HIP executor, PLE readers, and bounded storage workers to
/// share.
class SsdStats {
public:
  SsdStats() = default;
  ~SsdStats();
  SsdStats(const SsdStats&) = delete;
  SsdStats& operator=(const SsdStats&) = delete;

  void Start(std::size_t budget_bytes);
  void RecordExpertDemand(std::uint64_t requests, std::uint64_t resident_hits,
                          std::uint64_t prefetched_hits_in_time,
                          std::uint64_t cold_misses,
                          std::uint64_t late_prefetch_requests,
                          std::chrono::nanoseconds hard_stall) noexcept;
  void RecordPrefetchRequests(std::uint64_t requests) noexcept;
  void RecordPleLookup(bool cache_hit) noexcept;
  void RecordStorageBytes(std::uint64_t bytes) noexcept;
  void RecordStorageElapsed(std::chrono::nanoseconds elapsed) noexcept;
  void RecordStorageRead(std::uint64_t bytes,
                         std::chrono::nanoseconds elapsed) noexcept;
  void RecordSessionWrite(std::uint64_t bytes) noexcept;

  [[nodiscard]] SsdStatsSnapshot Snapshot() const noexcept;
  [[nodiscard]] std::string Format() const;

private:
  void SampleMemory() noexcept;
  void MonitorMemory() noexcept;

  std::atomic<bool> enabled_{false};
  std::atomic<bool> stop_monitor_{false};
  std::thread monitor_;
  std::size_t budget_bytes_{0};
  std::chrono::steady_clock::time_point started_{};
  std::uint64_t swap_at_start_bytes_{0};
  std::atomic<std::uint64_t> peak_working_set_bytes_{0};
  std::atomic<std::uint64_t> swap_now_bytes_{0};
  std::atomic<std::uint64_t> expert_requests_{0};
  std::atomic<std::uint64_t> resident_hits_{0};
  std::atomic<std::uint64_t> prefetched_hits_in_time_{0};
  std::atomic<std::uint64_t> prefetch_requests_{0};
  std::atomic<std::uint64_t> cold_misses_{0};
  std::atomic<std::uint64_t> late_prefetch_requests_{0};
  std::atomic<std::uint64_t> hard_stall_nanoseconds_{0};
  std::atomic<std::uint64_t> ple_requests_{0};
  std::atomic<std::uint64_t> ple_hits_{0};
  std::atomic<std::uint64_t> ssd_bytes_read_{0};
  std::atomic<std::uint64_t> ssd_read_nanoseconds_{0};
  std::atomic<std::uint64_t> session_bytes_written_{0};
};

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_SSD_STATS_HPP_
