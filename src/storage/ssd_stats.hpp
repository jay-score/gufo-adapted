#ifndef GUFO_STORAGE_SSD_STATS_HPP_
#define GUFO_STORAGE_SSD_STATS_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

namespace gufo::storage {

inline constexpr std::size_t kMaxTrackedLayers = 128;

struct SsdPhaseStatsSnapshot {
  std::uint64_t expert_requests{0};
  std::uint64_t cache_hits{0};
  std::uint64_t cold_misses{0};
  std::uint64_t prefetch_requests{0};
  std::uint64_t late_prefetch_requests{0};
  std::uint64_t prefetch_unused{0};
  std::uint64_t prefetch_skipped_cap{0};
  std::uint64_t hard_stall_nanoseconds{0};
  std::uint64_t demand_bytes_read{0};
  std::uint64_t prefetch_bytes_read{0};
  std::uint64_t ple_bytes_read{0};
  std::uint64_t demand_read_nanoseconds{0};
  std::uint64_t prefetch_read_nanoseconds{0};
  std::uint64_t ple_read_nanoseconds{0};
  std::uint64_t ple_requests{0};
  std::uint64_t ple_hits{0};
};

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
  std::uint64_t prefetch_unused{0};
  std::uint64_t prefetch_skipped_cap{0};
  std::uint64_t hard_stall_nanoseconds{0};
  /// Stage 0: cold demand reads and hard-stall time the rejected tail rows
  /// of speculative verification cycles forced.
  std::uint64_t wasted_tail_cold_misses{0};
  std::uint64_t wasted_tail_bytes_read{0};
  std::uint64_t wasted_tail_stall_nanoseconds{0};
  std::uint64_t ple_requests{0};
  std::uint64_t ple_hits{0};
  std::uint64_t ssd_bytes_read{0};
  std::uint64_t ple_bytes_read{0};
  std::uint64_t ssd_read_nanoseconds{0};
  std::uint64_t ple_read_nanoseconds{0};
  std::uint64_t session_bytes_written{0};
  std::uint64_t elapsed_nanoseconds{0};
  /// Decode-phase expert demand skew, indexed by transformer layer.
  std::size_t decode_layer_count{0};
  std::array<std::uint64_t, kMaxTrackedLayers> decode_layer_cold_misses{};
  std::array<std::uint64_t, kMaxTrackedLayers> decode_layer_resident_hits{};
  SsdPhaseStatsSnapshot prefill;
  SsdPhaseStatsSnapshot decode;
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
                          std::chrono::nanoseconds hard_stall,
                          bool prefill = false) noexcept;
  /// Per-layer decode demand outcomes, used to decide whether cache slots
  /// should be redistributed across transformer layers.
  void RecordLayerDemand(std::uint32_t layer_id, std::uint64_t resident_hits,
                         std::uint64_t cold_misses,
                         bool prefill = false) noexcept;
  void RecordPrefetchRequests(std::uint64_t requests,
                              bool prefill = false) noexcept;
  void RecordPrefetchUnused(std::uint64_t requests,
                            bool prefill = false) noexcept;
  void RecordPrefetchSkipped(bool prefill = false) noexcept;
  void RecordPleLookup(bool cache_hit, bool prefill = false) noexcept;
  void RecordStorageBytes(std::uint64_t bytes, bool prefill = false,
                          bool prefetch = false) noexcept;
  void RecordStorageElapsed(std::chrono::nanoseconds elapsed,
                            bool prefill = false,
                            bool prefetch = false) noexcept;
  void RecordPleStorageBytes(std::uint64_t bytes,
                             bool prefill = false) noexcept;
  void RecordPleStorageElapsed(std::chrono::nanoseconds elapsed,
                               bool prefill = false) noexcept;
  void RecordStorageRead(std::uint64_t bytes, std::chrono::nanoseconds elapsed,
                         bool prefill = false, bool prefetch = false) noexcept;
  void RecordSessionWrite(std::uint64_t bytes) noexcept;

  /// Cold demand reads and hard-stall time that a speculative cycle's
  /// rejected tail rows forced; recorded once per verification cycle.
  void RecordVerifyTailWaste(std::uint64_t cold_misses, std::uint64_t bytes,
                             std::uint64_t stall_nanoseconds) noexcept;

  [[nodiscard]] SsdStatsSnapshot Snapshot() const noexcept;
  [[nodiscard]] std::string Format() const;

private:
  struct LayerCounters {
    std::atomic<std::uint64_t> cold_misses{0};
    std::atomic<std::uint64_t> resident_hits{0};
  };

  struct PhaseCounters {
    std::atomic<std::uint64_t> expert_requests{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> cold_misses{0};
    std::atomic<std::uint64_t> prefetch_requests{0};
    std::atomic<std::uint64_t> late_prefetch_requests{0};
    std::atomic<std::uint64_t> prefetch_unused{0};
    std::atomic<std::uint64_t> prefetch_skipped_cap{0};
    std::atomic<std::uint64_t> hard_stall_nanoseconds{0};
    std::atomic<std::uint64_t> demand_bytes_read{0};
    std::atomic<std::uint64_t> prefetch_bytes_read{0};
    std::atomic<std::uint64_t> ple_bytes_read{0};
    std::atomic<std::uint64_t> demand_read_nanoseconds{0};
    std::atomic<std::uint64_t> prefetch_read_nanoseconds{0};
    std::atomic<std::uint64_t> ple_read_nanoseconds{0};
    std::atomic<std::uint64_t> ple_requests{0};
    std::atomic<std::uint64_t> ple_hits{0};
  };

  [[nodiscard]] SsdPhaseStatsSnapshot PhaseSnapshot(
      bool prefill) const noexcept;
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
  std::atomic<std::uint64_t> prefetch_unused_{0};
  std::atomic<std::uint64_t> prefetch_skipped_cap_{0};
  std::atomic<std::uint64_t> hard_stall_nanoseconds_{0};
  std::atomic<std::uint64_t> ple_requests_{0};
  std::atomic<std::uint64_t> ple_hits_{0};
  std::atomic<std::uint64_t> ssd_bytes_read_{0};
  std::atomic<std::uint64_t> ple_bytes_read_{0};
  std::atomic<std::uint64_t> ssd_read_nanoseconds_{0};
  std::atomic<std::uint64_t> ple_read_nanoseconds_{0};
  std::atomic<std::uint64_t> session_bytes_written_{0};
  std::atomic<std::uint64_t> wasted_tail_cold_misses_{0};
  std::atomic<std::uint64_t> wasted_tail_bytes_read_{0};
  std::atomic<std::uint64_t> wasted_tail_stall_nanoseconds_{0};
  std::array<PhaseCounters, 2> phase_counters_{};
  std::array<LayerCounters, kMaxTrackedLayers> layer_counters_{};
  std::atomic<std::uint32_t> decode_layer_count_{0};
};

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_SSD_STATS_HPP_
