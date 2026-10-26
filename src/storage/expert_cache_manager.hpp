#ifndef GUFO_STORAGE_EXPERT_CACHE_MANAGER_HPP_
#define GUFO_STORAGE_EXPERT_CACHE_MANAGER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/storage/cache_policy.hpp"

namespace gufo::storage {
class SsdStats;

/// Per-layer packed GTT slots for the three routed expert matrices. An
/// expert's slot is exactly expert_bytes long; SsdBatchReader handles file and
/// destination sub-page alignment without changing the HIP weight layout.
class ExpertCacheManager {
public:
  static constexpr std::size_t kSharedScratchBufferCount = 32;
  static constexpr std::size_t kSpeculativeTransactionCap = 2;

  using VictimPolicy = ExpertCacheVictimPolicy;

  /// Canonical CLI spelling: lru, hotness, or decayed.
  [[nodiscard]] static constexpr std::optional<VictimPolicy> ParseVictimPolicy(
      std::string_view name) noexcept {
    return ParseExpertCacheVictimPolicy(name);
  }

  struct WarmupResult {
    std::uint64_t bytes_read{0};
    std::chrono::nanoseconds elapsed{0};
  };

  /// Optional demand-resolve telemetry. Each cold demand read's count and
  /// bytes are attributed to the expert's first routed occurrence within the
  /// window; each late prefetch join charges its observed join wall time to
  /// the same row; this call's demand-batch wall time is distributed across
  /// rows proportional to this call's cold bytes only. The row spans are
  /// accumulators: callers pass spans shared across layers.
  struct ExpertResolveTelemetry {
    std::uint32_t rows{0};
    std::span<std::uint64_t> row_cold_experts{};
    std::span<std::uint64_t> row_cold_bytes{};
    std::span<std::uint64_t> row_late_join_nanoseconds{};
    std::span<std::uint64_t> row_demand_read_nanoseconds{};
    std::uint64_t demand_read_nanoseconds{0};
  };

  ~ExpertCacheManager();
  ExpertCacheManager(const ExpertCacheManager&) = delete;
  ExpertCacheManager& operator=(const ExpertCacheManager&) = delete;

  [[nodiscard]] static std::unique_ptr<ExpertCacheManager> Create(
      std::span<const models::qwen38_flash_next::rocm::DeviceLayer> layers,
      std::size_t budget_bytes, std::uint32_t experts_used,
      std::uint32_t num_experts, std::span<std::int32_t> expert_id_to_slot,
      std::string* error_msg = nullptr, SsdStats* stats = nullptr,
      float cache_prior_bias = 0.0F,
      VictimPolicy victim_policy = VictimPolicy::kLru,
      std::uint32_t hotness_decay_tokens = 1024);

  [[nodiscard]] const std::vector<models::qwen38_flash_next::rocm::DeviceLayer>&
  device_layers() const noexcept;
  [[nodiscard]] std::uint32_t slots_per_layer() const noexcept;
  [[nodiscard]] std::uint32_t demand_slots_available(
      std::uint32_t layer) const noexcept;
  [[nodiscard]] std::uint32_t room_slots_available(
      std::uint32_t layer) const noexcept;
  [[nodiscard]] std::size_t allocated_bytes() const noexcept;
  [[nodiscard]] static std::size_t SharedScratchBytes(
      std::span<const models::qwen38_flash_next::rocm::DeviceLayer> layers);
  [[nodiscard]] bool Prewarm(WarmupResult* result,
                             std::string* error_msg = nullptr);
  void FinishAllPrefetch() noexcept;

  /// Starts a bounded asynchronous prefetch without evicting any slot it
  /// reserves. A second prefetch for the same layer is ignored while its
  /// current prediction batch still owns slots. Speculative transactions are
  /// globally capped; non-speculative Room reads use their higher-priority
  /// background lane and are not counted against that cap.
  [[nodiscard]] bool Prefetch(std::uint32_t layer,
                              std::span<const std::int32_t> expert_ids,
                              std::size_t max_slots,
                              std::string* error_msg = nullptr,
                              bool prefill = false, bool speculative = false);
  [[nodiscard]] bool PrefetchTokenPriors(std::uint32_t layer,
                                         std::span<const std::int32_t> tokens,
                                         std::size_t max_slots,
                                         std::string* error_msg = nullptr,
                                         bool prefill = false,
                                         bool speculative = true);
  void RecordTokenRouting(std::uint32_t layer,
                          std::span<const std::int32_t> tokens,
                          std::span<const std::int32_t> expert_ids);
  [[nodiscard]] bool PinDesk(std::uint32_t layer,
                             std::span<const std::int32_t> expert_ids,
                             std::span<std::int32_t> slot_ids,
                             std::string* error_msg = nullptr,
                             bool prefill = false);
  void ReleaseDesk(std::uint32_t layer,
                   std::span<const std::int32_t> expert_ids) noexcept;
  void FinishPrefetch(std::uint32_t layer) noexcept;

  /// Ensures the selected experts are resident and pinned, then writes their
  /// slot IDs to `slot_ids`. Release after the layer's HIP MoE work completes.
  [[nodiscard]] bool Resolve(
      std::uint32_t layer, std::span<const std::int32_t> expert_ids,
      std::span<std::int32_t> slot_ids, std::string* error_msg = nullptr,
      std::span<const std::int32_t> future_expert_ids = {},
      std::uint64_t* bytes_read = nullptr, bool record_stats = true,
      bool prefill = false, bool record_demand_stats = true,
      std::uint32_t routed_tokens = 0, bool count_route_hotness = true,
      ExpertResolveTelemetry* telemetry = nullptr);
  void Release(std::uint32_t layer) noexcept;

private:
  struct Impl;
  ExpertCacheManager();
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_EXPERT_CACHE_MANAGER_HPP_
