#ifndef GUFO_STORAGE_CACHE_POLICY_HPP_
#define GUFO_STORAGE_CACHE_POLICY_HPP_

#include <cstdint>
#include <optional>
#include <string_view>

namespace gufo::storage {

/// One measured point of a layer's expert-cache miss curve. `slots` is a
/// resident slot count and `rate_milli_percent` is the decode cold-miss rate
/// (cold misses per expert request) in units of 0.001 %.
struct SlotMarginalPoint {
  std::uint32_t slots{0};
  std::uint32_t rate_milli_percent{0};
};

/// Monotone, piecewise-linear miss curve used by the marginal slot allocator.
/// Points must be strictly increasing in `slots` and non-increasing in
/// `rate_milli_percent`. Rates are extrapolated with the nearest segment and
/// clamped at zero, so adding slots beyond saturation yields no further gain.
/// Slots below `first.slots` are filled before any layer is given slots inside
/// its measured range, so the allocator never extrapolates a shallow segment
/// into an under-calibrated region and stays monotone in the slot budget.
struct SlotMarginalCurve {
  SlotMarginalPoint first{};
  SlotMarginalPoint second{};
  SlotMarginalPoint third{};
};

/// Expert-cache victim selection. Kept free of runtime headers so command
/// line validation links without the HIP expert cache.
enum class ExpertCacheVictimPolicy : std::uint8_t {
  kLru,
  kCumulativeHotness,
  kDecayedHotness,
};

/// Canonical CLI spelling: lru, hotness, or decayed.
[[nodiscard]] constexpr std::optional<ExpertCacheVictimPolicy>
ParseExpertCacheVictimPolicy(std::string_view name) noexcept {
  if (name == "lru")
    return ExpertCacheVictimPolicy::kLru;
  if (name == "hotness")
    return ExpertCacheVictimPolicy::kCumulativeHotness;
  if (name == "decayed")
    return ExpertCacheVictimPolicy::kDecayedHotness;
  return std::nullopt;
}

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_CACHE_POLICY_HPP_
