#ifndef GUFO_STORAGE_CACHE_POLICY_HPP_
#define GUFO_STORAGE_CACHE_POLICY_HPP_

#include <cstdint>
#include <optional>
#include <string_view>

namespace gufo::storage {

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
