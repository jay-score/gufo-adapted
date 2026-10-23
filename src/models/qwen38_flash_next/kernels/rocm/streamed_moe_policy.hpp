#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_STREAMED_MOE_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_STREAMED_MOE_POLICY_HPP_

#include <algorithm>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Desk/Room needs at least one active top-k set beyond its equal Desk/Room
/// split; express this without overflowing `2 * experts_used`.
[[nodiscard]] constexpr bool DeskRoomHasSlotHeadroom(
    std::uint32_t slots, std::uint32_t experts_used) noexcept {
  return experts_used != 0 && slots > experts_used &&
         slots - experts_used > experts_used;
}

struct DeskRoomPrefetchPlan {
  std::uint32_t prefetch_slots{0};
  std::uint32_t compute_slots{0};
  std::uint32_t passes{0};
};

[[nodiscard]] constexpr std::uint32_t CeilDiv(std::uint32_t value,
                                              std::uint32_t divisor) noexcept {
  return divisor == 0 ? 0 : value / divisor + (value % divisor != 0 ? 1 : 0);
}

/// Size Room prefetching so Room experts are covered in the fewest full passes
/// while retaining a useful prefetch window. Small rooms keep the earlier
/// one-third window; rooms with at least two active top-k sets use an 8..16
/// slot prefetch floor.
[[nodiscard]] constexpr DeskRoomPrefetchPlan PlanDeskRoomPrefetch(
    std::uint32_t room_slots, std::uint32_t room_experts,
    std::uint32_t experts_used, std::uint32_t max_prefetch = 16,
    std::uint32_t min_prefetch = 8) noexcept {
  if (room_slots == 0) {
    return {};
  }
  if (room_experts == 0) {
    return {0, room_slots, 0};
  }
  const bool has_two_used = experts_used != 0 && room_slots >= experts_used &&
                            room_slots - experts_used >= experts_used;
  const std::uint32_t old_prefetch_cap = std::min(max_prefetch, room_slots / 3);
  if (!has_two_used || old_prefetch_cap < min_prefetch) {
    const std::uint32_t prefetch = has_two_used ? old_prefetch_cap : 0;
    const std::uint32_t compute_slots = room_slots - prefetch;
    return {prefetch, compute_slots, CeilDiv(room_experts, compute_slots)};
  }
  const std::uint32_t fast_compute_slots = room_slots - min_prefetch;
  const std::uint32_t min_passes = CeilDiv(room_experts, fast_compute_slots);
  const std::uint32_t target = CeilDiv(room_experts, min_passes);
  std::uint32_t prefetch = room_slots - target;
  const std::uint32_t prefetch_cap = std::min(max_prefetch, room_slots / 3);
  if (prefetch < min_prefetch) {
    prefetch = min_prefetch;
  }
  if (prefetch > prefetch_cap) {
    prefetch = prefetch_cap;
  }
  const std::uint32_t compute_slots = room_slots - prefetch;
  return {prefetch, compute_slots, CeilDiv(room_experts, compute_slots)};
}

[[nodiscard]] constexpr bool UsesRoutedF16ExpertFormats(
    core::GgmlType gate, core::GgmlType up, core::GgmlType down,
    std::uint32_t hidden_size, std::uint32_t expert_ff) noexcept {
  return (gate == core::GgmlType::kQ4_K || gate == core::GgmlType::kQ5_K) &&
         up == gate &&
         (down == core::GgmlType::kQ5_1 || down == core::GgmlType::kQ8_0) &&
         hidden_size % 256 == 0 && expert_ff % 64 == 0;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_STREAMED_MOE_POLICY_HPP_
