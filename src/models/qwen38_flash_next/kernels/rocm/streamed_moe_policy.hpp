#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_STREAMED_MOE_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_STREAMED_MOE_POLICY_HPP_

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
