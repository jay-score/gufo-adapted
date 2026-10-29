#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SLOT_PROFILE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SLOT_PROFILE_HPP_

#include <array>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

/// Validated per-layer expert-cache slots for 48 streamed layers. The total is
/// 13,920 slots, the same total as a uniform 290 slots per layer. The storage
/// layer scales this shape to the budget-derived total when the cache is
/// smaller or larger.
inline constexpr std::array<std::uint32_t, 48> kSkewedSlotProfile = {
    460, 436, 456, 264, 312, 356, 380, 388, 332, 248, 312, 224,
    288, 304, 264, 156, 184, 296, 340, 240, 272, 364, 408, 276,
    292, 252, 272, 260, 284, 268, 284, 184, 220, 280, 288, 240,
    248, 300, 364, 228, 220, 260, 244, 276, 276, 288, 244, 288,
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SLOT_PROFILE_HPP_
