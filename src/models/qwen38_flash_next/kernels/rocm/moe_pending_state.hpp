#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_MOE_PENDING_STATE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_MOE_PENDING_STATE_HPP_

namespace gufo::models::qwen38_flash_next::rocm {

/// Tracks an MoE epilogue deferred into the following combine operation.
class MoePendingState {
public:
  void Reset() noexcept { pending_ = false; }
  void SetForOutput(const void* output, const void* block_output) noexcept {
    pending_ = output == block_output;
  }
  [[nodiscard]] bool pending() const noexcept { return pending_; }
  [[nodiscard]] bool Consume() noexcept {
    const bool was_pending = pending_;
    pending_ = false;
    return was_pending;
  }

private:
  bool pending_{false};
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_MOE_PENDING_STATE_HPP_
