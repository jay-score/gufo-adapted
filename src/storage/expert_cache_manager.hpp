#ifndef GUFO_STORAGE_EXPERT_CACHE_MANAGER_HPP_
#define GUFO_STORAGE_EXPERT_CACHE_MANAGER_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

namespace gufo::storage {

/// Per-layer packed GTT slots for the three routed expert matrices. An
/// expert's slot is exactly expert_bytes long; SsdBatchReader handles file and
/// destination sub-page alignment without changing the HIP weight layout.
class ExpertCacheManager {
public:
  ~ExpertCacheManager();
  ExpertCacheManager(const ExpertCacheManager&) = delete;
  ExpertCacheManager& operator=(const ExpertCacheManager&) = delete;

  [[nodiscard]] static std::unique_ptr<ExpertCacheManager> Create(
      std::span<const models::qwen38_flash_next::rocm::DeviceLayer> layers,
      std::size_t budget_bytes, std::uint32_t experts_used,
      std::uint32_t num_experts, std::span<std::int32_t> expert_id_to_slot,
      std::string* error_msg = nullptr);

  [[nodiscard]] const std::vector<models::qwen38_flash_next::rocm::DeviceLayer>&
  device_layers() const noexcept;
  [[nodiscard]] std::uint32_t slots_per_layer() const noexcept;
  [[nodiscard]] std::size_t allocated_bytes() const noexcept;

  /// Ensures the selected experts are resident and pinned, then writes their
  /// slot IDs to `slot_ids`. Release after the layer's HIP MoE work completes.
  [[nodiscard]] bool Resolve(std::uint32_t layer,
                             std::span<const std::int32_t> expert_ids,
                             std::span<std::int32_t> slot_ids,
                             std::string* error_msg = nullptr);
  void Release(std::uint32_t layer) noexcept;

private:
  struct Impl;
  ExpertCacheManager();
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_EXPERT_CACHE_MANAGER_HPP_
