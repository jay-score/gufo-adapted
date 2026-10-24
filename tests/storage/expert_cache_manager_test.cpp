#include "src/storage/expert_cache_manager.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kExperts = 4;
constexpr std::uint32_t kExpertsUsed = 2;
constexpr std::size_t kExpertBytes = 1501;
constexpr std::array<std::uint64_t, 3> kTensorOffsets{4096 + 11, 16384 + 27,
                                                      28672 + 43};
constexpr std::size_t kTailMargin = 4096;

std::uint8_t Pattern(std::uint64_t offset) {
  std::uint64_t value = offset + 0x9e3779b97f4a7c15ULL;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<std::uint8_t>(value >> 24);
}

void CheckHip(const char* operation, hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

struct TempFile {
  std::filesystem::path path;
  ~TempFile() {
    if (!path.empty()) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  }
};

void CheckExpertBytes(
    const gufo::models::qwen38_flash_next::rocm::DeviceTensor& t,
    std::uint32_t slot, std::uint64_t file_offset) {
  std::vector<std::uint8_t> actual(kExpertBytes);
  const auto* source = static_cast<const std::uint8_t*>(t.data) +
                       static_cast<std::size_t>(slot) * kExpertBytes;
  CheckHip(
      "read cache slot for test",
      hipMemcpy(actual.data(), source, kExpertBytes, hipMemcpyDeviceToHost));
  for (std::size_t byte = 0; byte < actual.size(); ++byte) {
    if (actual[byte] != Pattern(file_offset + byte))
      throw std::runtime_error("packed cache slot byte mismatch");
  }
}

void Run() {
  const auto file_bytes =
      static_cast<std::size_t>(kTensorOffsets.back() + kExperts * kExpertBytes);
  std::vector<std::uint8_t> contents(file_bytes);
  for (std::size_t byte = 0; byte < contents.size(); ++byte)
    contents[byte] = Pattern(byte);

  TempFile fixture;
  fixture.path =
      std::filesystem::temp_directory_path() /
      ("gufo-expert-cache-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".bin");
  {
    std::ofstream output(fixture.path, std::ios::binary | std::ios::trunc);
    if (!output || !output.write(reinterpret_cast<const char*>(contents.data()),
                                 static_cast<std::streamsize>(contents.size())))
      throw std::runtime_error("failed to write expert cache fixture");
  }

  int devices = 0;
  CheckHip("hipGetDeviceCount", hipGetDeviceCount(&devices));
  if (devices == 0)
    throw std::runtime_error("no HIP device found");
  CheckHip("hipSetDevice", hipSetDevice(0));

  using namespace gufo::models::qwen38_flash_next::rocm;
  DeviceLayer layer;
  layer.layer_index = 0;
  layer.experts_streamed = true;
  layer.ffn_gate_exps = {nullptr, gufo::core::GgmlType::kIQ4_XS, 256, 1,
                         kExperts};
  layer.ffn_up_exps = layer.ffn_gate_exps;
  layer.ffn_down_exps = layer.ffn_gate_exps;
  for (std::size_t weight = 0; weight < kTensorOffsets.size(); ++weight) {
    layer.expert_sources[weight] = {fixture.path, kTensorOffsets[weight],
                                    kExpertBytes};
  }

  constexpr std::size_t two_slot_budget =
      kExpertsUsed * 3 * kExpertBytes + 3 * kTailMargin;
  std::array<std::int32_t, kExperts> indirection{};
  indirection.fill(-1);
  std::string error;
  auto manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), two_slot_budget, kExpertsUsed,
      kExperts, indirection, &error);
  if (!manager)
    throw std::runtime_error("cache creation failed: " + error);
  if (manager->slots_per_layer() != kExpertsUsed)
    throw std::runtime_error("cache did not select the expected slot count");
  gufo::storage::ExpertCacheManager::WarmupResult warmup;
  if (!manager->Prewarm(&warmup, &error) || warmup.bytes_read == 0 ||
      warmup.elapsed.count() <= 0 || indirection[0] < 0 || indirection[1] < 0)
    throw std::runtime_error("expert cache prewarm failed: " + error);

  const std::array<std::int32_t, kExpertsUsed> first_ids{0, 1};
  std::array<std::int32_t, kExpertsUsed> first_slots{};
  if (!manager->Resolve(0, first_ids, first_slots, &error))
    throw std::runtime_error("first resolve failed: " + error);
  const auto& cache_view = manager->device_layers()[0];
  for (std::size_t weight = 0; weight < 3; ++weight) {
    const DeviceTensor* tensor = weight == 0   ? &cache_view.ffn_gate_exps
                                 : weight == 1 ? &cache_view.ffn_up_exps
                                               : &cache_view.ffn_down_exps;
    for (std::size_t i = 0; i < first_ids.size(); ++i) {
      CheckExpertBytes(
          *tensor, static_cast<std::uint32_t>(first_slots[i]),
          kTensorOffsets[weight] +
              static_cast<std::uint64_t>(first_ids[i]) * kExpertBytes);
    }
  }
  manager->Release(0);

  const std::array<std::int32_t, kExpertsUsed> second_ids{2, 1};
  std::array<std::int32_t, kExpertsUsed> second_slots{};
  if (!manager->Resolve(0, second_ids, second_slots, &error))
    throw std::runtime_error("second resolve failed: " + error);
  if (second_slots[0] != first_slots[0] || second_slots[1] != first_slots[1] ||
      indirection[2] != first_slots[0] || indirection[1] != first_slots[1] ||
      indirection[0] != -1) {
    throw std::runtime_error(
        "hot/LRU slot indirection did not evict as expected");
  }
  for (std::size_t weight = 0; weight < 3; ++weight) {
    const DeviceTensor* tensor = weight == 0   ? &cache_view.ffn_gate_exps
                                 : weight == 1 ? &cache_view.ffn_up_exps
                                               : &cache_view.ffn_down_exps;
    for (std::size_t i = 0; i < second_ids.size(); ++i) {
      CheckExpertBytes(
          *tensor, static_cast<std::uint32_t>(second_slots[i]),
          kTensorOffsets[weight] +
              static_cast<std::uint64_t>(second_ids[i]) * kExpertBytes);
    }
  }
  manager->Release(0);

  // Count every occurrence in a resolve window, not merely one hit per
  // window, so a frequently routed expert survives a later prefill burst.
  constexpr std::size_t three_slot_budget =
      3 * 3 * kExpertBytes + 3 * kTailMargin;
  std::array<std::int32_t, kExperts> hot_indirection{};
  hot_indirection.fill(-1);
  auto hot_manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), three_slot_budget, kExpertsUsed,
      kExperts, hot_indirection, &error, nullptr, 0.0F,
      gufo::storage::ExpertCacheManager::VictimPolicy::kCumulativeHotness);
  if (!hot_manager || hot_manager->slots_per_layer() != 3)
    throw std::runtime_error("hotness cache creation failed: " + error);
  const std::array<std::int32_t, kExpertsUsed> hot_seed{0, 1};
  std::array<std::int32_t, kExpertsUsed> hot_seed_slots{};
  if (!hot_manager->Resolve(0, hot_seed, hot_seed_slots, &error))
    throw std::runtime_error("hotness seed resolve failed: " + error);
  hot_manager->Release(0);

  std::array<std::int32_t, kExperts> future_indirection{};
  future_indirection.fill(-1);
  auto future_manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), three_slot_budget, kExpertsUsed,
      kExperts, future_indirection, &error);
  if (!future_manager)
    throw std::runtime_error("future-use cache creation failed: " + error);
  const std::array<std::int32_t, 3> future_seed{0, 1, 2};
  std::array<std::int32_t, 3> future_seed_slots{};
  if (!future_manager->Resolve(0, future_seed, future_seed_slots, &error))
    throw std::runtime_error("future-use seed resolve failed: " + error);
  future_manager->Release(0);
  const std::array<std::int32_t, 1> current_window{3};
  const std::array<std::int32_t, 1> future_window{1};
  std::array<std::int32_t, 1> current_slots{};
  if (!future_manager->Resolve(0, current_window, current_slots, &error,
                               future_window) ||
      future_indirection[1] < 0 || future_indirection[0] != -1) {
    throw std::runtime_error(
        "future-use hint did not protect the upcoming expert");
  }
  future_manager->Release(0);
  const std::array<std::int32_t, 4> hot_window{0, 0, 0, 1};
  std::array<std::int32_t, 4> hot_window_slots{};
  if (!hot_manager->Resolve(0, hot_window, hot_window_slots, &error))
    throw std::runtime_error("hotness window resolve failed: " + error);
  hot_manager->Release(0);
  const std::array<std::int32_t, kExpertsUsed> cold_burst{2, 3};
  std::array<std::int32_t, kExpertsUsed> cold_burst_slots{};
  if (!hot_manager->Resolve(0, cold_burst, cold_burst_slots, &error) ||
      hot_indirection[0] < 0 || hot_indirection[1] != -1) {
    throw std::runtime_error(
        "route frequency did not protect the repeatedly activated expert");
  }
  hot_manager->Release(0);

  std::array<std::int32_t, kExperts> decayed_indirection{};
  decayed_indirection.fill(-1);
  auto decayed_manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), three_slot_budget, kExpertsUsed,
      kExperts, decayed_indirection, &error, nullptr, 0.0F,
      gufo::storage::ExpertCacheManager::VictimPolicy::kDecayedHotness, 1);
  if (!decayed_manager)
    throw std::runtime_error("decayed cache creation failed: " + error);
  if (!decayed_manager->Resolve(0, hot_seed, hot_seed_slots, &error, {},
                                nullptr, true, true))
    throw std::runtime_error("decayed cache seed failed: " + error);
  decayed_manager->Release(0);
  if (!decayed_manager->Resolve(0, hot_window, hot_window_slots, &error, {},
                                nullptr, true, true))
    throw std::runtime_error("decayed cache prefill failed: " + error);
  decayed_manager->Release(0);
  if (!decayed_manager->Resolve(0, cold_burst, cold_burst_slots, &error, {},
                                nullptr, true, false) ||
      decayed_indirection[0] != -1 || decayed_indirection[1] < 0) {
    throw std::runtime_error(
        "prefill hotness was not reset on transition to decode");
  }
  decayed_manager->Release(0);

  std::array<std::int32_t, kExperts> desk_indirection{};
  desk_indirection.fill(-1);
  auto desk_manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), three_slot_budget, kExpertsUsed,
      kExperts, desk_indirection, &error);
  if (!desk_manager)
    throw std::runtime_error("desk cache creation failed: " + error);
  const std::array<std::int32_t, 2> desk_experts{0, 1};
  std::array<std::int32_t, 2> desk_slots{};
  if (!desk_manager->PinDesk(0, desk_experts, desk_slots, &error, true) ||
      desk_manager->room_slots_available(0) != 1) {
    throw std::runtime_error("desk slot reservation failed: " + error);
  }
  const std::array<std::int32_t, 2> room_experts{2, 3};
  std::array<std::int32_t, 2> room_slots{};
  if (desk_manager->Resolve(0, room_experts, room_slots, &error)) {
    desk_manager->Release(0);
    throw std::runtime_error("room resolve evicted a protected desk expert");
  }
  const std::array<std::int32_t, 1> room_prediction{2};
  if (!desk_manager->Prefetch(0, room_prediction, 1, &error))
    throw std::runtime_error("room prefetch beside desk failed: " + error);
  const std::array<std::int32_t, 1> prefetched_room{2};
  std::array<std::int32_t, 1> prefetched_room_slot{};
  if (!desk_manager->Resolve(0, prefetched_room, prefetched_room_slot, &error))
    throw std::runtime_error("prefetched room resolve failed: " + error);
  desk_manager->Release(0);
  desk_manager->FinishPrefetch(0);
  if (desk_indirection[0] < 0 || desk_indirection[1] < 0)
    throw std::runtime_error("room prefetch displaced a protected desk expert");
  desk_manager->ReleaseDesk(0, desk_experts);
  if (!desk_manager->Resolve(0, room_experts, room_slots, &error))
    throw std::runtime_error("room resolve failed after releasing desk: " +
                             error);
  desk_manager->Release(0);

  std::array<std::int32_t, kExperts> resident_indirection{};
  resident_indirection.fill(-1);
  auto resident_manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), three_slot_budget, kExpertsUsed,
      kExperts, resident_indirection, &error);
  if (!resident_manager)
    throw std::runtime_error("resident desk cache creation failed: " + error);
  const std::array<std::int32_t, 3> pass_zero_resident{0, 1, 2};
  std::array<std::int32_t, 3> pass_zero_slots{};
  if (!resident_manager->PinDesk(0, pass_zero_resident, pass_zero_slots, &error,
                                 true) ||
      resident_manager->room_slots_available(0) != 0) {
    throw std::runtime_error("Pass 0 did not protect every resident expert: " +
                             error);
  }
  resident_manager->ReleaseDesk(
      0, std::span<const std::int32_t>(pass_zero_resident).subspan(1));
  if (resident_manager->room_slots_available(0) != 2 ||
      resident_indirection[0] < 0) {
    throw std::runtime_error(
        "releasing non-kept Pass 0 slots did not form Room capacity");
  }
  const std::array<std::int32_t, 1> first_room_expert{3};
  std::array<std::int32_t, 1> first_room_slot{};
  if (!resident_manager->Resolve(0, first_room_expert, first_room_slot,
                                 &error) ||
      resident_indirection[0] < 0) {
    throw std::runtime_error("Room evicted the kept Desk expert: " + error);
  }
  resident_manager->Release(0);
  resident_manager->ReleaseDesk(
      0, std::span<const std::int32_t>(pass_zero_resident).first(1));

  std::array<std::int32_t, kExperts> async_indirection{};
  async_indirection.fill(-1);
  auto async_manager = gufo::storage::ExpertCacheManager::Create(
      std::span<const DeviceLayer>(&layer, 1), three_slot_budget, kExpertsUsed,
      kExperts, async_indirection, &error);
  if (!async_manager)
    throw std::runtime_error("async cache creation failed: " + error);
  if (async_manager->slots_per_layer() != 3)
    throw std::runtime_error("async cache did not allocate three slots");
  const std::array<std::int32_t, 1> predicted{2};
  if (!async_manager->Prefetch(0, predicted, 8, &error))
    throw std::runtime_error("async prefetch submission failed: " + error);
  const std::array<std::int32_t, kExpertsUsed> async_ids{2, 3};
  std::array<std::int32_t, kExpertsUsed> async_slots{};
  if (!async_manager->Resolve(0, async_ids, async_slots, &error))
    throw std::runtime_error("async resolve failed: " + error);
  const auto& async_view = async_manager->device_layers()[0];
  for (std::size_t weight = 0; weight < 3; ++weight) {
    const DeviceTensor* tensor = weight == 0   ? &async_view.ffn_gate_exps
                                 : weight == 1 ? &async_view.ffn_up_exps
                                               : &async_view.ffn_down_exps;
    for (std::size_t i = 0; i < async_ids.size(); ++i) {
      CheckExpertBytes(
          *tensor, static_cast<std::uint32_t>(async_slots[i]),
          kTensorOffsets[weight] +
              static_cast<std::uint64_t>(async_ids[i]) * kExpertBytes);
    }
  }
  async_manager->Release(0);
  async_manager->FinishPrefetch(0);

  std::array<DeviceLayer, 3> speculative_layers{layer, layer, layer};
  for (std::uint32_t i = 0; i < speculative_layers.size(); ++i)
    speculative_layers[i].layer_index = i;
  std::array<std::int32_t, 3 * kExperts> speculative_indirection{};
  speculative_indirection.fill(-1);
  constexpr std::size_t three_layer_budget = 256 * 1024;
  auto speculative_manager = gufo::storage::ExpertCacheManager::Create(
      speculative_layers, three_layer_budget, kExpertsUsed, kExperts,
      speculative_indirection, &error);
  if (!speculative_manager)
    throw std::runtime_error("speculative cache creation failed: " + error);
  const std::array<std::int32_t, 1> speculative_prediction{2};
  for (std::uint32_t i = 0;
       i < gufo::storage::ExpertCacheManager::kSpeculativeTransactionCap; ++i) {
    if (!speculative_manager->Prefetch(i, speculative_prediction, 1, &error,
                                       false, true)) {
      throw std::runtime_error("speculative prefetch submission failed: " +
                               error);
    }
  }
  const std::size_t third_layer_map = 2 * kExperts + 2;
  if (!speculative_manager->Prefetch(2, speculative_prediction, 1, &error,
                                     false, true) ||
      speculative_indirection[third_layer_map] != -1) {
    throw std::runtime_error(
        "speculative prefetch exceeded its model-wide transaction cap");
  }
  // Keep k+1 outstanding while retiring k; the rolling horizon must admit k+2.
  speculative_manager->FinishPrefetch(0);
  if (!speculative_manager->Prefetch(2, speculative_prediction, 1, &error,
                                     false, true) ||
      speculative_indirection[third_layer_map] < 0) {
    throw std::runtime_error(
        "rolling speculative credit did not admit layer k+2 after finishing "
        "layer k: " +
        error);
  }
  speculative_manager->FinishPrefetch(1);
  speculative_manager->FinishPrefetch(2);

  // A prediction that is already resident submits no I/O batch. Resolve must
  // consume the READY pin without waiting for a nonexistent completion.
  if (!async_manager->Prefetch(
          0, std::span<const std::int32_t>(async_ids).first(1), 8, &error))
    throw std::runtime_error("resident prefetch failed: " + error);
  const std::array<std::int32_t, 1> resident_id{2};
  std::array<std::int32_t, 1> resident_slot{};
  if (!async_manager->Resolve(0, resident_id, resident_slot, &error))
    throw std::runtime_error("resident prefetch resolve failed: " + error);
  async_manager->Release(0);
  async_manager->FinishPrefetch(0);

  std::cout
      << "expert_cache_manager_test: packed GTT arenas, asynchronous and empty "
         "prefetch batches, speculative cap, scratch reads, pin release, and "
         "LRU indirection passed\n";
}

}  // namespace

int main() {
  try {
    Run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert_cache_manager_test: " << error.what() << '\n';
    return 1;
  }
}
