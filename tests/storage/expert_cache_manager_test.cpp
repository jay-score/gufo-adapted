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

  constexpr std::size_t three_slot_budget =
      3 * 3 * kExpertBytes + 3 * kTailMargin;
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
         "prefetch batches, scratch reads, pin release, and LRU indirection "
         "passed\n";
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
