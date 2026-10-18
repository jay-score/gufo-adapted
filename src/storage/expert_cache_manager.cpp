#include "src/storage/expert_cache_manager.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "src/storage/ssd_batch_reader.hpp"

namespace gufo::storage {
namespace {

namespace qwenrocm = gufo::models::qwen38_flash_next::rocm;
using qwenrocm::DeviceLayer;
using qwenrocm::DeviceTensor;

constexpr std::size_t kTailMargin = 4096;
constexpr std::size_t kReaderThreads = 8;

enum class SlotState : std::uint8_t {
  kFree,
  kInFlightPinned,
  kReadyPinned,
  kCachedUnpinned,
};

struct Slot {
  std::atomic<SlotState> state{SlotState::kFree};
  std::atomic<std::uint32_t> pin_refcount{0};
  std::int32_t expert_id{-1};
  std::uint64_t last_used{0};
};

struct CacheLayer {
  DeviceLayer view;
  std::uint32_t num_experts{0};
  std::uint32_t slots{0};
  std::unique_ptr<Slot[]> slot_state;
  std::array<void*, 3> host_arenas{};
  std::array<void*, 3> device_arenas{};
  std::array<std::size_t, 3> expert_bytes{};
  std::vector<std::uint32_t> route_hotness;
  std::vector<std::uint32_t> pinned_slots;
  std::uint64_t use_counter{0};

  ~CacheLayer() {
    for (void* arena : host_arenas) {
      if (arena != nullptr) {
        const hipError_t status = hipHostFree(arena);
        if (status != hipSuccess)
          std::fprintf(stderr, "expert cache hipHostFree failed: %s\n",
                       hipGetErrorString(status));
      }
    }
  }
};

DeviceTensor* ExpertTensor(DeviceLayer& layer, std::size_t index) {
  switch (index) {
    case 0:
      return &layer.ffn_gate_exps;
    case 1:
      return &layer.ffn_up_exps;
    default:
      return &layer.ffn_down_exps;
  }
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
}

}  // namespace

struct ExpertCacheManager::Impl {
  std::vector<DeviceLayer> views;
  std::vector<std::unique_ptr<CacheLayer>> caches;
  std::span<std::int32_t> expert_id_to_slot;
  std::uint32_t num_experts{0};
  std::uint32_t experts_used{0};
  std::uint32_t slots_per_layer{0};
  std::size_t allocated_bytes{0};
  std::size_t max_expert_bytes{0};
  std::map<std::filesystem::path, std::unique_ptr<SsdBatchReader>> readers;
  std::mutex mutex;

  [[nodiscard]] std::int32_t& Mapping(std::uint32_t layer,
                                      std::uint32_t expert) {
    return expert_id_to_slot[static_cast<std::size_t>(layer) * num_experts +
                             expert];
  }

  [[nodiscard]] std::int32_t ChooseVictim(CacheLayer& cache) const {
    std::int32_t victim = -1;
    for (std::uint32_t slot = 0; slot < cache.slots; ++slot) {
      Slot& candidate = cache.slot_state[slot];
      const auto state = candidate.state.load(std::memory_order_acquire);
      if (state == SlotState::kFree)
        return static_cast<std::int32_t>(slot);
      if (state != SlotState::kCachedUnpinned ||
          candidate.pin_refcount.load(std::memory_order_acquire) != 0)
        continue;
      if (victim < 0) {
        victim = static_cast<std::int32_t>(slot);
        continue;
      }
      const Slot& current = cache.slot_state[static_cast<std::size_t>(victim)];
      const auto hot = cache.route_hotness[candidate.expert_id];
      const auto current_hot = cache.route_hotness[current.expert_id];
      if (hot < current_hot ||
          (hot == current_hot && candidate.last_used < current.last_used))
        victim = static_cast<std::int32_t>(slot);
    }
    return victim;
  }

  void ReleaseLocked(CacheLayer& cache) noexcept {
    for (std::uint32_t slot_id : cache.pinned_slots) {
      Slot& slot = cache.slot_state[slot_id];
      const auto previous =
          slot.pin_refcount.fetch_sub(1, std::memory_order_acq_rel);
      if (previous == 1)
        slot.state.store(SlotState::kCachedUnpinned, std::memory_order_release);
    }
    cache.pinned_slots.clear();
  }

  void Rollback(CacheLayer& cache,
                std::span<const std::uint32_t> newly_reserved) noexcept {
    ReleaseLocked(cache);
    for (std::uint32_t slot_id : newly_reserved) {
      Slot& slot = cache.slot_state[slot_id];
      if (slot.expert_id >= 0 &&
          Mapping(cache.view.layer_index,
                  static_cast<std::uint32_t>(slot.expert_id)) ==
              static_cast<std::int32_t>(slot_id)) {
        Mapping(cache.view.layer_index,
                static_cast<std::uint32_t>(slot.expert_id)) = -1;
      }
      slot.expert_id = -1;
      slot.pin_refcount.store(0, std::memory_order_release);
      slot.state.store(SlotState::kFree, std::memory_order_release);
    }
  }

  SsdBatchReader* Reader(const std::filesystem::path& path,
                         std::string* error_msg) {
    auto found = readers.find(path);
    if (found != readers.end())
      return found->second.get();
    try {
      SsdBatchReader::Options options;
      options.io_threads = kReaderThreads;
      options.max_scratch_bytes = std::max<std::size_t>(max_expert_bytes, 4096);
      auto reader = std::make_unique<SsdBatchReader>(path, options);
      auto* raw = reader.get();
      readers.emplace(path, std::move(reader));
      return raw;
    } catch (const std::exception& error) {
      SetError(error_msg, "opening expert shard reader failed: " +
                              std::string(error.what()));
      return nullptr;
    }
  }
};

ExpertCacheManager::ExpertCacheManager() : impl_(std::make_unique<Impl>()) {}
ExpertCacheManager::~ExpertCacheManager() = default;

std::unique_ptr<ExpertCacheManager> ExpertCacheManager::Create(
    std::span<const DeviceLayer> layers, std::size_t budget_bytes,
    std::uint32_t experts_used, std::uint32_t num_experts,
    std::span<std::int32_t> expert_id_to_slot, std::string* error_msg) {
  auto manager = std::unique_ptr<ExpertCacheManager>(new ExpertCacheManager());
  auto& impl = *manager->impl_;
  impl.views.assign(layers.begin(), layers.end());
  impl.caches.resize(layers.size());
  impl.num_experts = num_experts;
  impl.experts_used = experts_used;
  impl.expert_id_to_slot = expert_id_to_slot;
  if (num_experts == 0 || experts_used == 0 || experts_used > num_experts ||
      layers.size() > std::numeric_limits<std::size_t>::max() / num_experts ||
      expert_id_to_slot.size() < layers.size() * num_experts) {
    SetError(error_msg, "expert cache mapping table has invalid dimensions");
    return nullptr;
  }
  std::fill_n(expert_id_to_slot.begin(), layers.size() * num_experts, -1);

  std::size_t bytes_per_all_layer_slot = 0;
  std::size_t tail_bytes = 0;
  std::size_t streamed_layers = 0;
  for (std::size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
    const auto& layer = layers[layer_id];
    if (!layer.experts_streamed)
      continue;
    if (layer.layer_index != layer_id) {
      SetError(error_msg, "streamed expert layer IDs are not contiguous");
      return nullptr;
    }
    ++streamed_layers;
    std::uint32_t layer_experts = layer.ffn_gate_exps.experts;
    for (std::size_t weight = 0; weight < layer.expert_sources.size();
         ++weight) {
      const auto& source = layer.expert_sources[weight];
      if (source.file_path.empty() || source.expert_bytes == 0 ||
          layer_experts != num_experts ||
          source.expert_bytes > std::numeric_limits<std::size_t>::max() -
                                    bytes_per_all_layer_slot ||
          tail_bytes > std::numeric_limits<std::size_t>::max() - kTailMargin) {
        SetError(error_msg, "streamed expert source has invalid geometry");
        return nullptr;
      }
      bytes_per_all_layer_slot += source.expert_bytes;
      tail_bytes += kTailMargin;
      impl.max_expert_bytes =
          std::max(impl.max_expert_bytes, source.expert_bytes);
    }
  }
  if (streamed_layers == 0 || bytes_per_all_layer_slot == 0) {
    SetError(error_msg, "model has no routed experts to stream");
    return nullptr;
  }
  if (budget_bytes <= tail_bytes) {
    SetError(error_msg, "expert cache budget cannot hold one slot per tensor");
    return nullptr;
  }
  const std::size_t possible_slots =
      (budget_bytes - tail_bytes) / bytes_per_all_layer_slot;
  const std::uint32_t slots = static_cast<std::uint32_t>(
      std::min<std::size_t>(num_experts, possible_slots));
  if (slots < experts_used) {
    SetError(error_msg,
             "expert cache budget cannot hold the active layer's top-k set");
    return nullptr;
  }
  impl.slots_per_layer = slots;

  try {
    for (std::size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
      if (!layers[layer_id].experts_streamed)
        continue;
      auto cache = std::make_unique<CacheLayer>();
      cache->view = layers[layer_id];
      cache->num_experts = num_experts;
      cache->slots = slots;
      cache->view.expert_cache_slots = slots;
      cache->slot_state = std::make_unique<Slot[]>(slots);
      cache->route_hotness.resize(num_experts);
      cache->pinned_slots.reserve(experts_used);
      for (std::size_t weight = 0; weight < cache->expert_bytes.size();
           ++weight) {
        auto* tensor = ExpertTensor(cache->view, weight);
        const auto& source = cache->view.expert_sources[weight];
        cache->expert_bytes[weight] = source.expert_bytes;
        if (slots > (std::numeric_limits<std::size_t>::max() - kTailMargin) /
                        source.expert_bytes) {
          SetError(error_msg,
                   "expert arena size overflows the host size range");
          return nullptr;
        }
        const std::size_t arena_bytes =
            static_cast<std::size_t>(slots) * source.expert_bytes + kTailMargin;
        if (hipHostMalloc(&cache->host_arenas[weight], arena_bytes,
                          hipHostMallocMapped) != hipSuccess) {
          SetError(error_msg, "hipHostMalloc failed for expert slot arena");
          return nullptr;
        }
        if (reinterpret_cast<std::uintptr_t>(cache->host_arenas[weight]) %
                SsdBatchReader::Alignment() !=
            0) {
          SetError(error_msg, "HIP expert arena is not 4096-byte aligned");
          return nullptr;
        }
        if (hipHostGetDevicePointer(&cache->device_arenas[weight],
                                    cache->host_arenas[weight],
                                    0) != hipSuccess) {
          SetError(error_msg,
                   "hipHostGetDevicePointer failed for expert slot arena");
          return nullptr;
        }
        tensor->data = cache->device_arenas[weight];
        tensor->experts = slots;
        if (arena_bytes >
            std::numeric_limits<std::size_t>::max() - impl.allocated_bytes) {
          SetError(error_msg, "expert cache allocation total overflows");
          return nullptr;
        }
        impl.allocated_bytes += arena_bytes;
      }
      impl.caches[layer_id] = std::move(cache);
      impl.views[layer_id] = impl.caches[layer_id]->view;
    }
  } catch (const std::exception& error) {
    SetError(error_msg,
             "initializing expert cache failed: " + std::string(error.what()));
    return nullptr;
  }
  return manager;
}

const std::vector<DeviceLayer>& ExpertCacheManager::device_layers()
    const noexcept {
  return impl_->views;
}

std::uint32_t ExpertCacheManager::slots_per_layer() const noexcept {
  return impl_->slots_per_layer;
}

std::size_t ExpertCacheManager::allocated_bytes() const noexcept {
  return impl_->allocated_bytes;
}

bool ExpertCacheManager::Resolve(std::uint32_t layer_id,
                                 std::span<const std::int32_t> expert_ids,
                                 std::span<std::int32_t> slot_ids,
                                 std::string* error_msg) {
  if (error_msg != nullptr)
    error_msg->clear();
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr ||
      expert_ids.size() != slot_ids.size()) {
    SetError(error_msg, "expert cache resolve has invalid dimensions");
    return false;
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  if (!cache.pinned_slots.empty()) {
    SetError(error_msg, "expert cache layer still has pinned slots");
    return false;
  }

  std::vector<std::int32_t> unique;
  unique.reserve(expert_ids.size());
  std::vector<std::uint8_t> seen(cache.num_experts, 0);
  for (const auto id : expert_ids) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= cache.num_experts) {
      SetError(error_msg, "router selected an out-of-range expert ID");
      return false;
    }
    if (!seen[static_cast<std::uint32_t>(id)]) {
      seen[static_cast<std::uint32_t>(id)] = 1;
      unique.push_back(id);
    }
  }
  if (unique.size() > cache.slots) {
    SetError(error_msg, "active expert set exceeds the layer slot capacity");
    return false;
  }

  std::vector<std::pair<std::int32_t, std::uint32_t>> misses;
  std::vector<std::uint32_t> newly_reserved;
  misses.reserve(unique.size());
  newly_reserved.reserve(unique.size());
  cache.pinned_slots.reserve(unique.size());
  for (const auto expert : unique) {
    const auto mapped =
        impl_->expert_id_to_slot[static_cast<std::size_t>(layer_id) *
                                     impl_->num_experts +
                                 expert];
    if (mapped >= 0) {
      Slot& slot = cache.slot_state[static_cast<std::uint32_t>(mapped)];
      const auto state = slot.state.load(std::memory_order_acquire);
      if (state != SlotState::kCachedUnpinned ||
          slot.pin_refcount.load(std::memory_order_acquire) != 0) {
        SetError(error_msg, "expert cache slot is already pinned or in flight");
        impl_->Rollback(cache, newly_reserved);
        return false;
      }
      slot.pin_refcount.store(1, std::memory_order_release);
      slot.state.store(SlotState::kReadyPinned, std::memory_order_release);
      slot.last_used = ++cache.use_counter;
      cache.pinned_slots.push_back(static_cast<std::uint32_t>(mapped));
      ++cache.route_hotness[static_cast<std::uint32_t>(expert)];
      continue;
    }

    const std::int32_t victim = impl_->ChooseVictim(cache);
    if (victim < 0) {
      SetError(error_msg, "no unpinned expert slot is available");
      impl_->Rollback(cache, newly_reserved);
      return false;
    }
    Slot& slot = cache.slot_state[static_cast<std::uint32_t>(victim)];
    if (slot.expert_id >= 0) {
      impl_->expert_id_to_slot[static_cast<std::size_t>(layer_id) *
                                   impl_->num_experts +
                               static_cast<std::uint32_t>(slot.expert_id)] = -1;
    }
    slot.expert_id = expert;
    slot.last_used = ++cache.use_counter;
    slot.pin_refcount.store(1, std::memory_order_release);
    slot.state.store(SlotState::kInFlightPinned, std::memory_order_release);
    impl_->expert_id_to_slot[static_cast<std::size_t>(layer_id) *
                                 impl_->num_experts +
                             static_cast<std::uint32_t>(expert)] = victim;
    cache.pinned_slots.push_back(static_cast<std::uint32_t>(victim));
    misses.emplace_back(expert, static_cast<std::uint32_t>(victim));
    newly_reserved.push_back(static_cast<std::uint32_t>(victim));
    ++cache.route_hotness[static_cast<std::uint32_t>(expert)];
  }

  struct PendingRead {
    std::size_t weight;
    std::uint32_t slot;
    SsdReadRequest request;
  };
  std::map<std::filesystem::path, std::vector<PendingRead>> grouped;
  for (const auto [expert, slot] : misses) {
    for (std::size_t weight = 0; weight < cache.expert_bytes.size(); ++weight) {
      const auto& source = cache.view.expert_sources[weight];
      if (static_cast<std::uint64_t>(expert) >
          (std::numeric_limits<std::uint64_t>::max() - source.file_offset) /
              source.expert_bytes) {
        SetError(error_msg, "expert source file offset overflows");
        impl_->Rollback(cache, newly_reserved);
        return false;
      }
      auto* destination =
          static_cast<std::uint8_t*>(cache.host_arenas[weight]) +
          static_cast<std::size_t>(slot) * cache.expert_bytes[weight];
      grouped[source.file_path].push_back(
          {weight,
           slot,
           {.file_offset =
                source.file_offset +
                static_cast<std::uint64_t>(expert) * source.expert_bytes,
            .bytes = source.expert_bytes,
            .destination = destination,
            .destination_capacity = source.expert_bytes}});
    }
  }

  for (auto& [path, reads] : grouped) {
    auto reader = impl_->Reader(path, error_msg);
    if (reader == nullptr) {
      impl_->Rollback(cache, newly_reserved);
      return false;
    }
    std::vector<SsdReadRequest> requests;
    requests.reserve(reads.size());
    for (auto& read : reads)
      requests.push_back(read.request);
    if (!reader->ReadBatch(requests, error_msg)) {
      impl_->Rollback(cache, newly_reserved);
      return false;
    }
  }

  for (const auto [expert, slot_id] : misses) {
    (void)expert;
    cache.slot_state[slot_id].state.store(SlotState::kReadyPinned,
                                          std::memory_order_release);
  }
  for (std::size_t i = 0; i < expert_ids.size(); ++i) {
    slot_ids[i] =
        impl_->expert_id_to_slot[static_cast<std::size_t>(layer_id) *
                                     impl_->num_experts +
                                 static_cast<std::uint32_t>(expert_ids[i])];
    if (slot_ids[i] < 0) {
      SetError(error_msg, "expert indirection entry is missing");
      impl_->Rollback(cache, newly_reserved);
      return false;
    }
  }
  return true;
}

void ExpertCacheManager::Release(std::uint32_t layer_id) noexcept {
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
    return;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->ReleaseLocked(*impl_->caches[layer_id]);
}

}  // namespace gufo::storage
