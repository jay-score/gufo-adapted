#include "src/storage/expert_cache_manager.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "src/storage/ssd_batch_reader.hpp"
#include "src/storage/ssd_stats.hpp"

namespace gufo::storage {
namespace {

namespace qwenrocm = gufo::models::qwen38_flash_next::rocm;
using qwenrocm::DeviceLayer;
using qwenrocm::DeviceTensor;

constexpr std::size_t kTailMargin = 4096;
constexpr std::size_t kReaderThreads = 8;
constexpr std::size_t kReaderInflight = 32;

enum class SlotState : std::uint8_t {
  kFree,
  kInFlightPinned,
  kReadyPinned,
  kCachedUnpinned,
};

struct Slot {
  std::atomic<SlotState> state{SlotState::kFree};
  std::atomic<std::uint32_t> pin_refcount{0};
  std::atomic<std::uint32_t> pending_reads{0};
  std::atomic<bool> read_failed{false};
  std::int32_t expert_id{-1};
  std::uint64_t last_used{0};
  std::uint64_t generation{0};
  std::uint32_t future_uses{0};
  bool prefetch_pinned{false};
  bool prefetch_origin{false};
  bool desk_reserved{false};
};

struct PrefetchJob {
  std::shared_future<SsdReadBatchResult> result;
  SsdBatchReader* reader{nullptr};
  SsdBatchReader::BatchId batch_id{0};
  std::vector<std::uint32_t> slots;
  std::vector<std::uint64_t> generations;
  bool prefill{false};
  bool speculative{true};
  bool finalized{false};
};

struct TokenPrior {
  std::vector<std::int32_t> experts;
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
  std::vector<std::uint64_t> route_hotness;
  std::vector<float> prior_score;
  std::vector<std::uint32_t> pinned_slots;
  std::vector<std::uint32_t> prefetch_slots;
  std::vector<std::shared_ptr<PrefetchJob>> prefetch_jobs;
  std::unordered_map<std::int32_t, TokenPrior> token_priors;
  std::uint64_t prior_clock{0};
  std::uint64_t use_counter{0};
  std::uint64_t route_tokens_since_decay{0};
  bool has_route_phase{false};
  bool last_route_was_prefill{false};

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
  std::size_t reader_count{1};
  std::shared_ptr<SsdScratchPool> scratch_pool;
  std::map<std::filesystem::path, std::unique_ptr<SsdBatchReader>> readers;
  std::mutex mutex;
  SsdStats* stats{nullptr};
  float cache_prior_bias{0.0F};
  ExpertCacheManager::VictimPolicy victim_policy{
      ExpertCacheManager::VictimPolicy::kLru};
  std::uint32_t hotness_decay_tokens{1024};

  [[nodiscard]] std::int32_t& Mapping(std::uint32_t layer,
                                      std::uint32_t expert) {
    return expert_id_to_slot[static_cast<std::size_t>(layer) * num_experts +
                             expert];
  }

  [[nodiscard]] std::int32_t ChooseVictim(CacheLayer& cache) const {
    std::int32_t victim = -1;
    std::int32_t protected_victim = -1;
    for (std::uint32_t slot = 0; slot < cache.slots; ++slot) {
      Slot& candidate = cache.slot_state[slot];
      const auto state = candidate.state.load(std::memory_order_acquire);
      if (state == SlotState::kFree)
        return static_cast<std::int32_t>(slot);
      if (state != SlotState::kCachedUnpinned ||
          candidate.pin_refcount.load(std::memory_order_acquire) != 0 ||
          candidate.desk_reserved)
        continue;
      auto less_valuable = [&](std::int32_t previous) {
        const Slot& current =
            cache.slot_state[static_cast<std::size_t>(previous)];
        if (candidate.future_uses != current.future_uses)
          return candidate.future_uses < current.future_uses;
        const double candidate_hotness =
            victim_policy == ExpertCacheManager::VictimPolicy::kLru
                ? 0.0
                : static_cast<double>(cache.route_hotness[candidate.expert_id]);
        const double current_hotness =
            victim_policy == ExpertCacheManager::VictimPolicy::kLru
                ? 0.0
                : static_cast<double>(cache.route_hotness[current.expert_id]);
        const double hot =
            candidate_hotness +
            cache_prior_bias * cache.prior_score[candidate.expert_id];
        const double current_hot =
            current_hotness +
            cache_prior_bias * cache.prior_score[current.expert_id];
        return hot < current_hot ||
               (hot == current_hot && candidate.last_used < current.last_used);
      };
      if (candidate.future_uses != 0) {
        if (protected_victim < 0 || less_valuable(protected_victim))
          protected_victim = static_cast<std::int32_t>(slot);
        continue;
      }
      if (victim < 0) {
        victim = static_cast<std::int32_t>(slot);
        continue;
      }
      if (less_valuable(victim))
        victim = static_cast<std::int32_t>(slot);
    }
    return victim >= 0 ? victim : protected_victim;
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
      options.io_threads = std::max<std::size_t>(
          1, (kReaderThreads + reader_count - 1) / reader_count);
      options.max_inflight_reads = std::max<std::size_t>(1, kReaderInflight);
      if (max_expert_bytes > (std::numeric_limits<std::size_t>::max() -
                              2 * SsdBatchReader::Alignment()) /
                                 2) {
        throw std::overflow_error("coalesced expert scratch size overflows");
      }
      options.max_scratch_bytes = std::max<std::size_t>(
          2 * max_expert_bytes + 2 * SsdBatchReader::Alignment(), 4096);
      options.scratch_pool = scratch_pool;
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

  bool FinishPrefetchJob(CacheLayer& cache,
                         const std::shared_ptr<PrefetchJob>& job, bool wait) {
    if (job->finalized)
      return true;
    if (!wait && job->result.wait_for(std::chrono::nanoseconds{0}) !=
                     std::future_status::ready) {
      return false;
    }
    if (wait)
      job->result.wait();

    SsdReadBatchResult result;
    bool got_result = true;
    try {
      result = job->result.get();
    } catch (...) {
      got_result = false;
    }
    std::uint64_t transferred = 0;
    if (got_result) {
      transferred = result.bytes_read;
      if (stats != nullptr)
        stats->RecordStorageRead(transferred, result.elapsed, job->prefill,
                                 job->speculative);
    }

    for (std::size_t i = 0; i < job->slots.size(); ++i) {
      const std::uint32_t slot_id = job->slots[i];
      Slot& slot = cache.slot_state[slot_id];
      if (slot.generation != job->generations[i])
        continue;
      if (!got_result || i >= result.requests.size() ||
          !result.requests[i].completed) {
        slot.read_failed.store(true, std::memory_order_release);
      }
      const std::uint32_t pending =
          slot.pending_reads.fetch_sub(1, std::memory_order_acq_rel);
      if (pending != 1)
        continue;
      if (slot.read_failed.load(std::memory_order_acquire)) {
        if (slot.expert_id >= 0 &&
            Mapping(cache.view.layer_index,
                    static_cast<std::uint32_t>(slot.expert_id)) ==
                static_cast<std::int32_t>(slot_id)) {
          Mapping(cache.view.layer_index,
                  static_cast<std::uint32_t>(slot.expert_id)) = -1;
        }
        slot.expert_id = -1;
        slot.prefetch_pinned = false;
        slot.prefetch_origin = false;
        slot.pin_refcount.store(0, std::memory_order_release);
        slot.state.store(SlotState::kFree, std::memory_order_release);
      } else if (slot.prefetch_pinned) {
        slot.state.store(SlotState::kReadyPinned, std::memory_order_release);
      } else {
        slot.prefetch_origin = false;
        slot.pin_refcount.store(0, std::memory_order_release);
        slot.state.store(SlotState::kCachedUnpinned, std::memory_order_release);
      }
    }
    job->finalized = true;
    return true;
  }

  void ReapPrefetchJobs(CacheLayer& cache) {
    for (const auto& job : cache.prefetch_jobs)
      (void)FinishPrefetchJob(cache, job, false);
    std::erase_if(cache.prefetch_jobs,
                  [](const auto& job) { return job->finalized; });
    std::erase_if(cache.prefetch_slots, [&](std::uint32_t slot_id) {
      const Slot& slot = cache.slot_state[slot_id];
      return !slot.prefetch_pinned &&
             slot.state.load(std::memory_order_acquire) !=
                 SlotState::kInFlightPinned;
    });
  }

  void DrainPrefetch() {
    const std::lock_guard<std::mutex> lock(mutex);
    for (auto& cache_ptr : caches) {
      if (cache_ptr == nullptr)
        continue;
      CacheLayer& cache = *cache_ptr;
      for (const auto& job : cache.prefetch_jobs)
        (void)FinishPrefetchJob(cache, job, true);
      cache.prefetch_jobs.clear();
    }
  }
};

ExpertCacheManager::ExpertCacheManager() : impl_(std::make_unique<Impl>()) {}
ExpertCacheManager::~ExpertCacheManager() {
  if (impl_ != nullptr)
    impl_->DrainPrefetch();
}

std::unique_ptr<ExpertCacheManager> ExpertCacheManager::Create(
    std::span<const DeviceLayer> layers, std::size_t budget_bytes,
    std::uint32_t experts_used, std::uint32_t num_experts,
    std::span<std::int32_t> expert_id_to_slot, std::string* error_msg,
    SsdStats* stats, float cache_prior_bias, VictimPolicy victim_policy,
    std::uint32_t hotness_decay_tokens) {
  auto manager = std::unique_ptr<ExpertCacheManager>(new ExpertCacheManager());
  auto& impl = *manager->impl_;
  impl.views.assign(layers.begin(), layers.end());
  impl.caches.resize(layers.size());
  impl.num_experts = num_experts;
  impl.experts_used = experts_used;
  impl.expert_id_to_slot = expert_id_to_slot;
  impl.stats = stats;
  impl.cache_prior_bias = cache_prior_bias;
  impl.victim_policy = victim_policy;
  impl.hotness_decay_tokens = std::max(1U, hotness_decay_tokens);
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
  std::set<std::filesystem::path> source_paths;
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
      source_paths.insert(source.file_path);
    }
  }
  if (streamed_layers == 0 || bytes_per_all_layer_slot == 0) {
    SetError(error_msg, "model has no routed experts to stream");
    return nullptr;
  }
  impl.reader_count = std::max<std::size_t>(source_paths.size(), 1);
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
    const std::size_t shared_scratch_bytes = SharedScratchBytes(layers);
    const std::size_t scratch_buffer_bytes =
        shared_scratch_bytes / kSharedScratchBufferCount;
    impl.scratch_pool =
        SsdScratchPool::Create(kSharedScratchBufferCount, scratch_buffer_bytes);
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
      cache->prior_score.resize(num_experts);
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

std::uint32_t ExpertCacheManager::demand_slots_available(
    std::uint32_t layer_id) const noexcept {
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
    return impl_->slots_per_layer;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const CacheLayer& cache = *impl_->caches[layer_id];
  std::uint32_t unavailable = 0;
  for (std::uint32_t slot_id = 0; slot_id < cache.slots; ++slot_id) {
    const Slot& slot = cache.slot_state[slot_id];
    if (slot.prefetch_pinned || slot.state.load(std::memory_order_acquire) ==
                                    SlotState::kInFlightPinned) {
      ++unavailable;
    }
  }
  return cache.slots - unavailable;
}

std::uint32_t ExpertCacheManager::room_slots_available(
    std::uint32_t layer_id) const noexcept {
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
    return impl_->slots_per_layer;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  const CacheLayer& cache = *impl_->caches[layer_id];
  std::uint32_t unavailable = 0;
  for (std::uint32_t slot_id = 0; slot_id < cache.slots; ++slot_id) {
    const Slot& slot = cache.slot_state[slot_id];
    if (slot.desk_reserved || slot.prefetch_pinned ||
        slot.state.load(std::memory_order_acquire) ==
            SlotState::kInFlightPinned)
      ++unavailable;
  }
  return cache.slots - unavailable;
}

std::size_t ExpertCacheManager::allocated_bytes() const noexcept {
  return impl_->allocated_bytes;
}

std::size_t ExpertCacheManager::SharedScratchBytes(
    std::span<const DeviceLayer> layers) {
  std::size_t max_expert_bytes = 0;
  for (const auto& layer : layers) {
    if (!layer.experts_streamed)
      continue;
    for (const auto& source : layer.expert_sources)
      max_expert_bytes = std::max(max_expert_bytes, source.expert_bytes);
  }
  if (max_expert_bytes == 0 ||
      max_expert_bytes > (std::numeric_limits<std::size_t>::max() -
                          2 * SsdBatchReader::Alignment()) /
                             2) {
    throw std::invalid_argument("streamed expert scratch geometry is invalid");
  }
  const std::size_t max_group_bytes =
      2 * max_expert_bytes + 2 * SsdBatchReader::Alignment();
  const std::size_t buffer_bytes =
      SsdScratchPool::EstimateBufferBytes(max_group_bytes);
  if (buffer_bytes >
      std::numeric_limits<std::size_t>::max() / kSharedScratchBufferCount) {
    throw std::overflow_error("shared expert scratch pool size overflows");
  }
  return buffer_bytes * kSharedScratchBufferCount;
}

bool ExpertCacheManager::Prewarm(WarmupResult* result, std::string* error_msg) {
  if (result != nullptr)
    *result = {};
  if (error_msg != nullptr)
    error_msg->clear();
  const auto start = std::chrono::steady_clock::now();
  std::uint64_t bytes_read = 0;
  for (std::uint32_t layer_id = 0; layer_id < impl_->caches.size();
       ++layer_id) {
    if (impl_->caches[layer_id] == nullptr)
      continue;
    const std::uint32_t slots = impl_->caches[layer_id]->slots;
    std::vector<std::int32_t> experts(slots);
    std::vector<std::int32_t> slot_ids(slots);
    for (std::uint32_t slot = 0; slot < slots; ++slot)
      experts[slot] = static_cast<std::int32_t>(slot);
    std::uint64_t layer_bytes = 0;
    if (!Resolve(layer_id, experts, slot_ids, error_msg, {}, &layer_bytes,
                 false)) {
      Release(layer_id);
      if (result != nullptr) {
        result->bytes_read = bytes_read + layer_bytes;
        result->elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start);
      }
      return false;
    }
    Release(layer_id);
    if (layer_bytes > std::numeric_limits<std::uint64_t>::max() - bytes_read) {
      SetError(error_msg, "expert cache prewarm byte count overflows");
      return false;
    }
    bytes_read += layer_bytes;
  }
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto& cache : impl_->caches) {
      if (cache != nullptr)
        std::fill(cache->route_hotness.begin(), cache->route_hotness.end(), 0);
    }
  }
  if (result != nullptr) {
    result->bytes_read = bytes_read;
    result->elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start);
  }
  return true;
}

void ExpertCacheManager::FinishAllPrefetch() noexcept {
  for (std::uint32_t layer_id = 0; layer_id < impl_->caches.size(); ++layer_id)
    FinishPrefetch(layer_id);
}

bool ExpertCacheManager::Prefetch(std::uint32_t layer_id,
                                  std::span<const std::int32_t> expert_ids,
                                  std::size_t max_slots, std::string* error_msg,
                                  bool prefill) {
  if (error_msg != nullptr)
    error_msg->clear();
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr) {
    SetError(error_msg, "expert cache prefetch has an invalid layer");
    return false;
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  impl_->ReapPrefetchJobs(cache);
  // Stage A and Stage B may predict the same future layer. Keep the first
  // batch intact; never mutate its slot vector or submit a duplicate flight.
  if (!cache.prefetch_slots.empty() ||
      std::any_of(cache.slot_state.get(), cache.slot_state.get() + cache.slots,
                  [](const Slot& slot) {
                    return slot.state.load(std::memory_order_acquire) ==
                           SlotState::kInFlightPinned;
                  })) {
    return true;
  }

  struct Candidate {
    std::int32_t expert;
    std::uint32_t frequency;
  };
  std::unordered_map<std::int32_t, std::uint32_t> frequency;
  for (const auto expert : expert_ids) {
    if (expert >= 0 && static_cast<std::uint32_t>(expert) < cache.num_experts)
      ++frequency[expert];
  }
  std::vector<Candidate> candidates;
  candidates.reserve(frequency.size());
  for (const auto& [expert, count] : frequency)
    candidates.push_back({expert, count});
  std::sort(candidates.begin(), candidates.end(),
            [&](const Candidate& a, const Candidate& b) {
              if (a.frequency != b.frequency)
                return a.frequency > b.frequency;
              const float a_prior = cache.prior_score[a.expert];
              const float b_prior = cache.prior_score[b.expert];
              const double a_score =
                  static_cast<double>(cache.route_hotness[a.expert]) +
                  impl_->cache_prior_bias * a_prior;
              const double b_score =
                  static_cast<double>(cache.route_hotness[b.expert]) +
                  impl_->cache_prior_bias * b_prior;
              return a_score != b_score ? a_score > b_score
                                        : a.expert < b.expert;
            });
  const std::size_t slot_limit = std::min(
      {max_slots, std::size_t{16}, static_cast<std::size_t>(cache.slots / 3),
       cache.slots > impl_->experts_used
           ? static_cast<std::size_t>(cache.slots - impl_->experts_used)
           : std::size_t{0}});
  const std::size_t selected = std::min(slot_limit, candidates.size());
  if (selected == 0)
    return true;
  try {
    cache.prefetch_jobs.reserve(cache.prefetch_jobs.size() + selected * 3);
  } catch (const std::exception& error) {
    SetError(error_msg, "reserving expert prefetch queue failed: " +
                            std::string(error.what()));
    return false;
  }

  const auto fail_prefetch_read = [&](std::uint32_t slot_id,
                                      std::uint64_t generation) {
    Slot& slot = cache.slot_state[slot_id];
    if (slot.generation != generation)
      return;
    slot.read_failed.store(true, std::memory_order_release);
    const std::uint32_t pending =
        slot.pending_reads.fetch_sub(1, std::memory_order_acq_rel);
    if (pending == 1) {
      if (slot.expert_id >= 0 &&
          impl_->Mapping(layer_id,
                         static_cast<std::uint32_t>(slot.expert_id)) ==
              static_cast<std::int32_t>(slot_id)) {
        impl_->Mapping(layer_id, static_cast<std::uint32_t>(slot.expert_id)) =
            -1;
      }
      slot.expert_id = -1;
      slot.prefetch_pinned = false;
      slot.prefetch_origin = false;
      slot.pin_refcount.store(0, std::memory_order_release);
      slot.state.store(SlotState::kFree, std::memory_order_release);
    }
  };

  struct GroupRead {
    std::vector<SsdReadRequest> requests;
    std::vector<std::uint32_t> slots;
    std::vector<std::uint64_t> generations;
  };
  std::map<std::filesystem::path, GroupRead> groups;
  for (std::size_t i = 0; i < selected; ++i) {
    const std::int32_t expert = candidates[i].expert;
    const std::int32_t mapped = impl_->Mapping(layer_id, expert);
    if (mapped >= 0) {
      Slot& slot = cache.slot_state[static_cast<std::uint32_t>(mapped)];
      if (slot.state.load(std::memory_order_acquire) ==
              SlotState::kCachedUnpinned &&
          slot.pin_refcount.load(std::memory_order_acquire) == 0) {
        slot.pin_refcount.store(1, std::memory_order_release);
        slot.prefetch_pinned = true;
        slot.prefetch_origin = false;
        slot.state.store(SlotState::kReadyPinned, std::memory_order_release);
        slot.last_used = ++cache.use_counter;
        cache.prefetch_slots.push_back(static_cast<std::uint32_t>(mapped));
      }
      continue;
    }

    const std::int32_t victim = impl_->ChooseVictim(cache);
    if (victim < 0)
      continue;
    Slot& slot = cache.slot_state[static_cast<std::uint32_t>(victim)];
    if (slot.expert_id >= 0)
      impl_->Mapping(layer_id, static_cast<std::uint32_t>(slot.expert_id)) = -1;
    slot.expert_id = expert;
    slot.future_uses = 0;
    slot.generation += 1;
    slot.pending_reads.store(0, std::memory_order_release);
    slot.read_failed.store(false, std::memory_order_release);
    slot.last_used = ++cache.use_counter;
    slot.pin_refcount.store(1, std::memory_order_release);
    slot.prefetch_pinned = true;
    slot.prefetch_origin = true;
    slot.state.store(SlotState::kInFlightPinned, std::memory_order_release);
    impl_->Mapping(layer_id, static_cast<std::uint32_t>(expert)) = victim;
    cache.prefetch_slots.push_back(static_cast<std::uint32_t>(victim));
    cache.route_hotness[static_cast<std::uint32_t>(expert)] +=
        candidates[i].frequency;
    cache.prior_score[static_cast<std::uint32_t>(expert)] = std::min(
        1000.0F, cache.prior_score[static_cast<std::uint32_t>(expert)] + 1.0F);

    for (std::size_t weight = 0; weight < cache.expert_bytes.size(); ++weight) {
      const auto& source = cache.view.expert_sources[weight];
      if (static_cast<std::uint64_t>(expert) >
          (std::numeric_limits<std::uint64_t>::max() - source.file_offset) /
              source.expert_bytes) {
        SetError(error_msg, "prefetch source file offset overflows");
        slot.read_failed.store(true, std::memory_order_release);
        continue;
      }
      auto* destination =
          static_cast<std::uint8_t*>(cache.host_arenas[weight]) +
          static_cast<std::size_t>(victim) * cache.expert_bytes[weight];
      auto& group = groups[source.file_path];
      group.requests.push_back(
          {.file_offset =
               source.file_offset +
               static_cast<std::uint64_t>(expert) * source.expert_bytes,
           .bytes = source.expert_bytes,
           .destination = destination,
           .destination_capacity = source.expert_bytes});
      group.slots.push_back(static_cast<std::uint32_t>(victim));
      group.generations.push_back(slot.generation);
      slot.pending_reads.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  // An all-resident prefetch is already complete. No future or completion
  // signal is required; Resolve will consume these pinned READY slots.
  for (auto& [path, group] : groups) {
    auto* reader = impl_->Reader(path, error_msg);
    if (reader == nullptr) {
      for (std::size_t i = 0; i < group.slots.size(); ++i)
        fail_prefetch_read(group.slots[i], group.generations[i]);
      return false;
    }
    auto job = std::make_shared<PrefetchJob>();
    job->reader = reader;
    job->prefill = prefill;
    job->slots = std::move(group.slots);
    job->generations = std::move(group.generations);
    try {
      job->result =
          reader->SubmitBatchAsync(std::move(group.requests), &job->batch_id)
              .share();
      cache.prefetch_jobs.push_back(job);
    } catch (const std::exception& error) {
      SetError(error_msg, "submitting expert prefetch failed: " +
                              std::string(error.what()));
      if (job->result.valid()) {
        job->result.wait();
        (void)job->result.get();
      }
      for (std::size_t i = 0; i < job->slots.size(); ++i)
        fail_prefetch_read(job->slots[i], job->generations[i]);
      return false;
    }
  }
  for (const std::uint32_t slot_id : cache.prefetch_slots) {
    Slot& slot = cache.slot_state[slot_id];
    if (slot.state.load(std::memory_order_acquire) !=
            SlotState::kInFlightPinned ||
        slot.pending_reads.load(std::memory_order_acquire) != 0)
      continue;
    if (slot.read_failed.load(std::memory_order_acquire)) {
      if (slot.expert_id >= 0 &&
          impl_->Mapping(layer_id,
                         static_cast<std::uint32_t>(slot.expert_id)) ==
              static_cast<std::int32_t>(slot_id)) {
        impl_->Mapping(layer_id, static_cast<std::uint32_t>(slot.expert_id)) =
            -1;
      }
      slot.expert_id = -1;
      slot.prefetch_pinned = false;
      slot.prefetch_origin = false;
      slot.pin_refcount.store(0, std::memory_order_release);
      slot.state.store(SlotState::kFree, std::memory_order_release);
    } else {
      slot.state.store(SlotState::kReadyPinned, std::memory_order_release);
    }
  }
  if (impl_->stats != nullptr) {
    std::uint64_t submitted = 0;
    for (const std::uint32_t slot_id : cache.prefetch_slots) {
      const Slot& slot = cache.slot_state[slot_id];
      submitted += slot.prefetch_origin &&
                   slot.pending_reads.load(std::memory_order_acquire) != 0 &&
                   !slot.read_failed.load(std::memory_order_acquire);
    }
    impl_->stats->RecordPrefetchRequests(submitted, prefill);
  }
  return true;
}

bool ExpertCacheManager::PrefetchTokenPriors(
    std::uint32_t layer_id, std::span<const std::int32_t> tokens,
    std::size_t max_slots, std::string* error_msg, bool prefill) {
  std::vector<std::int32_t> predictions;
  {
    if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
      return true;
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    CacheLayer& cache = *impl_->caches[layer_id];
    for (const auto token : tokens) {
      const auto found = cache.token_priors.find(token);
      if (found == cache.token_priors.end())
        continue;
      found->second.last_used = ++cache.prior_clock;
      predictions.insert(predictions.end(), found->second.experts.begin(),
                         found->second.experts.end());
    }
  }
  return Prefetch(layer_id, predictions, max_slots, error_msg, prefill);
}

void ExpertCacheManager::RecordTokenRouting(
    std::uint32_t layer_id, std::span<const std::int32_t> tokens,
    std::span<const std::int32_t> expert_ids) {
  if (layer_id != 0 || layer_id >= impl_->caches.size() ||
      impl_->caches[layer_id] == nullptr ||
      expert_ids.size() < tokens.size() * impl_->experts_used)
    return;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    auto& prior = cache.token_priors[tokens[row]];
    prior.experts.clear();
    prior.last_used = ++cache.prior_clock;
    for (std::uint32_t i = 0; i < impl_->experts_used; ++i) {
      const std::int32_t expert = expert_ids[row * impl_->experts_used + i];
      if (expert >= 0 &&
          expert < static_cast<std::int32_t>(cache.num_experts) &&
          std::find(prior.experts.begin(), prior.experts.end(), expert) ==
              prior.experts.end()) {
        prior.experts.push_back(expert);
        float& score = cache.prior_score[static_cast<std::uint32_t>(expert)];
        score = std::min(1000.0F, score + 1.0F);
      }
    }
    if (cache.token_priors.size() > 4096) {
      const auto oldest =
          std::min_element(cache.token_priors.begin(), cache.token_priors.end(),
                           [](const auto& a, const auto& b) {
                             return a.second.last_used < b.second.last_used;
                           });
      if (oldest != cache.token_priors.end() && oldest->first != tokens[row])
        cache.token_priors.erase(oldest);
    }
  }
}

bool ExpertCacheManager::PinDesk(std::uint32_t layer_id,
                                 std::span<const std::int32_t> expert_ids,
                                 std::span<std::int32_t> slot_ids,
                                 std::string* error_msg, bool prefill) {
  if (expert_ids.size() != slot_ids.size()) {
    SetError(error_msg, "desk expert and slot arrays have different sizes");
    return false;
  }
  if (expert_ids.empty())
    return true;
  if (!Resolve(layer_id, expert_ids, slot_ids, error_msg, {}, nullptr, true,
               prefill, false, 0, false)) {
    return false;
  }
  bool valid_slots = true;
  bool layer_available = true;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    if (layer_id >= impl_->caches.size() ||
        impl_->caches[layer_id] == nullptr) {
      SetError(error_msg, "desk layer is no longer available");
      layer_available = false;
    } else {
      CacheLayer& cache = *impl_->caches[layer_id];
      for (const std::int32_t slot_id : slot_ids) {
        if (slot_id < 0 || static_cast<std::uint32_t>(slot_id) >= cache.slots) {
          SetError(error_msg, "desk resolve returned an invalid slot");
          valid_slots = false;
          break;
        }
      }
      if (valid_slots) {
        for (const std::int32_t slot_id : slot_ids)
          cache.slot_state[static_cast<std::uint32_t>(slot_id)].desk_reserved =
              true;
      }
    }
  }
  Release(layer_id);
  return valid_slots && layer_available;
}

void ExpertCacheManager::ReleaseDesk(
    std::uint32_t layer_id, std::span<const std::int32_t> expert_ids) noexcept {
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
    return;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  for (const std::int32_t expert : expert_ids) {
    if (expert < 0 || static_cast<std::uint32_t>(expert) >= impl_->num_experts)
      continue;
    const std::int32_t slot_id =
        impl_->Mapping(layer_id, static_cast<std::uint32_t>(expert));
    if (slot_id >= 0 && static_cast<std::uint32_t>(slot_id) < cache.slots)
      cache.slot_state[static_cast<std::uint32_t>(slot_id)].desk_reserved =
          false;
  }
}

bool ExpertCacheManager::Resolve(
    std::uint32_t layer_id, std::span<const std::int32_t> expert_ids,
    std::span<std::int32_t> slot_ids, std::string* error_msg,
    std::span<const std::int32_t> future_expert_ids, std::uint64_t* bytes_read,
    bool record_stats, bool prefill, bool record_demand_stats,
    std::uint32_t routed_tokens, bool count_route_hotness) {
  if (error_msg != nullptr)
    error_msg->clear();
  if (bytes_read != nullptr)
    *bytes_read = 0;
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr ||
      expert_ids.size() != slot_ids.size()) {
    SetError(error_msg, "expert cache resolve has invalid dimensions");
    return false;
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  if (impl_->victim_policy == VictimPolicy::kDecayedHotness &&
      count_route_hotness) {
    if (cache.has_route_phase && cache.last_route_was_prefill && !prefill) {
      std::fill(cache.route_hotness.begin(), cache.route_hotness.end(), 0);
      cache.route_tokens_since_decay = 0;
    }
    cache.has_route_phase = true;
    cache.last_route_was_prefill = prefill;
  }
  impl_->ReapPrefetchJobs(cache);
  if (!cache.pinned_slots.empty()) {
    SetError(error_msg, "expert cache layer still has pinned slots");
    return false;
  }

  std::vector<std::int32_t> unique;
  unique.reserve(expert_ids.size());
  std::vector<std::uint8_t> seen(cache.num_experts, 0);
  std::vector<std::uint32_t> route_hits(cache.num_experts, 0);
  std::vector<std::uint32_t> future_hits(cache.num_experts, 0);
  for (const auto id : future_expert_ids) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= cache.num_experts) {
      SetError(error_msg, "future expert hint is out of range");
      return false;
    }
    auto& hits = future_hits[static_cast<std::uint32_t>(id)];
    if (hits != std::numeric_limits<std::uint32_t>::max())
      ++hits;
  }
  for (const auto id : expert_ids) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= cache.num_experts) {
      SetError(error_msg, "router selected an out-of-range expert ID");
      return false;
    }
    auto& hits = route_hits[static_cast<std::uint32_t>(id)];
    if (hits != std::numeric_limits<std::uint32_t>::max())
      ++hits;
    if (!seen[static_cast<std::uint32_t>(id)]) {
      seen[static_cast<std::uint32_t>(id)] = 1;
      unique.push_back(id);
    }
  }
  if (unique.size() > cache.slots) {
    SetError(error_msg, "active expert set exceeds the layer slot capacity");
    return false;
  }
  for (std::uint32_t slot_id = 0; slot_id < cache.slots; ++slot_id) {
    Slot& slot = cache.slot_state[slot_id];
    slot.future_uses =
        slot.expert_id >= 0
            ? future_hits[static_cast<std::uint32_t>(slot.expert_id)]
            : 0;
  }

  std::vector<std::pair<std::int32_t, std::uint32_t>> misses;
  std::vector<std::int32_t> missing_experts;
  std::vector<std::uint32_t> newly_reserved;
  misses.reserve(unique.size());
  missing_experts.reserve(unique.size());
  newly_reserved.reserve(unique.size());
  cache.pinned_slots.reserve(unique.size());
  std::uint64_t resident_hits = 0;
  std::uint64_t prefetched_in_time = 0;
  std::uint64_t late_prefetch_requests = 0;
  std::chrono::nanoseconds prefetch_wait{0};
  for (const auto expert : unique) {
    auto mapped = impl_->expert_id_to_slot[static_cast<std::size_t>(layer_id) *
                                               impl_->num_experts +
                                           expert];
    if (mapped >= 0) {
      Slot& slot = cache.slot_state[static_cast<std::uint32_t>(mapped)];
      auto state = slot.state.load(std::memory_order_acquire);
      bool was_prefetched_ready = state == SlotState::kReadyPinned &&
                                  slot.prefetch_pinned && slot.prefetch_origin;
      bool waited_for_prefetch = false;
      if (state == SlotState::kInFlightPinned) {
        ++late_prefetch_requests;
        const auto wait_start = std::chrono::steady_clock::now();
        const auto generation = slot.generation;
        bool found_job = false;
        for (const auto& job : cache.prefetch_jobs) {
          const bool owns_slot = [&] {
            for (std::size_t i = 0; i < job->slots.size(); ++i) {
              if (job->slots[i] == static_cast<std::uint32_t>(mapped) &&
                  job->generations[i] == generation)
                return true;
            }
            return false;
          }();
          if (owns_slot) {
            found_job = true;
            if (job->reader != nullptr &&
                job->reader->PromotePrefetch(job->batch_id)) {
              job->speculative = false;
            }
            (void)impl_->FinishPrefetchJob(cache, job, true);
          }
        }
        const auto waited =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_start);
        state = slot.state.load(std::memory_order_acquire);
        if (state == SlotState::kInFlightPinned) {
          SetError(error_msg, found_job
                                  ? "prefetched expert read is incomplete"
                                  : "in-flight expert has no completion batch");
          impl_->Rollback(cache, newly_reserved);
          return false;
        }
        waited_for_prefetch = waited.count() != 0;
        if (waited_for_prefetch) {
          prefetch_wait += waited;
        }
      }
      if (state == SlotState::kFree) {
        impl_->Mapping(layer_id, static_cast<std::uint32_t>(expert)) = -1;
        mapped = -1;
      } else if (state == SlotState::kReadyPinned && slot.prefetch_pinned) {
        // Convert the prefetch pin into the demand pin; this never exposes the
        // slot to victim selection between prediction and consumption.
        if (slot.prefetch_origin) {
          if (was_prefetched_ready)
            ++prefetched_in_time;
        } else {
          ++resident_hits;
        }
        slot.prefetch_pinned = false;
        slot.prefetch_origin = false;
        slot.last_used = ++cache.use_counter;
        cache.pinned_slots.push_back(static_cast<std::uint32_t>(mapped));
        if (count_route_hotness)
          cache.route_hotness[static_cast<std::uint32_t>(expert)] +=
              route_hits[static_cast<std::uint32_t>(expert)];
        continue;
      } else if (state == SlotState::kCachedUnpinned &&
                 slot.pin_refcount.load(std::memory_order_acquire) == 0) {
        slot.pin_refcount.store(1, std::memory_order_release);
        slot.state.store(SlotState::kReadyPinned, std::memory_order_release);
        slot.last_used = ++cache.use_counter;
        cache.pinned_slots.push_back(static_cast<std::uint32_t>(mapped));
        if (count_route_hotness)
          cache.route_hotness[static_cast<std::uint32_t>(expert)] +=
              route_hits[static_cast<std::uint32_t>(expert)];
        if (!waited_for_prefetch)
          ++resident_hits;
        continue;
      } else if (mapped >= 0) {
        SetError(error_msg, "expert cache slot is already pinned or in flight");
        impl_->Rollback(cache, newly_reserved);
        return false;
      }
    }
    if (mapped < 0)
      missing_experts.push_back(expert);
  }

  // Pin every resident/prefetched member of the window before choosing any
  // victims. Otherwise an early miss can evict a hot resident expert that is
  // routed by a later row in this same batched window.
  for (const std::int32_t expert : missing_experts) {
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
    slot.future_uses = future_hits[static_cast<std::uint32_t>(expert)];
    slot.last_used = ++cache.use_counter;
    slot.pin_refcount.store(1, std::memory_order_release);
    slot.pending_reads.store(0, std::memory_order_release);
    slot.read_failed.store(false, std::memory_order_release);
    slot.prefetch_pinned = false;
    slot.prefetch_origin = false;
    slot.state.store(SlotState::kInFlightPinned, std::memory_order_release);
    impl_->expert_id_to_slot[static_cast<std::size_t>(layer_id) *
                                 impl_->num_experts +
                             static_cast<std::uint32_t>(expert)] = victim;
    cache.pinned_slots.push_back(static_cast<std::uint32_t>(victim));
    misses.emplace_back(expert, static_cast<std::uint32_t>(victim));
    newly_reserved.push_back(static_cast<std::uint32_t>(victim));
    if (count_route_hotness)
      cache.route_hotness[static_cast<std::uint32_t>(expert)] +=
          route_hits[static_cast<std::uint32_t>(expert)];
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

  const auto stall_start = std::chrono::steady_clock::now();
  struct DemandBatch {
    std::future<SsdReadBatchResult> result;
  };
  std::vector<DemandBatch> demand_batches;
  demand_batches.reserve(grouped.size());
  bool demand_submission_ok = true;
  for (auto& [path, reads] : grouped) {
    auto reader = impl_->Reader(path, error_msg);
    if (reader == nullptr) {
      demand_submission_ok = false;
      break;
    }
    std::vector<SsdReadRequest> requests;
    requests.reserve(reads.size());
    for (auto& read : reads)
      requests.push_back(read.request);
    demand_batches.push_back({reader->SubmitBatchDemand(std::move(requests))});
  }
  bool demand_reads_ok = demand_submission_ok;
  std::uint64_t total_bytes_read = 0;
  for (auto& batch : demand_batches) {
    SsdReadBatchResult result = batch.result.get();
    total_bytes_read =
        result.bytes_read >
                std::numeric_limits<std::uint64_t>::max() - total_bytes_read
            ? std::numeric_limits<std::uint64_t>::max()
            : total_bytes_read + result.bytes_read;
    if (record_stats && impl_->stats != nullptr)
      impl_->stats->RecordStorageRead(result.bytes_read, result.elapsed,
                                      prefill, false);
    if (!result.success) {
      if (demand_reads_ok)
        SetError(error_msg, result.error.empty() ? "expert demand read failed"
                                                 : result.error);
      demand_reads_ok = false;
    }
  }
  if (!demand_reads_ok) {
    if (bytes_read != nullptr)
      *bytes_read = total_bytes_read;
    impl_->Rollback(cache, newly_reserved);
    return false;
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
  if (bytes_read != nullptr)
    *bytes_read = total_bytes_read;
  if (record_stats && record_demand_stats && impl_->stats != nullptr) {
    impl_->stats->RecordExpertDemand(
        unique.size(), resident_hits, prefetched_in_time, misses.size(),
        late_prefetch_requests,
        prefetch_wait +
            (misses.empty()
                 ? std::chrono::nanoseconds{0}
                 : std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - stall_start)),
        prefill);
  }
  if (impl_->victim_policy == VictimPolicy::kDecayedHotness &&
      count_route_hotness) {
    const auto rows = routed_tokens == 0
                          ? expert_ids.size() / impl_->experts_used
                          : routed_tokens;
    cache.route_tokens_since_decay += rows;
    while (cache.route_tokens_since_decay >= impl_->hotness_decay_tokens) {
      for (auto& hotness : cache.route_hotness)
        hotness >>= 1;
      cache.route_tokens_since_decay -= impl_->hotness_decay_tokens;
    }
  }
  return true;
}

void ExpertCacheManager::Release(std::uint32_t layer_id) noexcept {
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
    return;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  impl_->ReleaseLocked(cache);
}

void ExpertCacheManager::FinishPrefetch(std::uint32_t layer_id) noexcept {
  if (layer_id >= impl_->caches.size() || impl_->caches[layer_id] == nullptr)
    return;
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  CacheLayer& cache = *impl_->caches[layer_id];
  for (const auto& job : cache.prefetch_jobs) {
    if (!job->finalized && job->reader != nullptr)
      (void)job->reader->CancelPrefetch(job->batch_id);
  }
  for (const std::uint32_t slot_id : cache.prefetch_slots) {
    Slot& slot = cache.slot_state[slot_id];
    if (!slot.prefetch_pinned)
      continue;
    slot.prefetch_pinned = false;
    slot.prefetch_origin = false;
    const std::uint32_t previous =
        slot.pin_refcount.fetch_sub(1, std::memory_order_acq_rel);
    if (previous == 1 &&
        slot.state.load(std::memory_order_acquire) == SlotState::kReadyPinned) {
      slot.state.store(SlotState::kCachedUnpinned, std::memory_order_release);
    }
  }
  cache.prefetch_slots.clear();
  impl_->ReapPrefetchJobs(cache);
}

}  // namespace gufo::storage
