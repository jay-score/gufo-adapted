#ifndef GUFO_STORAGE_SSD_BATCH_READER_HPP_
#define GUFO_STORAGE_SSD_BATCH_READER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::storage {

enum class SsdReadPriority : std::uint8_t {
  kDemand,
  kRoom,
  kSpeculative,
};

/// One unaligned file range read into a caller-owned, 4096-byte-aligned slot.
/// On success, Payload() addresses the first requested byte inside that slot.
struct SsdReadRequest {
  std::uint64_t file_offset{0};
  std::size_t bytes{0};
  void* destination{nullptr};
  std::size_t destination_capacity{0};

  std::size_t payload_offset{0};
  std::size_t file_subpage_offset{0};
  std::size_t aligned_bytes{0};
  std::size_t transferred_bytes{0};
  std::uint32_t system_error{0};
  bool completed{false};
  bool used_scratch{false};

  [[nodiscard]] const std::uint8_t* Payload() const noexcept {
    return static_cast<const std::uint8_t*>(destination) + payload_offset;
  }
};

struct SsdReadBatchResult {
  std::vector<SsdReadRequest> requests;
  std::string error;
  std::chrono::nanoseconds elapsed{0};
  std::uint64_t bytes_read{0};
  bool success{false};
};

/// Model-shared pool for aligned bounce buffers and in-flight I/O leases.
/// Each lease reserves one of `buffer_count()` entries until the matching
/// backend completion, even when an aligned direct read does not use its
/// scratch bytes.
class SsdScratchPool {
public:
  [[nodiscard]] static std::shared_ptr<SsdScratchPool> Create(
      std::size_t buffer_count, std::size_t buffer_bytes);
  ~SsdScratchPool();

  SsdScratchPool(const SsdScratchPool&) = delete;
  SsdScratchPool& operator=(const SsdScratchPool&) = delete;

  [[nodiscard]] std::size_t buffer_count() const noexcept;
  [[nodiscard]] std::size_t buffer_bytes() const noexcept;
  [[nodiscard]] std::size_t allocated_bytes() const noexcept;

  /// Blocking lease acquisition preserves demand > Room > speculative order.
  [[nodiscard]] std::size_t Acquire(SsdReadPriority priority);
  [[nodiscard]] void* Buffer(std::size_t lease) const noexcept;
  void Release(std::size_t lease) noexcept;

  [[nodiscard]] static std::size_t EstimateBufferBytes(
      std::size_t max_aligned_read_bytes);

private:
  struct Impl;
  explicit SsdScratchPool(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

/// Batched direct reads into caller-owned, GPU-visible host allocations.
/// Requests may start at arbitrary GGUF offsets; the reader pads each range to
/// 4096-byte boundaries and reports the sub-page offset through Payload().
class SsdBatchReader {
public:
  struct Options {
    std::size_t io_threads{8};
    std::size_t max_scratch_bytes{4 * 1024 * 1024};
    std::size_t max_inflight_reads{32};
    std::shared_ptr<SsdScratchPool> scratch_pool;
  };

  using BatchId = std::uint64_t;

  explicit SsdBatchReader(std::filesystem::path path);
  SsdBatchReader(std::filesystem::path path, Options options);
  ~SsdBatchReader();

  SsdBatchReader(const SsdBatchReader&) = delete;
  SsdBatchReader& operator=(const SsdBatchReader&) = delete;
  SsdBatchReader(SsdBatchReader&&) noexcept;
  SsdBatchReader& operator=(SsdBatchReader&&) noexcept;

  /// Reads every request concurrently. The destination buffers must remain
  /// alive and unmodified until this call returns.
  [[nodiscard]] bool ReadBatch(std::span<SsdReadRequest> requests,
                               std::string* error_msg = nullptr,
                               std::uint64_t* bytes_read = nullptr);

  /// Queues a background prefetch; Room work outranks speculative work, and
  /// both yield to demand. The reader must outlive the returned future and the
  /// destinations must remain valid until it is ready. `speculative=true`
  /// selects the lowest-priority lane; otherwise the batch is Room work.
  [[nodiscard]] std::future<SsdReadBatchResult> SubmitBatchAsync(
      std::vector<SsdReadRequest> requests, BatchId* batch_id = nullptr,
      bool speculative = false);

  /// Queues a demand batch on the high-priority lane and returns immediately.
  /// This allows independent shard files to read concurrently while retaining
  /// priority over queued Room and speculative prefetches on each reader.
  [[nodiscard]] std::future<SsdReadBatchResult> SubmitBatchDemand(
      std::vector<SsdReadRequest> requests);

  /// Moves an unsubmitted Room or speculative batch to the demand lane.
  /// Returns false if the batch is already in flight or has completed.
  [[nodiscard]] bool PromotePrefetch(BatchId batch_id);

  /// Cancels an unsubmitted speculative batch. In-flight I/O is allowed to
  /// finish because Windows direct-I/O buffers must remain valid until IOCP
  /// completion.
  [[nodiscard]] bool CancelPrefetch(BatchId batch_id);

  [[nodiscard]] static constexpr std::size_t Alignment() noexcept {
    return 4096;
  }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_SSD_BATCH_READER_HPP_
