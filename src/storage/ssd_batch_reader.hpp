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

/// Batched direct reads into caller-owned, GPU-visible host allocations.
/// Requests may start at arbitrary GGUF offsets; the reader pads each range to
/// 4096-byte boundaries and reports the sub-page offset through Payload().
class SsdBatchReader {
public:
  struct Options {
    std::size_t io_threads{8};
    std::size_t max_scratch_bytes{4 * 1024 * 1024};
    std::size_t max_inflight_reads{32};
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

  /// Starts a batch on a background thread. The reader must outlive the
  /// returned future; destination storage must remain valid until it is ready.
  [[nodiscard]] std::future<SsdReadBatchResult> SubmitBatchAsync(
      std::vector<SsdReadRequest> requests, BatchId* batch_id = nullptr);

  /// Queues a demand batch on the high-priority lane and returns immediately.
  /// This allows independent shard files to read concurrently while retaining
  /// priority over queued speculative prefetches on each reader.
  [[nodiscard]] std::future<SsdReadBatchResult> SubmitBatchDemand(
      std::vector<SsdReadRequest> requests);

  /// Moves an unsubmitted speculative batch to the demand lane. Returns false
  /// if the batch is already in flight or has completed.
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
