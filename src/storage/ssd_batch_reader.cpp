#include "src/storage/ssd_batch_reader.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#endif

namespace gufo::storage {
namespace {

constexpr std::size_t kAlignment = SsdBatchReader::Alignment();

std::size_t AlignUp(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() - (kAlignment - 1))
    throw std::overflow_error("SSD scratch-buffer alignment overflows");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}

void* AllocateAligned(std::size_t bytes) {
#ifdef _WIN32
  return _aligned_malloc(bytes, kAlignment);
#else
  void* result = nullptr;
  return posix_memalign(&result, kAlignment, bytes) == 0 ? result : nullptr;
#endif
}

void FreeAligned(void* pointer) {
#ifdef _WIN32
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

bool DirectEligible(const SsdReadRequest& request) {
  return request.file_offset % kAlignment == 0 &&
         request.bytes % kAlignment == 0 &&
         reinterpret_cast<std::uintptr_t>(request.destination) % kAlignment ==
             0;
}

bool PrepareRequest(SsdReadRequest& request, std::string* error_msg) {
  request.payload_offset = 0;
  request.file_subpage_offset = 0;
  request.aligned_bytes = 0;
  request.transferred_bytes = 0;
  request.system_error = 0;
  request.completed = false;
  request.used_scratch = false;
  if (request.bytes == 0 || request.destination == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "SSD read needs a non-empty range and destination";
    return false;
  }
  if (request.bytes > request.destination_capacity) {
    if (error_msg != nullptr)
      *error_msg = "SSD read destination is smaller than the payload";
    return false;
  }
  if (request.file_offset >
      std::numeric_limits<std::uint64_t>::max() - request.bytes) {
    if (error_msg != nullptr)
      *error_msg = "SSD read file range overflows";
    return false;
  }
  const auto begin = request.file_offset & ~(std::uint64_t{kAlignment} - 1);
  const auto delta = request.file_offset - begin;
  const auto end = request.file_offset + request.bytes;
  if (end > std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1)) {
    if (error_msg != nullptr)
      *error_msg = "SSD read aligned range overflows";
    return false;
  }
  const auto aligned_end =
      (end + kAlignment - 1) & ~(std::uint64_t{kAlignment} - 1);
  const auto aligned_bytes = aligned_end - begin;
  if (aligned_bytes > std::numeric_limits<std::size_t>::max()) {
    if (error_msg != nullptr)
      *error_msg = "SSD read aligned range exceeds the host size range";
    return false;
  }
  request.file_subpage_offset = static_cast<std::size_t>(delta);
  request.aligned_bytes = static_cast<std::size_t>(aligned_bytes);
  request.used_scratch = !DirectEligible(request);
  if (!request.used_scratch) {
    if (request.aligned_bytes > request.destination_capacity) {
      if (error_msg != nullptr)
        *error_msg = "aligned direct-read window exceeds destination capacity";
      return false;
    }
  }
  return true;
}

std::vector<std::size_t> SortedRequestIndices(
    std::span<const SsdReadRequest> requests) {
  std::vector<std::size_t> order(requests.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const auto& left = requests[a];
    const auto& right = requests[b];
    const std::uint64_t left_begin =
        left.file_offset - left.file_subpage_offset;
    const std::uint64_t right_begin =
        right.file_offset - right.file_subpage_offset;
    return left_begin != right_begin ? left_begin < right_begin
                                     : left.aligned_bytes < right.aligned_bytes;
  });
  return order;
}

}  // namespace

struct SsdScratchPool::Impl {
  ~Impl() {
    for (void* buffer : buffers)
      FreeAligned(buffer);
  }

  std::vector<void*> buffers;
  std::deque<std::size_t> free;
  std::size_t buffer_bytes{0};
  std::size_t demand_waiters{0};
  std::size_t prefetch_waiters{0};
  mutable std::mutex mutex;
  std::condition_variable available;
};

SsdScratchPool::SsdScratchPool(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::shared_ptr<SsdScratchPool> SsdScratchPool::Create(
    std::size_t buffer_count, std::size_t buffer_bytes) {
  if (buffer_count == 0 || buffer_bytes == 0 ||
      buffer_count > std::numeric_limits<std::size_t>::max() / buffer_bytes) {
    throw std::invalid_argument("SSD shared scratch-pool geometry is invalid");
  }
  auto impl = std::make_unique<Impl>();
  impl->buffer_bytes = AlignUp(buffer_bytes);
  if (buffer_count >
      std::numeric_limits<std::size_t>::max() / impl->buffer_bytes) {
    throw std::invalid_argument("SSD shared scratch-pool size overflows");
  }
  impl->buffers.reserve(buffer_count);
  for (std::size_t i = 0; i < buffer_count; ++i) {
    void* buffer = AllocateAligned(impl->buffer_bytes);
    if (buffer == nullptr)
      throw std::bad_alloc();
    impl->buffers.push_back(buffer);
    impl->free.push_back(i);
  }
  return std::shared_ptr<SsdScratchPool>(new SsdScratchPool(std::move(impl)));
}

SsdScratchPool::~SsdScratchPool() = default;

std::size_t SsdScratchPool::buffer_count() const noexcept {
  return impl_ != nullptr ? impl_->buffers.size() : 0;
}

std::size_t SsdScratchPool::buffer_bytes() const noexcept {
  return impl_ != nullptr ? impl_->buffer_bytes : 0;
}

std::size_t SsdScratchPool::allocated_bytes() const noexcept {
  return buffer_count() * buffer_bytes();
}

std::size_t SsdScratchPool::Acquire(bool demand) {
  if (impl_ == nullptr)
    throw std::logic_error("SSD shared scratch pool is unavailable");
  std::unique_lock<std::mutex> lock(impl_->mutex);
  auto& waiters = demand ? impl_->demand_waiters : impl_->prefetch_waiters;
  ++waiters;
  impl_->available.wait(lock, [&] {
    return !impl_->free.empty() && (demand || impl_->demand_waiters == 0);
  });
  --waiters;
  const std::size_t lease = impl_->free.front();
  impl_->free.pop_front();
  return lease;
}

void* SsdScratchPool::Buffer(std::size_t lease) const noexcept {
  return impl_ != nullptr && lease < impl_->buffers.size()
             ? impl_->buffers[lease]
             : nullptr;
}

void SsdScratchPool::Release(std::size_t lease) noexcept {
  if (impl_ == nullptr || lease >= impl_->buffers.size())
    return;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->free.push_back(lease);
  }
  impl_->available.notify_all();
}

std::size_t SsdScratchPool::EstimateBufferBytes(
    std::size_t max_aligned_read_bytes) {
  if (max_aligned_read_bytes == 0 ||
      max_aligned_read_bytes >
          std::numeric_limits<std::size_t>::max() - kAlignment) {
    throw std::invalid_argument("SSD scratch-buffer size is invalid");
  }
  return AlignUp(max_aligned_read_bytes + kAlignment);
}

struct SsdBatchReader::Impl {
  struct BatchJob {
    SsdBatchReader::BatchId id{0};
    bool prefetch{false};
    std::vector<SsdReadRequest> requests;
    std::promise<SsdReadBatchResult> completion;
  };

  explicit Impl(const std::filesystem::path& path, Options options) {
    if (options.io_threads == 0 || options.io_threads > 64)
      throw std::invalid_argument("SSD reader thread count must be in 1..64");
    if (options.max_inflight_reads == 0 || options.max_inflight_reads > 64)
      throw std::invalid_argument(
          "SSD reader in-flight request limit must be in 1..64");
    if (options.max_scratch_bytes == 0 ||
        options.max_scratch_bytes >
            std::numeric_limits<std::size_t>::max() - kAlignment) {
      throw std::invalid_argument("SSD reader scratch size is invalid");
    }
    max_inflight_reads = options.max_inflight_reads;
    scratch_pool = std::move(options.scratch_pool);
    scratch_capacity = scratch_pool != nullptr
                           ? scratch_pool->buffer_bytes()
                           : AlignUp(options.max_scratch_bytes + kAlignment);
    if (scratch_capacity == 0 ||
        (scratch_pool != nullptr && scratch_pool->buffer_count() == 0))
      throw std::invalid_argument("SSD scratch pool has no buffers");
#ifdef _WIN32
    file =
        CreateFileW(path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                        FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                    nullptr);
    if (file == INVALID_HANDLE_VALUE)
      throw std::runtime_error("CreateFileW failed for SSD reader: " +
                               std::to_string(GetLastError()));
    port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0,
                                  static_cast<DWORD>(options.io_threads));
    if (port == nullptr) {
      const DWORD error = GetLastError();
      CloseHandle(file);
      file = INVALID_HANDLE_VALUE;
      throw std::runtime_error("CreateIoCompletionPort failed: " +
                               std::to_string(error));
    }
    if (CreateIoCompletionPort(file, port, 0, 0) == nullptr) {
      const DWORD error = GetLastError();
      CloseHandle(port);
      CloseHandle(file);
      port = nullptr;
      file = INVALID_HANDLE_VALUE;
      throw std::runtime_error("associating SSD file with IOCP failed: " +
                               std::to_string(error));
    }
    try {
      if (scratch_pool == nullptr)
        AllocateScratch(options.max_inflight_reads);
      workers.reserve(options.io_threads);
      for (std::size_t i = 0; i < options.io_threads; ++i) {
        workers.emplace_back([this] { CompletionWorker(); });
      }
      batch_worker = std::thread([this] { BatchWorker(); });
    } catch (...) {
      {
        const std::lock_guard<std::mutex> lock(batch_mutex);
        stopping = true;
      }
      batch_ready.notify_all();
      if (batch_worker.joinable())
        batch_worker.join();
      for (std::size_t i = 0; i < workers.size(); ++i)
        (void)PostQueuedCompletionStatus(port, 0, 0, nullptr);
      for (auto& worker : workers)
        worker.join();
      FreeScratch();
      CloseHandle(port);
      CloseHandle(file);
      port = nullptr;
      file = INVALID_HANDLE_VALUE;
      throw;
    }
#else
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0)
      throw std::runtime_error("open(O_DIRECT) failed for SSD reader: " +
                               std::to_string(errno));
    try {
      if (scratch_pool == nullptr)
        AllocateScratch(options.io_threads);
      worker_queues.resize(options.io_threads);
      workers.reserve(options.io_threads);
      for (std::size_t i = 0; i < options.io_threads; ++i) {
        workers.emplace_back([this, i] { ReadWorker(i); });
      }
      batch_worker = std::thread([this] { BatchWorker(); });
    } catch (...) {
      {
        const std::lock_guard<std::mutex> lock(batch_mutex);
        stopping = true;
      }
      batch_ready.notify_all();
      if (batch_worker.joinable())
        batch_worker.join();
      {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        worker_stopping = true;
      }
      work_ready.notify_all();
      for (auto& worker : workers)
        worker.join();
      FreeScratch();
      ::close(fd);
      fd = -1;
      throw;
    }
#endif
  }

  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(batch_mutex);
      stopping = true;
    }
    batch_ready.notify_all();
    if (batch_worker.joinable())
      batch_worker.join();
#ifdef _WIN32
    if (port != nullptr) {
      for (std::size_t i = 0; i < workers.size(); ++i)
        (void)PostQueuedCompletionStatus(port, 0, 0, nullptr);
    }
    for (auto& worker : workers) {
      if (worker.joinable())
        worker.join();
    }
    FreeScratch();
    if (port != nullptr)
      CloseHandle(port);
    if (file != INVALID_HANDLE_VALUE)
      CloseHandle(file);
#else
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      worker_stopping = true;
    }
    work_ready.notify_all();
    for (auto& worker : workers) {
      if (worker.joinable())
        worker.join();
    }
    FreeScratch();
    if (fd >= 0)
      ::close(fd);
#endif
  }

  void AllocateScratch(std::size_t count) {
    scratch_buffers.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      void* scratch = AllocateAligned(scratch_capacity);
      if (scratch == nullptr) {
        FreeScratch();
        throw std::bad_alloc();
      }
      scratch_buffers.push_back(scratch);
    }
  }

  void FreeScratch() {
    for (void* scratch : scratch_buffers)
      FreeAligned(scratch);
    scratch_buffers.clear();
  }

  bool ReadBatch(std::span<SsdReadRequest> requests, std::string* error_msg) {
    return ReadBatch(requests, error_msg, nullptr);
  }

  bool ReadBatch(std::span<SsdReadRequest> requests, std::string* error_msg,
                 std::uint64_t* bytes_read) {
    if (error_msg != nullptr)
      error_msg->clear();
    if (bytes_read != nullptr)
      *bytes_read = 0;
    std::vector<SsdReadRequest> owned(requests.begin(), requests.end());
    auto future = EnqueueBatch(std::move(owned), false, nullptr);
    SsdReadBatchResult result = future.get();
    if (result.requests.size() == requests.size())
      std::copy(result.requests.begin(), result.requests.end(),
                requests.begin());
    if (error_msg != nullptr)
      *error_msg = result.error;
    if (bytes_read != nullptr)
      *bytes_read = result.bytes_read;
    return result.success;
  }

  std::future<SsdReadBatchResult> EnqueueBatch(
      std::vector<SsdReadRequest> requests, bool prefetch,
      SsdBatchReader::BatchId* batch_id) {
    auto job = std::make_shared<BatchJob>();
    job->prefetch = prefetch;
    job->requests = std::move(requests);
    auto future = job->completion.get_future();
    if (job->requests.empty()) {
      SsdReadBatchResult result;
      result.requests = std::move(job->requests);
      result.elapsed = std::chrono::nanoseconds{0};
      result.success = true;
      job->completion.set_value(std::move(result));
      if (batch_id != nullptr)
        *batch_id = 0;
      return future;
    }
    {
      const std::lock_guard<std::mutex> lock(batch_mutex);
      if (stopping) {
        SsdReadBatchResult result;
        result.requests = std::move(job->requests);
        result.error = "SSD reader is stopping";
        job->completion.set_value(std::move(result));
        if (batch_id != nullptr)
          *batch_id = 0;
        return future;
      }
      job->id = ++next_batch_id;
      (prefetch ? prefetch_batches : demand_batches).push_back(job);
    }
    if (batch_id != nullptr)
      *batch_id = job->id;
    batch_ready.notify_one();
    return future;
  }

  bool PromotePrefetch(SsdBatchReader::BatchId batch_id) {
    const std::lock_guard<std::mutex> lock(batch_mutex);
    const auto found =
        std::find_if(prefetch_batches.begin(), prefetch_batches.end(),
                     [&](const auto& job) { return job->id == batch_id; });
    if (found == prefetch_batches.end())
      return false;
    demand_batches.push_back(std::move(*found));
    prefetch_batches.erase(found);
    batch_ready.notify_one();
    return true;
  }

  bool CancelPrefetch(SsdBatchReader::BatchId batch_id) {
    std::shared_ptr<BatchJob> cancelled;
    {
      const std::lock_guard<std::mutex> lock(batch_mutex);
      const auto found =
          std::find_if(prefetch_batches.begin(), prefetch_batches.end(),
                       [&](const auto& job) { return job->id == batch_id; });
      if (found == prefetch_batches.end())
        return false;
      cancelled = std::move(*found);
      prefetch_batches.erase(found);
    }
    SsdReadBatchResult result;
    result.requests = std::move(cancelled->requests);
    result.error = "prefetch batch cancelled before submission";
    result.elapsed = std::chrono::nanoseconds{0};
    result.bytes_read = 0;
    result.success = false;
    for (auto& request : result.requests) {
      request.completed = false;
      request.transferred_bytes = 0;
#ifdef _WIN32
      request.system_error = ERROR_OPERATION_ABORTED;
#else
      request.system_error = ECANCELED;
#endif
    }
    cancelled->completion.set_value(std::move(result));
    return true;
  }

  void BatchWorker() {
    for (;;) {
      std::shared_ptr<BatchJob> job;
      {
        std::unique_lock<std::mutex> lock(batch_mutex);
        batch_ready.wait(lock, [&] {
          return stopping || !demand_batches.empty() ||
                 !prefetch_batches.empty();
        });
        if (demand_batches.empty() && prefetch_batches.empty() && stopping)
          return;
        auto& queue =
            demand_batches.empty() ? prefetch_batches : demand_batches;
        job = queue.front();
        queue.pop_front();
      }
      SsdReadBatchResult result;
      result.requests = std::move(job->requests);
      const auto start = std::chrono::steady_clock::now();
      try {
        result.success = ReadBatchInline(result.requests, &result.error,
                                         &result.bytes_read, !job->prefetch);
      } catch (const std::exception& error) {
        result.error = error.what();
        result.success = false;
      } catch (...) {
        result.error = "SSD batch reader failed with an unknown exception";
        result.success = false;
      }
      result.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start);
      job->completion.set_value(std::move(result));
      batch_ready.notify_one();
    }
  }

  bool ReadBatchInline(std::span<SsdReadRequest> requests,
                       std::string* error_msg, std::uint64_t* bytes_read,
                       bool demand) {
    if (error_msg != nullptr)
      error_msg->clear();
    if (bytes_read != nullptr)
      *bytes_read = 0;
    for (auto& request : requests) {
      if (!PrepareRequest(request, error_msg))
        return false;
      if (request.used_scratch && request.aligned_bytes > scratch_capacity) {
        if (error_msg != nullptr)
          *error_msg = "unaligned SSD read exceeds the worker scratch capacity";
        return false;
      }
    }
    if (requests.empty())
      return true;
    batch_bytes_read = 0;
#ifdef _WIN32
    const bool success = ReadBatchWindows(requests, error_msg, demand);
#else
    const bool success = ReadBatchPosix(requests, error_msg, demand);
#endif
    if (bytes_read != nullptr)
      *bytes_read = batch_bytes_read;
    return success;
  }

#ifdef _WIN32
  struct ReadSegment {
    std::size_t request_index{0};
    std::size_t buffer_offset{0};
  };

  struct ReadGroup {
    std::uint64_t file_offset{0};
    std::size_t bytes{0};
    std::vector<ReadSegment> segments;
  };

  struct Operation {
    OVERLAPPED overlapped{};
    std::span<SsdReadRequest> requests;
    std::vector<ReadSegment> segments;
    std::uint64_t file_offset{0};
    std::size_t bytes{0};
    void* io_buffer{nullptr};
    bool uses_scratch{false};
    std::size_t scratch_lease{std::numeric_limits<std::size_t>::max()};
  };

  void Complete(Operation* operation, DWORD bytes, DWORD error) {
    for (const auto& segment : operation->segments) {
      auto& request = operation->requests[segment.request_index];
      const std::size_t required =
          segment.buffer_offset + request.file_subpage_offset + request.bytes;
      const bool request_ok = bytes >= required && (error == ERROR_SUCCESS ||
                                                    error == ERROR_HANDLE_EOF);
      request.transferred_bytes =
          bytes > segment.buffer_offset
              ? std::min<std::size_t>(request.aligned_bytes,
                                      bytes - segment.buffer_offset)
              : 0;
      request.completed = request_ok;
      request.system_error =
          request_ok ? ERROR_SUCCESS
                     : (error == ERROR_SUCCESS ? ERROR_HANDLE_EOF : error);
      if (!request_ok)
        continue;
      if (operation->uses_scratch) {
        const auto* source =
            static_cast<const std::uint8_t*>(operation->io_buffer) +
            segment.buffer_offset + request.file_subpage_offset;
        std::memcpy(request.destination, source, request.bytes);
        request.payload_offset = 0;
        request.used_scratch = true;
      } else {
        request.payload_offset = request.file_subpage_offset;
        request.used_scratch = false;
      }
    }
    if (scratch_pool != nullptr &&
        operation->scratch_lease != std::numeric_limits<std::size_t>::max()) {
      scratch_pool->Release(operation->scratch_lease);
      operation->scratch_lease = std::numeric_limits<std::size_t>::max();
    }
    {
      const std::lock_guard<std::mutex> lock(completion_mutex);
      ++completed;
      batch_bytes_read += bytes;
    }
    completion_ready.notify_one();
  }

  void CompletionWorker() {
    for (;;) {
      DWORD bytes = 0;
      ULONG_PTR key = 0;
      OVERLAPPED* overlapped = nullptr;
      const BOOL ok =
          GetQueuedCompletionStatus(port, &bytes, &key, &overlapped, INFINITE);
      (void)key;
      if (overlapped == nullptr)
        return;
      const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
      auto* operation = reinterpret_cast<Operation*>(overlapped);
      Complete(operation, bytes, error);
    }
  }

  bool ReadBatchWindows(std::span<SsdReadRequest> requests,
                        std::string* error_msg, bool demand) {
    const auto order = SortedRequestIndices(requests);
    std::vector<ReadGroup> groups;
    groups.reserve(order.size());
    for (const std::size_t request_index : order) {
      const auto& request = requests[request_index];
      const std::uint64_t begin =
          request.file_offset - request.file_subpage_offset;
      const std::uint64_t end = begin + request.aligned_bytes;
      bool merged = false;
      if (!groups.empty()) {
        auto& previous = groups.back();
        const std::uint64_t previous_end =
            previous.file_offset + previous.bytes;
        const std::uint64_t combined_end = std::max(previous_end, end);
        if (begin <= previous_end &&
            combined_end - previous.file_offset <= scratch_capacity) {
          previous.segments.push_back(
              {request_index,
               static_cast<std::size_t>(begin - previous.file_offset)});
          previous.bytes =
              static_cast<std::size_t>(combined_end - previous.file_offset);
          merged = true;
        }
      }
      if (!merged) {
        ReadGroup group;
        group.file_offset = begin;
        group.bytes = request.aligned_bytes;
        group.segments.push_back({request_index, 0});
        groups.push_back(std::move(group));
      }
    }

    const std::size_t wave_size =
        scratch_pool != nullptr
            ? std::min<std::size_t>(max_inflight_reads,
                                    scratch_pool->buffer_count())
            : scratch_buffers.size();
    for (std::size_t base = 0; base < groups.size(); base += wave_size) {
      const std::size_t count = std::min(wave_size, groups.size() - base);
      std::vector<Operation> operations(count);
      {
        const std::lock_guard<std::mutex> lock(completion_mutex);
        completed = 0;
      }
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto& group = groups[base + lane];
        auto& operation = operations[lane];
        operation.requests = requests;
        operation.segments = group.segments;
        operation.file_offset = group.file_offset;
        operation.bytes = group.bytes;
        operation.uses_scratch =
            group.segments.size() != 1 ||
            requests[group.segments.front().request_index].used_scratch;
        void* scratch = nullptr;
        if (scratch_pool != nullptr) {
          operation.scratch_lease = scratch_pool->Acquire(demand);
          scratch = scratch_pool->Buffer(operation.scratch_lease);
        } else {
          scratch = scratch_buffers[lane];
        }
        operation.io_buffer =
            operation.uses_scratch
                ? scratch
                : requests[group.segments.front().request_index].destination;
        operation.overlapped.Offset =
            static_cast<DWORD>(operation.file_offset & 0xffffffffULL);
        operation.overlapped.OffsetHigh =
            static_cast<DWORD>(operation.file_offset >> 32);
        if (operation.bytes > std::numeric_limits<DWORD>::max()) {
          Complete(&operation, 0, ERROR_INVALID_PARAMETER);
          continue;
        }
        const BOOL started = ReadFile(file, operation.io_buffer,
                                      static_cast<DWORD>(operation.bytes),
                                      nullptr, &operation.overlapped);
        if (!started) {
          const DWORD error = GetLastError();
          if (error != ERROR_IO_PENDING)
            Complete(&operation, 0, error);
        }
      }
      {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_ready.wait(lock, [&] { return completed == count; });
      }
    }
    for (const auto& request : requests) {
      if (!request.completed) {
        if (error_msg != nullptr) {
          *error_msg = "overlapped SSD read failed (Win32 error " +
                       std::to_string(request.system_error) + ")";
        }
        return false;
      }
    }
    return true;
  }
#else
  struct Job {
    SsdReadRequest* request{nullptr};
    void* io_buffer{nullptr};
    bool uses_scratch{false};
    std::size_t scratch_lease{std::numeric_limits<std::size_t>::max()};
  };

  void ReadWorker(std::size_t worker_index) {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        work_ready.wait(lock, [&] {
          return worker_stopping || !worker_queues[worker_index].empty();
        });
        if (worker_stopping && worker_queues[worker_index].empty())
          return;
        job = worker_queues[worker_index].front();
        worker_queues[worker_index].pop_front();
      }
      auto& request = *job.request;
      auto* destination = static_cast<std::uint8_t*>(job.io_buffer);
      const std::uint64_t begin =
          request.file_offset - request.file_subpage_offset;
      const std::size_t required = request.file_subpage_offset + request.bytes;
      std::size_t done = 0;
      std::uint32_t error = 0;
      while (done < request.aligned_bytes) {
        const ssize_t count =
            ::pread(fd, destination + done, request.aligned_bytes - done,
                    static_cast<off_t>(begin + done));
        if (count < 0) {
          if (errno == EINTR)
            continue;
          error = static_cast<std::uint32_t>(errno);
          break;
        }
        if (count == 0) {
          if (done < required)
            error = static_cast<std::uint32_t>(EIO);
          break;
        }
        done += static_cast<std::size_t>(count);
        if (done >= required)
          break;
        if (done < request.aligned_bytes &&
            (done % kAlignment != 0 || (begin + done) % kAlignment != 0)) {
          error = static_cast<std::uint32_t>(EINVAL);
          break;
        }
      }
      request.system_error = error;
      request.completed = error == 0 && done >= required;
      request.transferred_bytes = done;
      if (request.completed && job.uses_scratch) {
        std::memcpy(request.destination,
                    destination + request.file_subpage_offset, request.bytes);
        request.payload_offset = 0;
        request.used_scratch = true;
      } else if (request.completed) {
        request.payload_offset = request.file_subpage_offset;
        request.used_scratch = false;
      }
      if (scratch_pool != nullptr &&
          job.scratch_lease != std::numeric_limits<std::size_t>::max())
        scratch_pool->Release(job.scratch_lease);
      {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        ++completed;
      }
      completion_ready.notify_one();
    }
  }

  bool ReadBatchPosix(std::span<SsdReadRequest> requests,
                      std::string* error_msg, bool demand) {
    const auto order = SortedRequestIndices(requests);
    const std::size_t wave_size =
        scratch_pool != nullptr ? std::min({max_inflight_reads, workers.size(),
                                            scratch_pool->buffer_count()})
                                : workers.size();
    for (std::size_t base = 0; base < order.size(); base += wave_size) {
      const std::size_t count = std::min(wave_size, order.size() - base);
      {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        completed = 0;
      }
      for (std::size_t lane = 0; lane < count; ++lane) {
        auto& request = requests[order[base + lane]];
        std::size_t lease = std::numeric_limits<std::size_t>::max();
        void* scratch = nullptr;
        if (scratch_pool != nullptr) {
          lease = scratch_pool->Acquire(demand);
          scratch = scratch_pool->Buffer(lease);
        } else {
          scratch = scratch_buffers[lane];
        }
        void* io_buffer = request.used_scratch ? scratch : request.destination;
        {
          const std::lock_guard<std::mutex> lock(queue_mutex);
          worker_queues[lane].push_back(
              {&request, io_buffer, request.used_scratch, lease});
        }
        work_ready.notify_one();
      }
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        completion_ready.wait(lock, [&] { return completed == count; });
        for (std::size_t lane = 0; lane < count; ++lane)
          batch_bytes_read += requests[order[base + lane]].transferred_bytes;
      }
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto& request = requests[order[base + lane]];
        if (!request.completed) {
          if (error_msg != nullptr) {
            *error_msg = "direct SSD read failed (errno " +
                         std::to_string(request.system_error) + ")";
          }
          return false;
        }
      }
    }
    return true;
  }
#endif

  std::mutex batch_mutex;
  std::condition_variable batch_ready;
  std::deque<std::shared_ptr<BatchJob>> demand_batches;
  std::deque<std::shared_ptr<BatchJob>> prefetch_batches;
  std::thread batch_worker;
  SsdBatchReader::BatchId next_batch_id{0};
  bool stopping{false};
  std::mutex completion_mutex;
  std::condition_variable completion_ready;
  std::size_t completed{0};
  std::uint64_t batch_bytes_read{0};
  std::vector<std::thread> workers;
  std::vector<void*> scratch_buffers;
  std::size_t scratch_capacity{0};
  std::size_t max_inflight_reads{0};
  std::shared_ptr<SsdScratchPool> scratch_pool;
#ifdef _WIN32
  HANDLE file{INVALID_HANDLE_VALUE};
  HANDLE port{nullptr};
#else
  int fd{-1};
  std::mutex queue_mutex;
  std::condition_variable work_ready;
  std::vector<std::deque<Job>> worker_queues;
  bool worker_stopping{false};
#endif
};

SsdBatchReader::SsdBatchReader(std::filesystem::path path)
    : SsdBatchReader(std::move(path), Options{}) {}

SsdBatchReader::SsdBatchReader(std::filesystem::path path, Options options)
    : impl_(std::make_unique<Impl>(path, options)) {}

SsdBatchReader::~SsdBatchReader() = default;
SsdBatchReader::SsdBatchReader(SsdBatchReader&&) noexcept = default;
SsdBatchReader& SsdBatchReader::operator=(SsdBatchReader&&) noexcept = default;

bool SsdBatchReader::ReadBatch(std::span<SsdReadRequest> requests,
                               std::string* error_msg,
                               std::uint64_t* bytes_read) {
  return impl_ != nullptr && impl_->ReadBatch(requests, error_msg, bytes_read);
}

std::future<SsdReadBatchResult> SsdBatchReader::SubmitBatchAsync(
    std::vector<SsdReadRequest> requests, BatchId* batch_id) {
  if (impl_ == nullptr) {
    std::promise<SsdReadBatchResult> promise;
    auto future = promise.get_future();
    SsdReadBatchResult result;
    result.error = "SSD reader is unavailable";
    promise.set_value(std::move(result));
    if (batch_id != nullptr)
      *batch_id = 0;
    return future;
  }
  return impl_->EnqueueBatch(std::move(requests), true, batch_id);
}

std::future<SsdReadBatchResult> SsdBatchReader::SubmitBatchDemand(
    std::vector<SsdReadRequest> requests) {
  if (impl_ == nullptr) {
    std::promise<SsdReadBatchResult> promise;
    auto future = promise.get_future();
    SsdReadBatchResult result;
    result.error = "SSD reader is unavailable";
    promise.set_value(std::move(result));
    return future;
  }
  return impl_->EnqueueBatch(std::move(requests), false, nullptr);
}

bool SsdBatchReader::PromotePrefetch(BatchId batch_id) {
  return impl_ != nullptr && impl_->PromotePrefetch(batch_id);
}

bool SsdBatchReader::CancelPrefetch(BatchId batch_id) {
  return impl_ != nullptr && impl_->CancelPrefetch(batch_id);
}

}  // namespace gufo::storage
