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

}  // namespace

struct SsdBatchReader::Impl {
  explicit Impl(const std::filesystem::path& path, Options options) {
    if (options.io_threads == 0 || options.io_threads > 64)
      throw std::invalid_argument("SSD reader thread count must be in 1..64");
    if (options.max_scratch_bytes == 0 ||
        options.max_scratch_bytes >
            std::numeric_limits<std::size_t>::max() - kAlignment) {
      throw std::invalid_argument("SSD reader scratch size is invalid");
    }
    scratch_capacity = AlignUp(options.max_scratch_bytes + kAlignment);
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
      AllocateScratch(options.io_threads);
      workers.reserve(options.io_threads);
      for (std::size_t i = 0; i < options.io_threads; ++i) {
        workers.emplace_back([this] { CompletionWorker(); });
      }
    } catch (...) {
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
      AllocateScratch(options.io_threads);
      worker_queues.resize(options.io_threads);
      workers.reserve(options.io_threads);
      for (std::size_t i = 0; i < options.io_threads; ++i) {
        workers.emplace_back([this, i] { ReadWorker(i); });
      }
    } catch (...) {
      {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        stopping = true;
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
      stopping = true;
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
    if (error_msg != nullptr)
      error_msg->clear();
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
    const std::lock_guard<std::mutex> batch_lock(batch_mutex);
#ifdef _WIN32
    return ReadBatchWindows(requests, error_msg);
#else
    return ReadBatchPosix(requests, error_msg);
#endif
  }

#ifdef _WIN32
  struct Operation {
    OVERLAPPED overlapped{};
    SsdReadRequest* request{nullptr};
    void* io_buffer{nullptr};
    bool uses_scratch{false};
  };

  void Complete(Operation* operation, DWORD bytes, DWORD error) {
    SsdReadRequest* request = operation->request;
    request->transferred_bytes = bytes;
    request->completed =
        bytes >= request->file_subpage_offset + request->bytes &&
        (error == ERROR_SUCCESS || error == ERROR_HANDLE_EOF);
    request->system_error =
        request->completed
            ? ERROR_SUCCESS
            : (error == ERROR_SUCCESS ? ERROR_HANDLE_EOF : error);
    if (request->completed && operation->uses_scratch) {
      const auto* source =
          static_cast<const std::uint8_t*>(operation->io_buffer) +
          request->file_subpage_offset;
      std::memcpy(request->destination, source, request->bytes);
      request->payload_offset = 0;
      request->used_scratch = true;
    } else if (request->completed) {
      request->payload_offset = request->file_subpage_offset;
      request->used_scratch = false;
    }
    {
      const std::lock_guard<std::mutex> lock(completion_mutex);
      ++completed;
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
                        std::string* error_msg) {
    const std::size_t wave_size = scratch_buffers.size();
    for (std::size_t base = 0; base < requests.size(); base += wave_size) {
      const std::size_t count = std::min(wave_size, requests.size() - base);
      std::vector<Operation> operations(count);
      {
        const std::lock_guard<std::mutex> lock(completion_mutex);
        completed = 0;
      }
      for (std::size_t lane = 0; lane < count; ++lane) {
        auto& request = requests[base + lane];
        auto& operation = operations[lane];
        operation.request = &request;
        operation.uses_scratch = request.used_scratch;
        operation.io_buffer =
            request.used_scratch ? scratch_buffers[lane] : request.destination;
        const std::uint64_t aligned_offset =
            request.file_offset - request.file_subpage_offset;
        operation.overlapped.Offset =
            static_cast<DWORD>(aligned_offset & 0xffffffffULL);
        operation.overlapped.OffsetHigh =
            static_cast<DWORD>(aligned_offset >> 32);
        if (request.aligned_bytes > std::numeric_limits<DWORD>::max()) {
          Complete(&operation, 0, ERROR_INVALID_PARAMETER);
          continue;
        }
        const BOOL started = ReadFile(file, operation.io_buffer,
                                      static_cast<DWORD>(request.aligned_bytes),
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
  };

  void ReadWorker(std::size_t worker_index) {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        work_ready.wait(lock, [&] {
          return stopping || !worker_queues[worker_index].empty();
        });
        if (stopping && worker_queues[worker_index].empty())
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
      {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        ++completed;
      }
      completion_ready.notify_one();
    }
  }

  bool ReadBatchPosix(std::span<SsdReadRequest> requests,
                      std::string* error_msg) {
    const std::size_t wave_size = workers.size();
    for (std::size_t base = 0; base < requests.size(); base += wave_size) {
      const std::size_t count = std::min(wave_size, requests.size() - base);
      {
        const std::lock_guard<std::mutex> lock(queue_mutex);
        completed = 0;
        for (std::size_t lane = 0; lane < count; ++lane) {
          auto& request = requests[base + lane];
          void* io_buffer = request.used_scratch ? scratch_buffers[lane]
                                                 : request.destination;
          worker_queues[lane].push_back(
              {&request, io_buffer, request.used_scratch});
        }
      }
      work_ready.notify_all();
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        completion_ready.wait(lock, [&] { return completed == count; });
      }
      for (std::size_t lane = 0; lane < count; ++lane) {
        const auto& request = requests[base + lane];
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
  std::mutex completion_mutex;
  std::condition_variable completion_ready;
  std::size_t completed{0};
  std::vector<std::thread> workers;
  std::vector<void*> scratch_buffers;
  std::size_t scratch_capacity{0};
#ifdef _WIN32
  HANDLE file{INVALID_HANDLE_VALUE};
  HANDLE port{nullptr};
#else
  int fd{-1};
  std::mutex queue_mutex;
  std::condition_variable work_ready;
  std::vector<std::deque<Job>> worker_queues;
  bool stopping{false};
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
                               std::string* error_msg) {
  return impl_ != nullptr && impl_->ReadBatch(requests, error_msg);
}

std::future<SsdReadBatchResult> SsdBatchReader::SubmitBatchAsync(
    std::vector<SsdReadRequest> requests) {
  return std::async(
      std::launch::async, [this, requests = std::move(requests)]() mutable {
        SsdReadBatchResult result;
        result.requests = std::move(requests);
        const auto start = std::chrono::steady_clock::now();
        result.success = ReadBatch(result.requests, &result.error);
        result.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start);
        return result;
      });
}

}  // namespace gufo::storage
