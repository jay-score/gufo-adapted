#include <hip/hip_runtime.h>
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kPage = 4096;
constexpr std::size_t kDefaultMiB = 256;
constexpr unsigned kReadRepeats = 3;
constexpr std::size_t kSubpageOffset = kPage + 123;
constexpr std::size_t kSubpageBytes = 64 * 1024 - 17;

__global__ void ChecksumBytes(const std::uint8_t* data, std::size_t bytes,
                              std::uint32_t* result) {
  __shared__ std::uint32_t partial[256];
  const unsigned tid = threadIdx.x;
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  std::uint32_t value = 0;
  for (std::size_t i = first; i < bytes; i += stride) {
    value ^= data[i];
  }
  partial[tid] = value;
  __syncthreads();
  for (unsigned width = blockDim.x / 2; width != 0; width >>= 1) {
    if (tid < width) {
      partial[tid] ^= partial[tid + width];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicXor(result, partial[0]);
  }
}

struct TempFile {
  std::wstring path;
  HANDLE handle{INVALID_HANDLE_VALUE};

  ~TempFile() {
    if (handle != INVALID_HANDLE_VALUE) {
      CloseHandle(handle);
    }
    if (!path.empty()) {
      DeleteFileW(path.c_str());
    }
  }
};

struct HostBuffer {
  void* host{nullptr};
  void* device{nullptr};
  bool hip_allocated{false};
  bool registered{false};
  std::size_t bytes{0};

  ~HostBuffer() {
    if (registered) {
      const hipError_t status = hipHostUnregister(host);
      if (status != hipSuccess) {
        std::cerr << "hipHostUnregister failed: " << hipGetErrorString(status)
                  << '\n';
      }
    }
    if (hip_allocated && host != nullptr) {
      const hipError_t status = hipHostFree(host);
      if (status != hipSuccess) {
        std::cerr << "hipHostFree failed: " << hipGetErrorString(status)
                  << '\n';
      }
    }
    if (!hip_allocated && host != nullptr) {
      VirtualFree(host, 0, MEM_RELEASE);
    }
  }
};

struct ReadResult {
  bool ok{false};
  DWORD error{ERROR_SUCCESS};
  DWORD bytes{0};
  double milliseconds{0.0};
};

struct CandidateResult {
  const char* name{nullptr};
  bool allocated{false};
  bool device_pointer{false};
  bool direct_read{false};
  bool checksum_ok{false};
  bool subpage_ok{false};
  double gib_per_second{0.0};
  DWORD error{ERROR_SUCCESS};
  std::uintptr_t host_mod_4096{0};
};

void PrintHipFailure(const char* operation, hipError_t status) {
  std::cerr << operation << " failed: " << hipGetErrorString(status) << " ("
            << static_cast<int>(status) << ")\n";
  // Allocation failures can leave a stale runtime last-error for the first
  // subsequent kernel launch even when the next HIP call succeeds.
  (void)hipGetLastError();
}

bool HipOk(const char* operation, hipError_t status) {
  if (status == hipSuccess) {
    return true;
  }
  PrintHipFailure(operation, status);
  return false;
}

std::uint8_t PatternByte(std::uint64_t offset) {
  std::uint64_t x = offset + 0x9e3779b97f4a7c15ULL;
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return static_cast<std::uint8_t>(x >> 24);
}

std::wstring TempPath() {
  std::vector<wchar_t> path(MAX_PATH + 1);
  const DWORD length =
      GetTempPathW(static_cast<DWORD>(path.size()), path.data());
  if (length == 0 || length >= path.size()) {
    return {};
  }
  return std::wstring(path.data(), length);
}

bool CreateFixture(TempFile* file, std::size_t bytes,
                   std::uint32_t* full_checksum,
                   std::uint32_t* subpage_checksum) {
  const std::wstring directory = TempPath();
  if (directory.empty()) {
    std::cerr << "GetTempPathW failed: " << GetLastError() << '\n';
    return false;
  }
  file->path = directory + L"gufo-ssd-gtt-spike-" +
               std::to_wstring(GetCurrentProcessId()) + L"-" +
               std::to_wstring(GetTickCount64()) + L".bin";
  file->handle = CreateFileW(file->path.c_str(), GENERIC_WRITE, 0, nullptr,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file->handle == INVALID_HANDLE_VALUE) {
    std::cerr << "CreateFileW fixture failed: " << GetLastError() << '\n';
    return false;
  }

  constexpr std::size_t kChunk = 1024 * 1024;
  std::vector<std::uint8_t> chunk(kChunk);
  std::uint32_t whole = 0;
  std::uint32_t subpage = 0;
  const std::uint64_t subpage_end = kSubpageOffset + kSubpageBytes;
  for (std::size_t base = 0; base < bytes; base += kChunk) {
    const std::size_t count = std::min(kChunk, bytes - base);
    for (std::size_t i = 0; i < count; ++i) {
      const std::uint64_t offset = base + i;
      const std::uint8_t value = PatternByte(offset);
      chunk[i] = value;
      whole ^= value;
      if (offset >= kSubpageOffset && offset < subpage_end) {
        subpage ^= value;
      }
    }
    DWORD written = 0;
    if (!WriteFile(file->handle, chunk.data(), static_cast<DWORD>(count),
                   &written, nullptr) ||
        written != count) {
      std::cerr << "WriteFile fixture failed: " << GetLastError() << '\n';
      return false;
    }
  }
  if (!FlushFileBuffers(file->handle)) {
    std::cerr << "FlushFileBuffers failed: " << GetLastError() << '\n';
    return false;
  }
  CloseHandle(file->handle);
  file->handle = INVALID_HANDLE_VALUE;
  *full_checksum = whole;
  *subpage_checksum = subpage;
  return true;
}

HANDLE OpenDirectRead(const std::wstring& path) {
  return CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                     OPEN_EXISTING,
                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                         FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                     nullptr);
}

ReadResult ReadAt(HANDLE file, HANDLE event, std::uint64_t offset, void* buffer,
                  std::size_t bytes) {
  ReadResult result;
  OVERLAPPED overlapped{};
  overlapped.hEvent = event;
  overlapped.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  ResetEvent(event);

  const auto begin = std::chrono::steady_clock::now();
  const BOOL started =
      ReadFile(file, buffer, static_cast<DWORD>(bytes), nullptr, &overlapped);
  if (!started) {
    const DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      result.error = error;
      return result;
    }
  }
  if (!GetOverlappedResult(file, &overlapped, &result.bytes, TRUE)) {
    result.error = GetLastError();
    return result;
  }
  result.milliseconds = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
  result.ok = result.bytes == bytes;
  if (!result.ok) {
    result.error = ERROR_HANDLE_EOF;
  }
  return result;
}

bool KernelChecksum(const std::uint8_t* device_data, std::size_t bytes,
                    hipStream_t stream, std::uint32_t* device_checksum,
                    std::uint32_t* checksum) {
  if (!HipOk("hipMemsetAsync checksum",
             hipMemsetAsync(device_checksum, 0, sizeof(*device_checksum),
                            stream))) {
    return false;
  }
  hipLaunchKernelGGL(ChecksumBytes, dim3(1024), dim3(256), 0, stream,
                     device_data, bytes, device_checksum);
  if (!HipOk("ChecksumBytes launch", hipGetLastError()) ||
      !HipOk("hipStreamSynchronize checksum", hipStreamSynchronize(stream)) ||
      !HipOk("hipMemcpy checksum",
             hipMemcpy(checksum, device_checksum, sizeof(*checksum),
                       hipMemcpyDeviceToHost))) {
    return false;
  }
  return true;
}

bool AllocateCandidate(const char* name, unsigned flags, bool virtual_alloc,
                       std::size_t bytes, HostBuffer* buffer) {
  buffer->bytes = bytes;
  buffer->hip_allocated = !virtual_alloc;
  buffer->registered = virtual_alloc;
  buffer->host = nullptr;
  buffer->device = nullptr;
  hipError_t status = hipSuccess;
  if (virtual_alloc) {
    buffer->host =
        VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (buffer->host == nullptr) {
      std::cerr << name << ": VirtualAlloc failed: " << GetLastError() << '\n';
      return false;
    }
    status = hipHostRegister(buffer->host, bytes, hipHostRegisterMapped);
    if (status != hipSuccess) {
      PrintHipFailure("hipHostRegister(VirtualAlloc, Mapped)", status);
      buffer->registered = false;
      return false;
    }
  } else {
    status = hipHostMalloc(&buffer->host, bytes, flags);
    if (status != hipSuccess) {
      PrintHipFailure(name, status);
      return false;
    }
  }
  status = hipHostGetDevicePointer(&buffer->device, buffer->host, 0);
  if (status != hipSuccess) {
    PrintHipFailure("hipHostGetDevicePointer", status);
    return false;
  }
  return true;
}

CandidateResult RunCandidate(const char* name, unsigned flags,
                             bool virtual_alloc, HANDLE file, HANDLE event,
                             std::size_t bytes, std::uint32_t expected_checksum,
                             std::uint32_t expected_subpage,
                             std::uint32_t* device_checksum,
                             hipStream_t stream) {
  CandidateResult result;
  result.name = name;
  HostBuffer buffer;
  result.allocated =
      AllocateCandidate(name, flags, virtual_alloc, bytes, &buffer);
  if (!result.allocated) {
    return result;
  }
  result.host_mod_4096 =
      reinterpret_cast<std::uintptr_t>(buffer.host) & (kPage - 1);
  result.device_pointer = buffer.device != nullptr;
  std::cout << "candidate=" << name << " host_mod_4096=" << result.host_mod_4096
            << " gpu_pointer=" << (result.device_pointer ? "yes" : "no")
            << '\n';

  double elapsed_ms = 0.0;
  std::size_t bytes_read = 0;
  bool all_checksums = true;
  for (unsigned repeat = 0; repeat < kReadRepeats; ++repeat) {
    const ReadResult read = ReadAt(file, event, 0, buffer.host, bytes);
    if (!read.ok) {
      result.error = read.error;
      std::cout << "  direct_read=FAIL win32_error=" << read.error
                << (read.error == ERROR_INVALID_PARAMETER
                        ? " (ERROR_INVALID_PARAMETER)"
                        : "")
                << '\n';
      return result;
    }
    result.direct_read = true;
    elapsed_ms += read.milliseconds;
    bytes_read += read.bytes;

    std::uint32_t checksum = 0;
    if (!KernelChecksum(static_cast<const std::uint8_t*>(buffer.device), bytes,
                        stream, device_checksum, &checksum)) {
      all_checksums = false;
      break;
    }
    all_checksums = all_checksums && checksum == expected_checksum;
  }
  result.checksum_ok = all_checksums;
  if (elapsed_ms > 0.0) {
    result.gib_per_second = static_cast<double>(bytes_read) /
                            (elapsed_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
  }
  std::cout << "  direct_read=" << (result.direct_read ? "PASS" : "FAIL")
            << " checksum=" << (result.checksum_ok ? "PASS" : "FAIL")
            << " read_GiB_per_s=" << result.gib_per_second
            << " bytes=" << bytes_read << " elapsed_ms=" << elapsed_ms << '\n';

  const std::uint64_t aligned_begin = kSubpageOffset & ~(kPage - 1);
  const std::uint64_t delta = kSubpageOffset - aligned_begin;
  const std::uint64_t aligned_end =
      (kSubpageOffset + kSubpageBytes + kPage - 1) & ~(kPage - 1);
  const std::size_t aligned_bytes = aligned_end - aligned_begin;
  const ReadResult subpage_read =
      ReadAt(file, event, aligned_begin, buffer.host, aligned_bytes);
  if (subpage_read.ok && buffer.device != nullptr) {
    std::uint32_t checksum = 0;
    const auto* row = static_cast<const std::uint8_t*>(buffer.device) + delta;
    result.subpage_ok = KernelChecksum(row, kSubpageBytes, stream,
                                       device_checksum, &checksum) &&
                        checksum == expected_subpage;
  } else {
    result.error = subpage_read.error;
  }
  std::cout << "  subpage_offset=" << kSubpageOffset
            << " aligned_begin=" << aligned_begin << " delta=" << delta
            << " aligned_bytes=" << aligned_bytes
            << " unaligned_expert=" << (result.subpage_ok ? "PASS" : "FAIL")
            << " win32_error=" << subpage_read.error << '\n';
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t mib = kDefaultMiB;
    if (argc > 1) {
      char* end = nullptr;
      const auto parsed = std::strtoull(argv[1], &end, 10);
      if (end == argv[1] || *end != '\0' || parsed < 128 || parsed > 1024 ||
          parsed % 4 != 0) {
        std::cerr << "usage: spike_ssd_gtt_dma [size_mib: 128..1024, multiple "
                     "of 4]\n";
        return 2;
      }
      mib = static_cast<std::size_t>(parsed);
    }
    const std::size_t bytes = mib * 1024 * 1024;
    if (bytes <= kSubpageOffset + kSubpageBytes || bytes % kPage != 0) {
      std::cerr << "fixture size cannot cover the unaligned expert probe\n";
      return 2;
    }

    int device_count = 0;
    if (!HipOk("hipGetDeviceCount", hipGetDeviceCount(&device_count)) ||
        device_count == 0) {
      std::cerr << "No HIP device is available\n";
      return 2;
    }
    hipDeviceProp_t properties{};
    if (!HipOk("hipGetDeviceProperties",
               hipGetDeviceProperties(&properties, 0))) {
      return 2;
    }
    if (!HipOk("hipSetDevice", hipSetDevice(0))) {
      return 2;
    }
    hipStream_t stream = nullptr;
    if (!HipOk("hipStreamCreateWithFlags",
               hipStreamCreateWithFlags(&stream, hipStreamNonBlocking))) {
      return 2;
    }
    std::uint32_t* device_checksum = nullptr;
    if (!HipOk("hipMalloc checksum",
               hipMalloc(&device_checksum, sizeof(*device_checksum)))) {
      (void)hipStreamDestroy(stream);
      return 2;
    }

    TempFile fixture;
    std::uint32_t expected_checksum = 0;
    std::uint32_t expected_subpage = 0;
    if (!CreateFixture(&fixture, bytes, &expected_checksum,
                       &expected_subpage)) {
      (void)hipFree(device_checksum);
      (void)hipStreamDestroy(stream);
      return 2;
    }
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (event == nullptr) {
      std::cerr << "CreateEventW failed: " << GetLastError() << '\n';
      (void)hipFree(device_checksum);
      (void)hipStreamDestroy(stream);
      return 2;
    }
    HANDLE direct_file = OpenDirectRead(fixture.path);
    if (direct_file == INVALID_HANDLE_VALUE) {
      std::cerr << "CreateFileW direct-read handle failed: " << GetLastError()
                << '\n';
      CloseHandle(event);
      (void)hipFree(device_checksum);
      (void)hipStreamDestroy(stream);
      return 2;
    }

    wchar_t volume_path[MAX_PATH]{};
    DWORD sectors_per_cluster = 0;
    DWORD bytes_per_sector = 0;
    DWORD free_clusters = 0;
    DWORD total_clusters = 0;
    if (GetVolumePathNameW(fixture.path.c_str(), volume_path, MAX_PATH) &&
        GetDiskFreeSpaceW(volume_path, &sectors_per_cluster, &bytes_per_sector,
                          &free_clusters, &total_clusters)) {
      std::cout << "volume=";
      std::wcout << volume_path;
      std::cout << " bytes_per_sector=" << bytes_per_sector << '\n';
    }
    std::cout << "device=" << properties.name
              << " HIP_arch=" << properties.gcnArchName << " fixture=" << mib
              << " MiB repeats=" << kReadRepeats
              << " no_buffering=1 overlapped=1\n";

    const std::vector<CandidateResult> results = {
        RunCandidate("hipHostMallocMapped", hipHostMallocMapped, false,
                     direct_file, event, bytes, expected_checksum,
                     expected_subpage, device_checksum, stream),
        RunCandidate("hipHostMallocUncached", hipHostMallocUncached, false,
                     direct_file, event, bytes, expected_checksum,
                     expected_subpage, device_checksum, stream),
        RunCandidate("VirtualAlloc+hipHostRegisterMapped",
                     hipHostRegisterMapped, true, direct_file, event, bytes,
                     expected_checksum, expected_subpage, device_checksum,
                     stream),
    };

    const CandidateResult* winner = nullptr;
    for (const auto& result : results) {
      if (result.direct_read && result.device_pointer && result.checksum_ok &&
          result.subpage_ok && result.gib_per_second > 3.5 &&
          (winner == nullptr ||
           result.gib_per_second > winner->gib_per_second)) {
        winner = &result;
      }
    }
    if (winner != nullptr) {
      std::cout << "decision=PASS path=" << winner->name
                << " read_GiB_per_s=" << winner->gib_per_second
                << " aligned_window_formula=begin(O,4096), delta=O-begin, "
                   "length=round_up(delta+N,4096)\n";
    } else {
      std::cout
          << "decision=NO_WINNER; no candidate met all criteria "
             "(direct read, HIP checksum, sub-page checksum, >3.5 GiB/s)\n";
    }

    CloseHandle(direct_file);
    CloseHandle(event);
    (void)hipFree(device_checksum);
    (void)hipStreamDestroy(stream);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 2;
  }
}
