#include "src/storage/ssd_batch_reader.hpp"

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

constexpr std::size_t kExperts = 64;
constexpr std::size_t kExpertBytes = 16 * 1024 + 73;
constexpr std::size_t kExpertStride = 32 * 1024;
constexpr std::size_t kSlotStride = kExpertBytes;
constexpr std::uint64_t kFirstExpertOffset = 4096;

__global__ void ChecksumExpert(const std::uint8_t* bytes, std::size_t size,
                               std::uint32_t* result) {
  __shared__ std::uint32_t partial[256];
  const unsigned tid = threadIdx.x;
  std::uint32_t value = 0;
  for (std::size_t i = tid; i < size; i += blockDim.x)
    value ^= bytes[i];
  partial[tid] = value;
  __syncthreads();
  for (unsigned width = blockDim.x / 2; width != 0; width >>= 1) {
    if (tid < width)
      partial[tid] ^= partial[tid + width];
    __syncthreads();
  }
  if (tid == 0)
    result[blockIdx.x] = partial[0];
}

void CheckHip(const char* operation, hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

std::uint8_t Pattern(std::uint64_t offset) {
  std::uint64_t value = offset + 0x9e3779b97f4a7c15ULL;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<std::uint8_t>(value >> 24);
}

std::uint64_t ExpertOffset(std::size_t index) {
  return kFirstExpertOffset + index * kExpertStride + (index % 7) * 13 + 1;
}

struct Resources {
  std::filesystem::path file;
  void* host_slots{nullptr};
  void* direct_slot{nullptr};
  std::uint32_t* device_checksums{nullptr};
  hipStream_t stream{nullptr};

  ~Resources() {
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
    if (device_checksums != nullptr)
      (void)hipFree(device_checksums);
    if (direct_slot != nullptr)
      (void)hipHostFree(direct_slot);
    if (host_slots != nullptr)
      (void)hipHostFree(host_slots);
    if (!file.empty()) {
      std::error_code error;
      std::filesystem::remove(file, error);
    }
  }
};

void WriteFixture(Resources& resources, std::vector<std::uint32_t>& expected) {
  const std::uint64_t file_bytes = ExpertOffset(kExperts - 1) + kExpertBytes;
  std::vector<std::uint8_t> contents(static_cast<std::size_t>(file_bytes));
  for (std::size_t i = 0; i < contents.size(); ++i)
    contents[i] = Pattern(i);
  expected.resize(kExperts);
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    std::uint32_t checksum = 0;
    const auto offset = ExpertOffset(expert);
    for (std::size_t byte = 0; byte < kExpertBytes; ++byte)
      checksum ^= contents[static_cast<std::size_t>(offset) + byte];
    expected[expert] = checksum;
  }

  resources.file =
      std::filesystem::temp_directory_path() /
      ("gufo-ssd-batch-reader-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".bin");
  std::ofstream output(resources.file, std::ios::binary | std::ios::trunc);
  if (!output || !output.write(reinterpret_cast<const char*>(contents.data()),
                               static_cast<std::streamsize>(contents.size())))
    throw std::runtime_error("failed to write SSD reader fixture");
}

void Run() {
  int devices = 0;
  CheckHip("hipGetDeviceCount", hipGetDeviceCount(&devices));
  if (devices == 0)
    throw std::runtime_error("no HIP device found");
  CheckHip("hipSetDevice", hipSetDevice(0));

  Resources resources;
  CheckHip("hipHostMalloc slots",
           hipHostMalloc(&resources.host_slots, kExperts * kSlotStride,
                         hipHostMallocMapped));
  void* device_slots = nullptr;
  CheckHip("hipHostGetDevicePointer slots",
           hipHostGetDevicePointer(&device_slots, resources.host_slots, 0));
  CheckHip("hipMalloc checksums", hipMalloc(&resources.device_checksums,
                                            kExperts * sizeof(std::uint32_t)));
  CheckHip("hipStreamCreate",
           hipStreamCreateWithFlags(&resources.stream, hipStreamNonBlocking));

  std::vector<std::uint32_t> expected;
  WriteFixture(resources, expected);
  gufo::storage::SsdBatchReader reader(resources.file, {.io_threads = 8});
  std::vector<gufo::storage::SsdReadRequest> requests(kExperts);
  auto* slots = static_cast<std::uint8_t*>(resources.host_slots);
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    requests[expert] = {
        .file_offset = ExpertOffset(expert),
        .bytes = kExpertBytes,
        .destination = slots + expert * kSlotStride,
        .destination_capacity = kSlotStride,
    };
  }
  std::string error;
  if (!reader.ReadBatch(requests, &error))
    throw std::runtime_error("ReadBatch failed: " + error);
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    const auto& request = requests[expert];
    if (!request.completed || !request.used_scratch ||
        request.payload_offset != 0 || request.aligned_bytes <= kSlotStride)
      throw std::runtime_error("reader returned an invalid slot range");
    const auto* payload = request.Payload();
    for (std::size_t byte = 0; byte < kExpertBytes; ++byte) {
      if (payload[byte] != Pattern(ExpertOffset(expert) + byte)) {
        throw std::runtime_error("packed slot byte mismatch for expert " +
                                 std::to_string(expert));
      }
    }
  }

  auto* device_bytes = static_cast<std::uint8_t*>(device_slots);
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    const auto* payload =
        device_bytes + expert * kSlotStride + requests[expert].payload_offset;
    hipLaunchKernelGGL(ChecksumExpert, dim3(1), dim3(256), 0, resources.stream,
                       payload, kExpertBytes,
                       resources.device_checksums + expert);
  }
  CheckHip("ChecksumExpert launch", hipGetLastError());
  CheckHip("HIP stream synchronize", hipStreamSynchronize(resources.stream));
  std::array<std::uint32_t, kExperts> actual{};
  CheckHip("copy checksums",
           hipMemcpy(actual.data(), resources.device_checksums,
                     actual.size() * sizeof(actual[0]), hipMemcpyDeviceToHost));
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    if (actual[expert] != expected[expert]) {
      throw std::runtime_error("checksum mismatch for expert " +
                               std::to_string(expert));
    }
  }

  CheckHip("hipHostMalloc direct slot",
           hipHostMalloc(&resources.direct_slot, 4096, hipHostMallocMapped));
  void* direct_device_slot = nullptr;
  CheckHip(
      "hipHostGetDevicePointer direct slot",
      hipHostGetDevicePointer(&direct_device_slot, resources.direct_slot, 0));
  gufo::storage::SsdReadRequest direct_request{
      .file_offset = 0,
      .bytes = 4096,
      .destination = resources.direct_slot,
      .destination_capacity = 4096,
  };
  if (!reader.ReadBatch(std::span(&direct_request, 1), &error) ||
      !direct_request.completed || direct_request.used_scratch) {
    throw std::runtime_error("aligned direct-read fast path failed: " + error);
  }
  const auto* direct_payload = direct_request.Payload();
  for (std::size_t byte = 0; byte < direct_request.bytes; ++byte) {
    if (direct_payload[byte] != Pattern(byte))
      throw std::runtime_error("aligned direct-read byte mismatch");
  }
  hipLaunchKernelGGL(ChecksumExpert, dim3(1), dim3(256), 0, resources.stream,
                     static_cast<const std::uint8_t*>(direct_device_slot),
                     direct_request.bytes, resources.device_checksums);
  CheckHip("direct fast-path checksum launch", hipGetLastError());
  CheckHip("direct fast-path checksum sync",
           hipStreamSynchronize(resources.stream));
  std::uint32_t direct_checksum = 0;
  CheckHip("direct fast-path checksum copy",
           hipMemcpy(&direct_checksum, resources.device_checksums,
                     sizeof(direct_checksum), hipMemcpyDeviceToHost));
  std::uint32_t expected_direct = 0;
  for (std::size_t byte = 0; byte < direct_request.bytes; ++byte)
    expected_direct ^= Pattern(byte);
  if (direct_checksum != expected_direct)
    throw std::runtime_error("direct fast-path HIP checksum mismatch");
  std::cout << "ssd_batch_reader_test: " << kExperts
            << " packed unaligned slots and the aligned fast path passed byte "
               "and HIP "
               "checksums\n";
}

}  // namespace

int main() {
  try {
    Run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ssd_batch_reader_test: " << error.what() << '\n';
    return 1;
  }
}
