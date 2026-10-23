#ifndef GUFO_CORE_PLATFORM_MAPPED_FILE_HPP_
#define GUFO_CORE_PLATFORM_MAPPED_FILE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace gufo::platform {

/// Read-only demand-paged view of a whole file.
///
/// Linux maps with PROT_READ/MAP_PRIVATE and a sequential-access hint;
/// Windows uses a read-only section mapping of a FILE_FLAG_SEQUENTIAL_SCAN
/// handle. Either way `data()` is backed by the OS page cache and stays
/// valid for the lifetime of this object. Empty files fail to open.
class ReadOnlyMappedFile {
public:
  ReadOnlyMappedFile() = default;
  ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile(ReadOnlyMappedFile&& other) noexcept;
  ReadOnlyMappedFile& operator=(ReadOnlyMappedFile&& other) noexcept;
  ~ReadOnlyMappedFile();

  /// Map `path` read-only. Returns an empty view and sets `error` on failure.
  static ReadOnlyMappedFile Open(const std::filesystem::path& path,
                                 std::string* error = nullptr);

  const std::uint8_t* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  bool valid() const noexcept { return data_ != nullptr; }

  /// Borrowed POSIX file descriptor backing the mapping; -1 on Windows and
  /// for an empty view. Valid for the lifetime of this object.
  int PlatformDescriptor() const noexcept;

  void Close() noexcept;

private:
  void Reset() noexcept;

  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
#if defined(_WIN32)
  void* handle_ = nullptr;   // HANDLE of the file
  void* mapping_ = nullptr;  // HANDLE of the section
#else
  int fd_ = -1;
#endif
};

/// C-style mapping helper for the safetensors loaders: opens, size-checks,
/// and maps a file read-only with a sequential hint, exposing the raw
/// pointer. Prefer ReadOnlyMappedFile for new code.
struct RawMappedFile {
  void* data{nullptr};
  std::size_t size{0};
  /// Start and length of the OS mapping; differs from `data`/`size` when
  /// OpenRange rounded the offset down to the mapping granularity.
  void* view_base{nullptr};
  std::size_t view_size{0};
#if defined(_WIN32)
  void* file{nullptr};
  void* section{nullptr};
#endif

  /// Maps `path`; on failure leaves the object untouched. Files of size zero
  /// fail, matching the loaders' fstat contract.
  bool Open(const std::filesystem::path& path) noexcept;
  /// Maps the byte range [offset, offset + length) of `path`. `data` points
  /// at the requested offset, which need not be page-aligned. Zero `length`
  /// fails; on failure the object is left untouched.
  bool OpenRange(const std::filesystem::path& path, std::uint64_t offset,
                 std::size_t length) noexcept;
  void Close() noexcept;
};

/// OS page size, never zero.
std::size_t SystemPageSize() noexcept;

/// Positional-read file handle: int fd on POSIX, HANDLE on Windows.
#if defined(_WIN32)
using PlatformFile = void*;
#else
using PlatformFile = int;
#endif

/// Opens `path` read-only; check FileValid() before use.
PlatformFile OpenReadFile(const std::filesystem::path& path) noexcept;
bool FileValid(PlatformFile file) noexcept;
void CloseFile(PlatformFile file) noexcept;
/// Reads up to `bytes` at `offset`; returns bytes read, 0 on failure/EOF.
std::size_t ReadFileAt(PlatformFile file, void* output, std::size_t bytes,
                       std::uint64_t offset) noexcept;
/// Total file size, or nullopt when it cannot be determined.
std::optional<std::uint64_t> FileSizeBytes(PlatformFile file) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_MAPPED_FILE_HPP_
