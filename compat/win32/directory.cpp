#include <dirent.h>
#include <io.h>
#include <windows.h>

#include <cerrno>
#include <memory>
#include <string>

struct gufo_directory {
  int fd{-1};
  HANDLE search{INVALID_HANDLE_VALUE};
  WIN32_FIND_DATAW data{};
  bool first{true};
  bool empty{false};
  dirent entry{};
};

extern "C" DIR* fdopendir(int fd) {
  const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(handle, &info)) {
    errno = EBADF;
    return nullptr;
  }
  if (!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
    errno = ENOTDIR;
    return nullptr;
  }
  wchar_t path[32768];
  const DWORD count =
      GetFinalPathNameByHandleW(handle, path, 32768, FILE_NAME_NORMALIZED);
  if (!count || count >= 32768) {
    errno = EIO;
    return nullptr;
  }
  auto directory = std::make_unique<gufo_directory>();
  const auto pattern = std::wstring(path, count) + L"\\*";
  directory->search = FindFirstFileW(pattern.c_str(), &directory->data);
  if (directory->search == INVALID_HANDLE_VALUE) {
    if (GetLastError() != ERROR_FILE_NOT_FOUND) {
      errno = EIO;
      return nullptr;
    }
    directory->empty = true;
  }
  directory->fd = fd;
  return directory.release();
}

extern "C" dirent* readdir(DIR* directory) {
  if (directory->empty)
    return nullptr;
  if (!directory->first &&
      !FindNextFileW(directory->search, &directory->data)) {
    if (GetLastError() != ERROR_NO_MORE_FILES)
      errno = EIO;
    return nullptr;
  }
  directory->first = false;
  if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                           directory->data.cFileName, -1,
                           directory->entry.d_name,
                           sizeof(directory->entry.d_name), nullptr, nullptr)) {
    errno = EILSEQ;
    return nullptr;
  }
  return &directory->entry;
}

extern "C" int closedir(DIR* directory) {
  if (directory->search != INVALID_HANDLE_VALUE)
    FindClose(directory->search);
  const int result = close(directory->fd);
  delete directory;
  return result;
}
