#ifndef GUFO_CLI_SERVE_DEVICE_FAILURE_HPP_
#define GUFO_CLI_SERVE_DEVICE_FAILURE_HPP_

#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <string_view>

namespace gufo::cli {

/// Avoid model destruction and logging locks after a confirmed device loss.
[[noreturn]] inline void ExitAfterDeviceLoss(std::string_view reason) noexcept {
  static constexpr std::string_view prefix =
      "[ERROR] [server] event=device_lost remedy=restart exit_status=75 "
      "reason=";
  char line[1024];
  std::size_t size = 0;
  for (const char c : prefix)
    line[size++] = c;
  for (const unsigned char c : reason) {
    if (size == sizeof(line) - 1)
      break;
    line[size++] = c < 0x20 || c == 0x7f ? ' ' : static_cast<char>(c);
  }
  line[size++] = '\n';
  (void)::write(STDERR_FILENO, line, size);
  std::_Exit(75);
}

}  // namespace gufo::cli

#endif  // GUFO_CLI_SERVE_DEVICE_FAILURE_HPP_
