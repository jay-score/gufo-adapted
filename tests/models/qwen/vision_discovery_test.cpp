#include <unistd.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen/vision/encoder.hpp"

namespace {
using gufo::models::qwen::vision::Encoder;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Files {
  std::filesystem::path directory;
  Files() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "gufo-vision-XXXXXX")
            .string();
    Require(::mkdtemp(pattern.data()) != nullptr,
            "cannot create test directory");
    directory = pattern;
  }
  ~Files() { std::filesystem::remove_all(directory); }
};

void RequireRejected(const std::filesystem::path& target,
                     const std::filesystem::path& sidecar) {
  try {
    (void)Encoder::Open(target, sidecar, 2560);
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("explicit invalid sidecar was accepted");
}
}  // namespace

int main() {
  try {
    Files files;
    const auto model_directory = files.directory / "quant";
    std::filesystem::create_directory(model_directory);
    const auto target = model_directory / "model.gguf";
    Require(!Encoder::Open(target, {}, 2560), "unexpected vision sidecar");

    // A valid empty GGUF has no matching vision metadata. It reproduces an
    // incompatible projector in the parent directory without model weights.
    const auto parent_sidecar = files.directory / "mmproj-BF16.gguf";
    const std::array<std::uint32_t, 8> header{0x46554747, 3};
    {
      std::ofstream out(parent_sidecar, std::ios::binary);
      out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
      Require(out.good(), "cannot write test sidecar");
    }
    RequireRejected(target, parent_sidecar);
    Require(!Encoder::Open(target, {}, 2560),
            "incompatible parent sidecar prevented text-only loading");

    const auto adjacent_sidecar = model_directory / "mmproj-BF16.gguf";
    std::filesystem::copy_file(parent_sidecar, adjacent_sidecar);
    RequireRejected(target, adjacent_sidecar);
    Require(!Encoder::Open(target, {}, 2560),
            "incompatible adjacent sidecar prevented text-only loading");
    RequireRejected(target, model_directory / "missing.gguf");
    std::cout << "PASS: optional discovery and explicit sidecar validation\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
