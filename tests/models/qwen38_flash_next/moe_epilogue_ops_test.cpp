#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/moe_pending_state.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/streamed_moe_policy.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

// The model's MoE combine geometry: top-10 slots over a 2,560-wide hidden
// state, router row stride num_experts + 1.
constexpr std::uint32_t kTokens = 37;
constexpr std::uint32_t kSlots = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kGateStride = 513;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale, float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = offset +
            scale *
                static_cast<float>(
                    static_cast<int>(NextRandom(&seed) & 0xFFFFU) - 32768) /
                32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

std::vector<__half> MakeHalfValues(std::size_t count, std::uint32_t seed,
                                   float scale) {
  const auto values = MakeValues(count, seed, scale);
  std::vector<__half> half(count);
  std::transform(values.begin(), values.end(), half.begin(),
                 [](float value) { return __float2half(value); });
  return half;
}

void RunDeferredWindowEpilogue() {
  constexpr std::uint32_t window_rows = 11;
  constexpr std::size_t expert_count =
      static_cast<std::size_t>(kTokens) * kSlots * kHidden;
  constexpr std::size_t output_count =
      static_cast<std::size_t>(kTokens) * kHidden;
  const auto expert_out = MakeHalfValues(expert_count, 0x716A45U, 0.2F);
  const auto weights = MakeValues(static_cast<std::size_t>(kTokens) * kSlots,
                                  0x1BADC0DEU, 0.5F, 0.5F);
  const auto shared = MakeValues(output_count, 0x2D17U, 0.3F);
  const auto gate = MakeValues(static_cast<std::size_t>(kTokens) * kGateStride,
                               0x329ABCU, 2.0F);

  HipBuffer<__half> d_expert(expert_count);
  HipBuffer<__half> d_accumulated(expert_count);
  HipBuffer<float> d_weights(weights.size());
  HipBuffer<float> d_shared(output_count);
  HipBuffer<float> d_gate(gate.size());
  HipBuffer<float> d_single(output_count);
  HipBuffer<float> d_windowed(output_count);
  CheckHip(hipMemcpy(d_expert.get(), expert_out.data(), d_expert.bytes(),
                     hipMemcpyHostToDevice),
           "upload F16 expert rows");
  Upload(&d_weights, weights);
  Upload(&d_shared, shared);
  Upload(&d_gate, gate);

  for (std::uint32_t first = 0; first < kTokens; first += window_rows) {
    const std::uint32_t count = std::min(window_rows, kTokens - first);
    const std::size_t expert_offset =
        static_cast<std::size_t>(first) * kSlots * kHidden;
    const std::size_t output_offset = static_cast<std::size_t>(first) * kHidden;
    q::MoeEpilogueVec4F16(
        d_expert.get() + expert_offset,
        d_weights.get() + static_cast<std::size_t>(first) * kSlots,
        d_shared.get() + output_offset,
        d_gate.get() + static_cast<std::size_t>(first) * kGateStride,
        kGateStride, d_windowed.get() + output_offset, count, kSlots, kHidden,
        nullptr);
    CheckHip(
        hipMemcpy(
            d_accumulated.get() + expert_offset, d_expert.get() + expert_offset,
            static_cast<std::size_t>(count) * kSlots * kHidden * sizeof(__half),
            hipMemcpyDeviceToDevice),
        "accumulate streamed expert window");
  }
  q::MoeEpilogueVec4F16(d_accumulated.get(), d_weights.get(), d_shared.get(),
                        d_gate.get(), kGateStride, d_single.get(), kTokens,
                        kSlots, kHidden, nullptr);
  CheckHip(hipDeviceSynchronize(), "deferred F16 epilogue synchronization");
  const auto single = Download(&d_single, output_count);
  const auto windowed = Download(&d_windowed, output_count);
  if (std::memcmp(single.data(), windowed.data(),
                  output_count * sizeof(float)) != 0)
    throw std::runtime_error(
        "one full-chunk F16 epilogue differs from windowed epilogues");

  gufo::models::qwen38_flash_next::rocm::MoePendingState pending;
  pending.Reset();
  pending.SetForOutput(d_single.get(), d_single.get());
  if (!pending.pending() || !pending.Consume() || pending.pending())
    throw std::runtime_error(
        "deferred MoE epilogue state was not preserved for Combine");
}

void CheckDeskRoomSlotBoundary() {
  using gufo::models::qwen38_flash_next::rocm::DeskRoomHasSlotHeadroom;
  constexpr std::uint32_t used = 10;
  if (DeskRoomHasSlotHeadroom(used, used) ||
      DeskRoomHasSlotHeadroom(2 * used, used) ||
      !DeskRoomHasSlotHeadroom(2 * used + 1, used)) {
    throw std::runtime_error("Desk/Room slot boundary changed");
  }
}

void CheckRoutedF16FormatMatrix() {
  using gufo::core::GgmlType;
  using gufo::models::qwen38_flash_next::rocm::UsesRoutedF16ExpertFormats;
  for (const auto gate : {GgmlType::kQ4_K, GgmlType::kQ5_K}) {
    for (const auto down : {GgmlType::kQ5_1, GgmlType::kQ8_0}) {
      if (!UsesRoutedF16ExpertFormats(gate, gate, down, 2560, 640))
        throw std::runtime_error("supported routed-F16 format was rejected");
    }
  }
  if (UsesRoutedF16ExpertFormats(GgmlType::kQ4_K, GgmlType::kQ5_K,
                                 GgmlType::kQ5_1, 2560, 640) ||
      UsesRoutedF16ExpertFormats(GgmlType::kQ4_K, GgmlType::kQ4_K,
                                 GgmlType::kQ4_K, 2560, 640) ||
      UsesRoutedF16ExpertFormats(GgmlType::kQ4_K, GgmlType::kQ4_K,
                                 GgmlType::kQ5_1, 2559, 640)) {
    throw std::runtime_error("unsupported routed-F16 format was accepted");
  }
}

}  // namespace

int main() {
  try {
    CheckDeskRoomSlotBoundary();
    CheckRoutedF16FormatMatrix();
    RunDeferredWindowEpilogue();
    constexpr std::size_t kExpert =
        static_cast<std::size_t>(kTokens) * kSlots * kHidden;
    constexpr std::size_t kOut = static_cast<std::size_t>(kTokens) * kHidden;
    const auto expert_out = MakeValues(kExpert, 0x1234ABCDU, 1.0F);
    const auto weights = MakeValues(static_cast<std::size_t>(kTokens) * kSlots,
                                    0xBADC0FFEU, 0.5F, 0.5F);
    const auto shared = MakeValues(kOut, 0xDEADBEEFU, 1.0F);
    const auto gate = MakeValues(
        static_cast<std::size_t>(kTokens) * kGateStride, 0xC0FFEE11U, 3.0F);
    HipBuffer<float> d_expert(kExpert);
    HipBuffer<float> d_weights(weights.size());
    HipBuffer<float> d_shared(kOut);
    HipBuffer<float> d_gate(gate.size());
    HipBuffer<float> d_ref(kOut);
    HipBuffer<float> d_vec(kOut);
    Upload(&d_expert, expert_out);
    Upload(&d_weights, weights);
    Upload(&d_shared, shared);
    Upload(&d_gate, gate);
    q::MoeEpilogue(d_expert.get(), d_weights.get(), d_shared.get(),
                   d_gate.get(), kGateStride, d_ref.get(), kTokens, kSlots,
                   kHidden, nullptr);
    q::MoeEpilogueVec4(d_expert.get(), d_weights.get(), d_shared.get(),
                       d_gate.get(), kGateStride, d_vec.get(), kTokens, kSlots,
                       kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "MoE epilogue synchronization");
    const auto ref = Download(&d_ref, kOut);
    const auto vec = Download(&d_vec, kOut);
    double worst = 0.0;
    for (std::size_t i = 0; i < kOut; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(ref[i] - vec[i])));
    }
    std::cout << "MoE epilogue vec4 worst absolute error " << worst << '\n';
    return worst == 0.0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
