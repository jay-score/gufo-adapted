// Copy bandwidth for the cache operations of the hybrid design on gfx1151:
// device-to-device (capture or restore inside GPU memory), device-to-host
// and host-to-device with pinned and pageable host memory (spill, persist,
// restore from RAM). Sizes cover a chunk (56 MB, 128 MiB), fixed state
// (114, 232 MiB) and full snapshots (1-10 GB).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(call)                                                   \
  do {                                                                 \
    hipError_t e = (call);                                             \
    if (e != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,           \
                   hipGetErrorString(e));                              \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

template <typename F>
double BestSeconds(F&& f, int reps) {
  double best = 1e30;
  for (int i = 0; i < reps; ++i) {
    HIP_OK(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    f();
    HIP_OK(hipDeviceSynchronize());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (s < best) best = s;
  }
  return best;
}

int main() {
  const std::size_t sizes[] = {56ull << 20,  114ull << 20, 128ull << 20,
                               232ull << 20, 1ull << 30,   4ull << 30,
                               10ull << 30};
  const std::size_t max_size = 10ull << 30;
  void* d_src = nullptr;
  void* d_dst = nullptr;
  void* h_pinned = nullptr;
  HIP_OK(hipMalloc(&d_src, max_size));
  HIP_OK(hipMalloc(&d_dst, max_size));
  HIP_OK(hipHostMalloc(&h_pinned, max_size, hipHostMallocDefault));
  char* h_pageable = static_cast<char*>(std::malloc(max_size));
  std::memset(h_pageable, 1, max_size);
  std::memset(h_pinned, 1, max_size);
  HIP_OK(hipMemset(d_src, 1, max_size));
  std::printf("size_mib,kind,best_ms,gb_per_s\n");
  for (std::size_t bytes : sizes) {
    const int reps = bytes >= (4ull << 30) ? 3 : 8;
    struct Case {
      const char* name;
      void* dst;
      const void* src;
      hipMemcpyKind kind;
    } cases[] = {
        {"d2d", d_dst, d_src, hipMemcpyDeviceToDevice},
        {"d2h_pinned", h_pinned, d_src, hipMemcpyDeviceToHost},
        {"h2d_pinned", d_dst, h_pinned, hipMemcpyHostToDevice},
        {"d2h_pageable", h_pageable, d_src, hipMemcpyDeviceToHost},
        {"h2d_pageable", d_dst, h_pageable, hipMemcpyHostToDevice},
    };
    for (const auto& c : cases) {
      double s = BestSeconds([&] { HIP_OK(hipMemcpy(c.dst, c.src, bytes, c.kind)); }, reps);
      std::printf("%zu,%s,%.3f,%.1f\n", bytes >> 20, c.name, s * 1e3, bytes / s / 1e9);
      std::fflush(stdout);
    }
  }
  return 0;
}
