// Restoring a session from KV chunks means one copy per (chunk, attention
// layer, K/V) instead of one large copy. This measures whether many smaller
// device-to-device copies, issued asynchronously on one stream, keep the
// bandwidth of a single copy. Total bytes are the same in every case (8 GiB).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define HIP_OK(call)                                                   \
  do {                                                                 \
    hipError_t e = (call);                                             \
    if (e != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,           \
                   hipGetErrorString(e));                              \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

int main() {
  const std::size_t total = 8ull << 30;
  char* src = nullptr;
  char* dst = nullptr;
  HIP_OK(hipMalloc(&src, total));
  HIP_OK(hipMalloc(&dst, total));
  HIP_OK(hipMemset(src, 1, total));
  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));
  // Piece sizes: a 2,048-token chunk split per attention layer and K/V is
  // roughly 1-4 MiB for these models; smaller pieces show the overhead curve.
  const std::size_t pieces[] = {total, 64ull << 20, 4ull << 20, 1ull << 20,
                                256ull << 10, 64ull << 10};
  std::printf("piece_kib,copies,best_ms,gb_per_s\n");
  for (std::size_t piece : pieces) {
    const std::size_t copies = total / piece;
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
      HIP_OK(hipDeviceSynchronize());
      auto t0 = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < copies; ++i) {
        // Scatter destinations like contiguous session KV per layer, and
        // read sources from separately allocated chunks (here: offsets).
        const std::size_t s = (i * 7919 % copies) * piece;
        HIP_OK(hipMemcpyAsync(dst + i * piece, src + s, piece,
                              hipMemcpyDeviceToDevice, stream));
      }
      HIP_OK(hipStreamSynchronize(stream));
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();
      if (ms < best) best = ms;
    }
    std::printf("%zu,%zu,%.2f,%.1f\n", piece >> 10, copies, best,
                total / (best / 1e3) / 1e9);
    std::fflush(stdout);
  }
  return 0;
}
