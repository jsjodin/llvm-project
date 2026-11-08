// The floor under array messages from a GPU: omp_target_memcpy alone, from
// main, with no kernel running and no message, over the same sizes as
// arraymessages.m (README.md). Sizes include the array's size word.
//
// Usage: memcpy-floor MIB
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(int argc, char **argv) {
  const std::uint64_t mib = argc > 1 ? std::atoll(argv[1]) : 64;
  const int device = omp_get_default_device();
  const int host = omp_get_initial_device();
  for (std::uint64_t words = 1; words <= (1u << 20); words *= 4) {
    const std::uint64_t bytes = 8 * (words + 1);
    std::uint64_t rounds = mib * (1u << 20) / (8 * words);
    rounds = rounds < 2 ? 2 : rounds > 20000 ? 20000 : rounds;
    void *on_host = std::calloc(words + 1, 8);
    void *on_device = omp_target_alloc(bytes, device);
    if (!on_host || !on_device) {
      std::fprintf(stderr, "out of memory\n");
      return 1;
    }
    for (int direction = 0; direction < 2; ++direction) {
      const double start = omp_get_wtime();
      for (std::uint64_t i = 0; i < rounds; ++i) {
        const int error = direction == 0
            ? omp_target_memcpy(on_host, on_device, bytes, 0, 0, host, device)
            : omp_target_memcpy(on_device, on_host, bytes, 0, 0, device, host);
        if (error) {
          std::fprintf(stderr, "omp_target_memcpy failed\n");
          return 1;
        }
      }
      const double seconds = omp_get_wtime() - start;
      std::printf(
          "%-10s %-5s %8lu words %6lu messages %6s right %10.1f MB/s %12.2f us each\n",
          "memcpy", direction == 0 ? "to" : "from", (unsigned long)words,
          (unsigned long)rounds, "-", 8.0 * words * rounds / seconds / 1e6,
          seconds / rounds * 1e6);
      std::fflush(stdout);
    }
    omp_target_free(on_device, device);
    std::free(on_host);
  }
}
