// The host of fine-memory-gpu.test: fine-grained device memory through
// liboffload (OL_ALLOC_TYPE_DEVICE_FINE_GRAINED, this fork's). On the first
// GPU, a kernel's atomics on a word of it, which is then freed. With two GPUs
// visible, a ping-pong of `rounds` round trips between kernels on the two,
// each polling a word of its own GPU's memory that the other writes: with
// fine-grained words, and then with ordinary ones, which a peer's write
// never reaches.
#include "../../../runtime/liboffload.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::vector<ol_device_handle_t> gpus;

bool FindGpus(ol_device_handle_t device, void *) {
  ol_platform_handle_t platform;
  std::uint32_t backend{0};
  if (olGetDeviceInfo(device, kOlDeviceInfoPlatform, sizeof platform,
          &platform) == nullptr &&
      olGetPlatformInfo(platform, kOlPlatformInfoBackend, sizeof backend,
          &backend) == nullptr &&
      backend == kOlPlatformBackendAmdgpu)
    gpus.push_back(device);
  return true;
}

void Check(ol_result_t result, const char *what) {
  if (result) {
    std::printf("%s: %s\n", what, result->details);
    std::exit(1);
  }
}

struct Gpu {
  ol_device_handle_t device;
  ol_program_handle_t program;
  ol_queue_handle_t queue;
};

ol_symbol_handle_t Kernel(const Gpu &gpu, const char *name) {
  ol_symbol_handle_t kernel;
  Check(olGetSymbol(gpu.program, name, kOlSymbolKindKernel, &kernel),
      "olGetSymbol");
  return kernel;
}

void *Allocate(ol_context_handle_t context, const Gpu &gpu, bool fine,
    std::size_t bytes) {
  void *memory;
  Check(olMemAlloc(context, gpu.device,
            fine ? kOlAllocTypeDeviceFineGrained : kOlAllocTypeDevice, bytes,
            &memory),
      "olMemAlloc");
  std::vector<char> zeros(bytes, 0);
  Check(olMemcpy(gpu.queue, memory, gpu.device, zeros.data(), nullptr, bytes),
      "olMemcpy");
  Check(olSyncQueue(gpu.queue), "olSyncQueue");
  return memory;
}

// Rounds of ping-pong between gpus a and b, with the words fine-grained or
// not: whether either side got stuck.
bool PingPong(ol_context_handle_t context, Gpu *two, bool fine,
    std::uint32_t rounds) {
  void *word[2];
  for (int i{0}; i < 2; ++i)
    word[i] = Allocate(context, two[i], fine, 64);
  void *stuck;
  Check(olMemAllocHost(context, two[0].device, 64, &stuck), "olMemAllocHost");
  *static_cast<std::uint32_t *>(stuck) = 0;
  for (int i{1}; i >= 0; --i) {
    std::uint32_t first{i == 0 ? 1u : 0u};
    void *arguments[]{&word[i], &word[1 - i], &rounds, &first, &stuck};
    std::size_t sizes[]{sizeof(void *), sizeof(void *), sizeof rounds,
        sizeof first, sizeof(void *)};
    ol_kernel_launch_size_args_t size{1, {1, 1, 1}, {64, 1, 1}, 0};
    Check(olLaunchKernel(two[i].queue, two[i].device,
              Kernel(two[i], "fine_ping"), &size, nullptr, 5, arguments,
              sizes),
        "olLaunchKernel");
  }
  for (int i{0}; i < 2; ++i)
    Check(olSyncQueue(two[i].queue), "olSyncQueue");
  bool wasStuck{*static_cast<std::uint32_t *>(stuck) != 0};
  for (int i{0}; i < 2; ++i)
    Check(olMemFree(context, word[i]), "olMemFree");
  Check(olMemFree(context, stuck), "olMemFree");
  return wasStuck;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::printf("usage: %s image rounds\n", argv[0]);
    return 1;
  }
  std::uint32_t rounds{static_cast<std::uint32_t>(std::atoi(argv[2]))};
  std::FILE *file{std::fopen(argv[1], "rb")};
  if (!file) {
    std::printf("cannot open %s\n", argv[1]);
    return 1;
  }
  std::vector<char> image;
  for (int c; (c = std::fgetc(file)) != EOF;)
    image.push_back(static_cast<char>(c));
  std::fclose(file);

  Check(olInit(nullptr), "olInit");
  Check(olIterateDevices(FindGpus, nullptr), "olIterateDevices");
  if (gpus.empty()) {
    std::printf("no GPU\n");
    return 1;
  }
  std::size_t count{gpus.size() < 2 ? gpus.size() : 2};
  ol_context_handle_t context;
  Check(olCreateContext(count, gpus.data(), &context), "olCreateContext");
  Gpu two[2];
  for (std::size_t i{0}; i < count; ++i) {
    two[i].device = gpus[i];
    Check(olCreateProgram(context, gpus[i], image.data(), image.size(),
              &two[i].program),
        "olCreateProgram");
    Check(olCreateQueue(context, gpus[i], &two[i].queue), "olCreateQueue");
  }

  // A kernel's atomics on a fine-grained word of the first GPU.
  void *counter{Allocate(context, two[0], true, 64)};
  std::uint32_t threads{256}, teams{8};
  void *arguments[]{&counter, &rounds};
  std::size_t sizes[]{sizeof counter, sizeof rounds};
  ol_kernel_launch_size_args_t size{1, {teams, 1, 1}, {threads, 1, 1}, 0};
  Check(olLaunchKernel(two[0].queue, two[0].device, Kernel(two[0], "fine_count"),
            &size, nullptr, 2, arguments, sizes),
      "olLaunchKernel");
  std::uint64_t total{0};
  Check(olMemcpy(two[0].queue, &total, nullptr, counter, two[0].device,
            sizeof total),
      "olMemcpy");
  Check(olSyncQueue(two[0].queue), "olSyncQueue");
  Check(olMemFree(context, counter), "olMemFree");
  std::printf("fine-grained counter: %llu of %llu\n",
      static_cast<unsigned long long>(total),
      static_cast<unsigned long long>(threads) * teams * rounds);

  if (count == 2) {
    std::printf("2 GPUs, %u round trips, fine-grained words: %s\n", rounds,
        PingPong(context, two, true, rounds) ? "stuck" : "done");
    std::printf("2 GPUs, %u round trips, coarse-grained words: %s\n", rounds,
        PingPong(context, two, false, rounds) ? "stuck" : "done");
  }
  return 0;
}
