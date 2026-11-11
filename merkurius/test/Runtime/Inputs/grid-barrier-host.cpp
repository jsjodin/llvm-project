// The host of grid-barrier-gpu.test: loads the kernel's image with liboffload
// and launches it cooperatively, with as many teams of the given threads as
// can be resident at once, then prints how many slots were wrong.
#include "../../../runtime/liboffload.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

ol_device_handle_t gpu{nullptr};

bool FindGpu(ol_device_handle_t device, void *) {
  ol_platform_handle_t platform;
  std::uint32_t backend{0};
  if (olGetDeviceInfo(device, kOlDeviceInfoPlatform, sizeof platform,
          &platform) == nullptr &&
      olGetPlatformInfo(platform, kOlPlatformInfoBackend, sizeof backend,
          &backend) == nullptr &&
      backend == kOlPlatformBackendAmdgpu && !gpu)
    gpu = device;
  return true;
}

void Check(ol_result_t result, const char *what) {
  if (result) {
    std::printf("%s: %s\n", what, result->details);
    std::exit(1);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::printf("usage: %s image threads rounds\n", argv[0]);
    return 1;
  }
  std::uint32_t threads{static_cast<std::uint32_t>(std::atoi(argv[2]))};
  std::uint32_t rounds{static_cast<std::uint32_t>(std::atoi(argv[3]))};
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
  Check(olIterateDevices(FindGpu, nullptr), "olIterateDevices");
  if (!gpu) {
    std::printf("no GPU\n");
    return 1;
  }
  ol_context_handle_t context;
  ol_program_handle_t program;
  ol_symbol_handle_t kernel;
  ol_queue_handle_t queue;
  Check(olCreateContext(1, &gpu, &context), "olCreateContext");
  Check(olCreateProgram(context, gpu, image.data(), image.size(), &program),
      "olCreateProgram");
  Check(olGetSymbol(program, "grid_barrier_test", kOlSymbolKindKernel, &kernel),
      "olGetSymbol");
  Check(olCreateQueue(context, gpu, &queue), "olCreateQueue");

  ol_kernel_launch_size_args_t size{1, {1, 1, 1}, {threads, 1, 1}, 0};
  std::uint32_t teams{0};
  Check(olGetKernelMaxCooperativeGroupCount(gpu, kernel, &size, &teams),
      "olGetKernelMaxCooperativeGroupCount");
  size.numGroups.x = teams;

  // The barrier's counts, the slots and the error count, zeroed.
  std::size_t slotCount{static_cast<std::size_t>(teams) * threads};
  std::size_t bytes{16 + 8 * slotCount + 8};
  void *memory;
  Check(olMemAlloc(context, gpu, kOlAllocTypeDevice, bytes, &memory),
      "olMemAlloc");
  std::vector<char> zeros(bytes, 0);
  Check(olMemcpy(queue, memory, gpu, zeros.data(), nullptr, bytes), "olMemcpy");
  char *base{static_cast<char *>(memory)};
  void *barrier{base};
  void *slots{base + 16};
  void *errors{base + 16 + 8 * slotCount};

  bool cooperative{true};
  ol_kernel_launch_prop_t properties[]{
      {kOlKernelLaunchPropTypeIsCooperative, &cooperative},
      {kOlKernelLaunchPropTypeNone, nullptr}};
  void *arguments[]{&barrier, &slots, &rounds, &errors};
  std::size_t sizes[]{
      sizeof barrier, sizeof slots, sizeof rounds, sizeof errors};
  Check(olLaunchKernel(
            queue, gpu, kernel, &size, properties, 4, arguments, sizes),
      "olLaunchKernel");
  std::uint32_t wrong{0};
  Check(
      olMemcpy(queue, &wrong, nullptr, errors, gpu, sizeof wrong), "olMemcpy");
  Check(olSyncQueue(queue), "olSyncQueue");
  std::printf("%u threads, %u rounds: %u wrong\n", threads, rounds, wrong);
  return 0;
}
