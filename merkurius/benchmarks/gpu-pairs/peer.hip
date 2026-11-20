// Peer probes between two GPUs of this machine, by HIP device numbers
// (ROCR_VISIBLE_DEVICES order): what the hardware gives for direct sends
// between GPUs, with no Merkurius in it (README.md; docs/INPROGRESS.md,
// Direct sends between the GPUs of a node). Built and run as
//
//   clang -x hip -O2 --offload-arch=gfx90a --rocm-path=/opt/rocm peer.hip \
//     -L/opt/rocm/lib -lamdhip64 -lstdc++ -Wl,-rpath,/opt/rocm/lib -o peer
//   utils/gpu/with-gpus.py --gpus A,B -- ./peer bw 0 1
//
// Modes:
//   peer info               every device's PCI bus, and each pair's link
//   peer bw A B             A -> B: hipMemcpyPeerAsync, A's kernel storing
//                           into B's memory, B's kernel loading from A's
//   peer lat A B [rounds]   a flag ping-pong between kernels on A and B,
//                           each polling its own memory, written by the other
//   peer coh A B [words] [rounds] [acquire] [fine]
//                           A's kernel writes data and a flag into B's memory
//                           while B's kernel holds that data in its caches;
//                           B checks each word after an acquire at system
//                           scope (acquire 1), at device scope (2) or none (0);
//                           the data in B's coarse-grained memory, or fine
//                           (fine 1); the flags are always fine-grained
//   peer pull A B [words] [rounds]
//                           experiment 1: A writes a buffer in its OWN
//                           coarse-grained memory, the same buffer every
//                           round, releases and sets a fine-grained flag in
//                           B; B, whose caches hold the old contents, loads
//                           it from A after an acquire; every pairing of A's
//                           release (system, device) and B's acquire (system,
//                           device, none)
//   peer pullbw A B         experiment 2: B copies from A's coarse-grained
//                           memory into its own, by one lane, one wave, a
//                           team of 4 waves, and 8, 32 and 104 teams, 4 KiB to
//                           8 MiB, each copy after a system-scope acquire;
//                           device-timed; and the same local copies on B
//   peer poll A B           experiment 3: the cost of a poll by one wave on B
//                           of a fine-grained word in B's memory (relaxed and
//                           acquire at system scope) and of a word in host
//                           memory, alone, and what each does to a kernel
//                           that reads 4 MiB of B's memory over and over
//   peer push A B           experiment 4: ping-pong between a wave on A and
//                           one on B carrying 8 B to 64 KiB each way, pushed
//                           (stored into the receiver's fine-grained memory)
//                           or pulled (stored in the sender's own memory and
//                           loaded by the receiver)
//   peer slot A B [messages] [words]
//                           experiment 5: the device runtime's inline peer
//                           send (PutPeer, TakePeer) alone. A wave on A puts
//                           messages of `words` words into slots in B's
//                           fine-grained memory, and publishes each by a
//                           store of the slot's sequence: relaxed with no
//                           wait (a control, to show the check can fail),
//                           a release at device scope, and at system scope
//                           (buffer_wbl2), each with A's L2 clean, with 64
//                           lines of A's own memory dirtied before each
//                           message, and with other teams on A writing 32
//                           MiB of its memory over and over meanwhile, as a
//                           place's other processes would. A
//                           wave on B takes each after an acquire at device
//                           scope and checks every word. Device-timed on A.
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, \
               hipGetErrorString(e)); std::exit(1); } } while (0)

static double Seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void Peers(int a, int b) {
  int can = 0;
  CHECK(hipDeviceCanAccessPeer(&can, a, b));
  if (!can) { std::fprintf(stderr, "%d cannot access %d\n", a, b); std::exit(1); }
  CHECK(hipSetDevice(a));
  hipError_t e = hipDeviceEnablePeerAccess(b, 0);
  if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CHECK(e);
  (void)hipGetLastError();
  CHECK(hipSetDevice(b));
  e = hipDeviceEnablePeerAccess(a, 0);
  if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CHECK(e);
  (void)hipGetLastError();
}

static void Info() {
  int n = 0;
  CHECK(hipGetDeviceCount(&n));
  for (int i = 0; i < n; ++i) {
    char bus[32];
    CHECK(hipDeviceGetPCIBusId(bus, sizeof bus, i));
    std::printf("device %d bus %s\n", i, bus);
  }
  std::printf("link type/hops (type 4 = XGMI):\n     ");
  for (int j = 0; j < n; ++j) std::printf("   %d  ", j);
  std::printf("\n");
  for (int i = 0; i < n; ++i) {
    std::printf("  %d  ", i);
    for (int j = 0; j < n; ++j) {
      if (i == j) { std::printf("   -  "); continue; }
      uint32_t type = 0, hops = 0;
      (void)hipExtGetLinkTypeAndHopCount(i, j, &type, &hops);
      std::printf(" %u/%u  ", type, hops);
    }
    std::printf("\n");
  }
}

__global__ void Store(uint4 *to, size_t n, uint32_t tag) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    to[i] = make_uint4(tag, (uint32_t)i, tag, (uint32_t)(i >> 32));
}

__global__ void Load(const uint4 *from, size_t n, uint4 *sink) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint4 v = from[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (acc.x == 0x12345678u && acc.y == 0x9abcdefu) sink[0] = acc;
}

static void Bandwidth(int a, int b) {
  Peers(a, b);
  const size_t most = size_t{256} << 20;
  void *onA, *onB, *sink;
  CHECK(hipSetDevice(a)); CHECK(hipMalloc(&onA, most));
  CHECK(hipMemset(onA, 1, most));
  CHECK(hipSetDevice(b)); CHECK(hipMalloc(&onB, most)); CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(onB, 2, most));
  hipStream_t sa, sb;
  CHECK(hipSetDevice(a)); CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
  CHECK(hipSetDevice(b)); CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
  int units = 0;
  CHECK(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, a));
  std::printf("%d -> %d, GB/s       memcpy   A stores   B loads\n", a, b);
  for (size_t bytes = size_t{8} << 10; bytes <= most; bytes *= 4) {
    int reps = bytes < (size_t{1} << 20) ? 2000 : bytes < (size_t{16} << 20) ? 200 : 20;
    double t[3];
    for (int way = 0; way < 3; ++way) {
      auto once = [&]() {
        if (way == 0) {
          CHECK(hipMemcpyPeerAsync(onB, b, onA, a, bytes, sa));
        } else if (way == 1) {
          CHECK(hipSetDevice(a));
          size_t n = bytes / sizeof(uint4);
          int blocks = (int)((n + 255) / 256); if (blocks > units * 8) blocks = units * 8;
          hipLaunchKernelGGL(Store, dim3(blocks), dim3(256), 0, sa, (uint4 *)onB, n, 7u);
        } else {
          CHECK(hipSetDevice(b));
          size_t n = bytes / sizeof(uint4);
          int blocks = (int)((n + 255) / 256); if (blocks > units * 8) blocks = units * 8;
          hipLaunchKernelGGL(Load, dim3(blocks), dim3(256), 0, sb, (const uint4 *)onA, n, (uint4 *)sink);
        }
      };
      hipStream_t s = way == 2 ? sb : sa;
      for (int w = 0; w < 3; ++w) once();
      CHECK(hipStreamSynchronize(s));
      double start = Seconds();
      for (int r = 0; r < reps; ++r) once();
      CHECK(hipStreamSynchronize(s));
      t[way] = (Seconds() - start) / reps;
    }
    std::printf("%10zu bytes  %8.2f  %8.2f  %8.2f   (us each: %.1f %.1f %.1f)\n", bytes,
                bytes / t[0] / 1e9, bytes / t[1] / 1e9, bytes / t[2] / 1e9,
                t[0] * 1e6, t[1] * 1e6, t[2] * 1e6);
  }
}

// Waits for `cond`, giving up after 2^22 polls, a few seconds at a
// microsecond a poll of a peer's memory at system scope: then
// it sets *stuck and leaves the kernel.
#define WAIT(cond) do { long spins_ = 0; while (!(cond)) { \
  if (++spins_ > (1L << 22)) { __scoped_atomic_store_n(stuck, 1L, __ATOMIC_RELAXED, \
      __MEMORY_SCOPE_SYSTEM); return; } } } while (0)

// Plays rounds of ping-pong: `mine` is in this device's memory and written
// by the other kernel, `theirs` in the other's. The first player starts.
__global__ void Player(long *mine, long *theirs, long rounds, int first,
                       long *stuck) {
  if (threadIdx.x != 0) return;
  for (long i = 1; i <= rounds; ++i) {
    if (first) {
      __scoped_atomic_store_n(theirs, i, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      WAIT(__scoped_atomic_load_n(mine, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) == i);
    } else {
      WAIT(__scoped_atomic_load_n(mine, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) == i);
      __scoped_atomic_store_n(theirs, i, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
  }
}

static void Latency(int a, int b, long rounds, bool fine) {
  Peers(a, b);
  long *flagA, *flagB;
  unsigned flags = fine ? hipDeviceMallocFinegrained : hipDeviceMallocDefault;
  CHECK(hipSetDevice(a)); CHECK(hipExtMallocWithFlags((void **)&flagA, 64, flags));
  CHECK(hipMemset(flagA, 0, 64));
  CHECK(hipSetDevice(b)); CHECK(hipExtMallocWithFlags((void **)&flagB, 64, flags));
  CHECK(hipMemset(flagB, 0, 64));
  CHECK(hipDeviceSynchronize());
  CHECK(hipSetDevice(a)); CHECK(hipDeviceSynchronize());
  hipStream_t sa, sb;
  CHECK(hipSetDevice(a)); CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
  CHECK(hipSetDevice(b)); CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
  long *stuck;
  CHECK(hipHostMalloc((void **)&stuck, 64, hipHostMallocCoherent));
  *stuck = 0;
  double start = Seconds();
  CHECK(hipSetDevice(b));
  hipLaunchKernelGGL(Player, dim3(1), dim3(64), 0, sb, flagB, flagA, rounds, 0, stuck);
  CHECK(hipSetDevice(a));
  hipLaunchKernelGGL(Player, dim3(1), dim3(64), 0, sa, flagA, flagB, rounds, 1, stuck);
  CHECK(hipStreamSynchronize(sa));
  CHECK(hipSetDevice(b)); CHECK(hipStreamSynchronize(sb));
  double t = Seconds() - start;
  if (*stuck) {
    std::printf("%d <-> %d, %s flags: STUCK, a flag's write was never seen\n", a, b,
                fine ? "fine-grained" : "coarse-grained");
    return;
  }
  std::printf("%d <-> %d, %s flags: %ld round trips, %.2f us each, %.2f us one way"
              " (launches included)\n", a, b, fine ? "fine-grained" : "coarse-grained",
              rounds, t / rounds * 1e6, t / rounds / 2 * 1e6);
}

// A writes `words` into B's buffer, then sets B's flag; B, which read the
// buffer before (so its caches hold it), waits for the flag and checks.
__global__ void Writer(unsigned long *data, long words, long *flag, long *ack,
                       long rounds, long *stuck) {
  long stride = (long)gridDim.x * blockDim.x;
  long me = (long)blockIdx.x * blockDim.x + threadIdx.x;
  for (long i = 1; i <= rounds; ++i) {
    if (me == 0)
      WAIT(__scoped_atomic_load_n(ack, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) == i - 1);
    __syncthreads();
    for (long w = me; w < words; w += stride)
      data[w] = ((unsigned long)i << 32) | (unsigned long)w;
    __syncthreads();
    if (me == 0)
      __scoped_atomic_store_n(flag, i, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void Checker(unsigned long *data, long words, long *flag, long *ack,
                        long rounds, int acquire, unsigned long long *stale,
                        long *stuck) {
  if (blockIdx.x != 0) return;
  __shared__ unsigned long long sink;
  long me = threadIdx.x;
  unsigned long long bad = 0;
  for (long i = 1; i <= rounds; ++i) {
    // Pull the old contents into this CU's L1 and the L2.
    unsigned long acc = 0;
    for (long w = me; w < words; w += blockDim.x) acc += data[w];
    if (acc == 42) sink = acc;
    if (me == 0) {
      __scoped_atomic_store_n(ack, i - 1, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      WAIT(__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) == i);
    }
    __syncthreads();
    if (acquire == 1)
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    else if (acquire == 2)
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
    for (long w = me; w < words; w += blockDim.x)
      if (data[w] != (((unsigned long)i << 32) | (unsigned long)w)) ++bad;
    __syncthreads();
  }
  atomicAdd(stale, bad);
}

static void Coherence(int a, int b, long words, long rounds, int acquire,
                      bool fineData) {
  Peers(a, b);
  unsigned long *data; long *flag, *ack; unsigned long long *stale; long *stuck;
  CHECK(hipHostMalloc((void **)&stuck, 64, hipHostMallocCoherent));
  *stuck = 0;
  CHECK(hipSetDevice(b));
  CHECK(hipExtMallocWithFlags((void **)&data, words * 8,
      fineData ? hipDeviceMallocFinegrained : hipDeviceMallocDefault));
  CHECK(hipMemset(data, 0, words * 8));
  CHECK(hipExtMallocWithFlags((void **)&flag, 64, hipDeviceMallocFinegrained));
  CHECK(hipMemset(flag, 0, 64));
  CHECK(hipMalloc(&stale, 64)); CHECK(hipMemset(stale, 0, 64));
  CHECK(hipSetDevice(a));
  CHECK(hipExtMallocWithFlags((void **)&ack, 64, hipDeviceMallocFinegrained));
  CHECK(hipMemset(ack, 0xff, 64));
  CHECK(hipDeviceSynchronize());
  CHECK(hipSetDevice(b)); CHECK(hipDeviceSynchronize());
  hipStream_t sa, sb;
  CHECK(hipSetDevice(a)); CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
  CHECK(hipSetDevice(b)); CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
  CHECK(hipSetDevice(b));
  hipLaunchKernelGGL(Checker, dim3(1), dim3(256), 0, sb, data, words, flag, ack,
                     rounds, acquire, stale, stuck);
  CHECK(hipSetDevice(a));
  // One block, so that its barrier orders all of its stores before the flag.
  hipLaunchKernelGGL(Writer, dim3(1), dim3(256), 0, sa, data, words, flag, ack, rounds,
                     stuck);
  CHECK(hipStreamSynchronize(sa));
  CHECK(hipSetDevice(b)); CHECK(hipStreamSynchronize(sb));
  unsigned long long bad = 0;
  CHECK(hipMemcpy(&bad, stale, 8, hipMemcpyDeviceToHost));
  std::printf("%d -> %d, %s data, %ld words, %ld rounds, acquire %s: %llu stale words%s\n",
              a, b, fineData ? "fine" : "coarse", words, rounds, acquire == 1 ? "system" : acquire == 2 ? "device" : "none",
              bad, *stuck ? " (STUCK: a flag's write was never seen)" : "");
}

// ---- Experiment 1: pull coherence with a reused buffer.
// A's buffer is in A's coarse-grained memory; flag in B (fine), ack in A
// (fine). B reads the old contents first, so its L1 and L2 may hold them.
__global__ void PullWriter(unsigned long *data, long words, long *flag, long *ack,
                           long rounds, int release, long *stuck) {
  long me = threadIdx.x;
  for (long i = 1; i <= rounds; ++i) {
    if (me == 0)
      WAIT(__scoped_atomic_load_n(ack, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) == i - 1);
    __syncthreads();
    for (long w = me; w < words; w += blockDim.x)
      data[w] = ((unsigned long)i << 32) | (unsigned long)w;
    __syncthreads();
    if (me == 0) {
      if (release == 1)
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      else
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n(flag, i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }
  }
}

__global__ void PullChecker(const unsigned long *data, long words, long *flag, long *ack,
                            long rounds, int acquire, unsigned long long *stale,
                            long *stuck) {
  __shared__ unsigned long long sink;
  long me = threadIdx.x;
  unsigned long long bad = 0;
  for (long i = 1; i <= rounds; ++i) {
    unsigned long acc = 0;
    for (long w = me; w < words; w += blockDim.x) acc += data[w];
    if (acc == 42) sink = acc;
    __syncthreads();
    if (me == 0) {
      __scoped_atomic_store_n(ack, i - 1, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      WAIT(__scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) == i);
    }
    __syncthreads();
    if (acquire == 1)
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    else if (acquire == 2)
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
    for (long w = me; w < words; w += blockDim.x)
      if (data[w] != (((unsigned long)i << 32) | (unsigned long)w)) ++bad;
    __syncthreads();
  }
  atomicAdd(stale, bad);
}

static void Pull(int a, int b, long words, long rounds) {
  Peers(a, b);
  for (int release = 1; release <= 2; ++release)
    for (int acquire : {1, 2, 0}) {
      unsigned long *data; long *flag, *ack; unsigned long long *stale; long *stuck;
      CHECK(hipHostMalloc((void **)&stuck, 64, hipHostMallocCoherent));
      *stuck = 0;
      CHECK(hipSetDevice(a));
      CHECK(hipMalloc(&data, words * 8)); CHECK(hipMemset(data, 0, words * 8));
      CHECK(hipExtMallocWithFlags((void **)&ack, 64, hipDeviceMallocFinegrained));
      CHECK(hipMemset(ack, 0xff, 64));
      CHECK(hipDeviceSynchronize());
      CHECK(hipSetDevice(b));
      CHECK(hipExtMallocWithFlags((void **)&flag, 64, hipDeviceMallocFinegrained));
      CHECK(hipMemset(flag, 0, 64));
      CHECK(hipMalloc(&stale, 64)); CHECK(hipMemset(stale, 0, 64));
      CHECK(hipDeviceSynchronize());
      hipStream_t sa, sb;
      CHECK(hipSetDevice(a)); CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
      CHECK(hipSetDevice(b)); CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
      hipLaunchKernelGGL(PullChecker, dim3(1), dim3(256), 0, sb, data, words, flag, ack,
                         rounds, acquire, stale, stuck);
      CHECK(hipSetDevice(a));
      hipLaunchKernelGGL(PullWriter, dim3(1), dim3(256), 0, sa, data, words, flag, ack,
                         rounds, release, stuck);
      CHECK(hipStreamSynchronize(sa));
      CHECK(hipSetDevice(b)); CHECK(hipStreamSynchronize(sb));
      unsigned long long bad = 0;
      CHECK(hipMemcpy(&bad, stale, 8, hipMemcpyDeviceToHost));
      std::printf("pull %d <- %d, %ld words, %ld rounds, A releases at %s, B acquires %s:"
                  " %llu stale words%s\n", b, a, words, rounds,
                  release == 1 ? "system" : "device",
                  acquire == 1 ? "system" : acquire == 2 ? "device" : "none", bad,
                  *stuck ? " (STUCK)" : "");
      CHECK(hipFree(stale)); CHECK(hipFree(flag));
      CHECK(hipSetDevice(a)); CHECK(hipFree(data)); CHECK(hipFree(ack));
      CHECK(hipStreamDestroy(sa)); CHECK(hipSetDevice(b)); CHECK(hipStreamDestroy(sb));
    }
}

// ---- Experiment 2: pull bandwidth by few units, device-timed.
// Every block copies its share, `reps` times; between copies the blocks meet
// at a barrier (device-scope atomics on `bar`) and each does a system-scope
// acquire, which on gfx90a invalidates L2, so no copy reads the last's lines.
__global__ void PullCopy(unsigned long *to, const unsigned long *from, long words,
                         long reps, unsigned *bar, int lanes, unsigned long long *ticks) {
  long nthreads = (long)gridDim.x * lanes;
  long me = (long)blockIdx.x * lanes + threadIdx.x;
  if ((int)threadIdx.x >= lanes) return;
  unsigned blocks = gridDim.x;
  unsigned long long start = 0;
  for (long r = 0; r <= reps; ++r) {
    // A barrier of the blocks: lane 0 of each arrives.
    if (threadIdx.x == 0) {
      unsigned target = (unsigned)(r + 1) * blocks;
      __scoped_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
      while (__scoped_atomic_load_n(bar, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < target) {}
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    if (r == 0) { if (me == 0) start = wall_clock64(); continue; }
    for (long w = me; w < words; w += nthreads) to[w] = from[w];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    __scoped_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
    unsigned target = (unsigned)(reps + 2) * blocks;
    while (__scoped_atomic_load_n(bar, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < target) {}
  }
  if (me == 0) *ticks = wall_clock64() - start;
}

static void PullBandwidth(int a, int b) {
  Peers(a, b);
  const long most = 1L << 20;  // words, 8 MiB
  unsigned long *onA, *onB, *localSrc;
  CHECK(hipSetDevice(a)); CHECK(hipMalloc(&onA, most * 8)); CHECK(hipMemset(onA, 3, most * 8));
  CHECK(hipDeviceSynchronize());
  CHECK(hipSetDevice(b));
  CHECK(hipMalloc(&onB, most * 8)); CHECK(hipMalloc(&localSrc, most * 8));
  CHECK(hipMemset(localSrc, 4, most * 8));
  unsigned *bar; unsigned long long *ticks;
  CHECK(hipMalloc(&bar, 64)); CHECK(hipMalloc(&ticks, 64));
  int rate = 0;
  CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, b));  // kHz
  CHECK(hipDeviceSynchronize());
  struct Shape { const char *name; int blocks, lanes; };
  const Shape shapes[] = {{"1 lane", 1, 1}, {"1 wave", 1, 64}, {"1 team", 1, 256},
                          {"8 teams", 8, 256}, {"32 teams", 32, 256}, {"104 teams", 104, 256}};
  std::printf("pull into %d from %d, GB/s, device clock %d kHz, after a system acquire each copy\n",
              b, a, rate);
  for (int local = 0; local <= 1; ++local) {
    std::printf("%s\n      bytes", local ? "local copy on B (B's memory to B's)"
                                         : "pull (A's memory to B's)");
    for (const Shape &sh : shapes) std::printf(" %10s", sh.name);
    std::printf("\n");
    for (long words = 512; words <= most; words *= 4) {
      std::printf("%11ld", words * 8);
      for (const Shape &sh : shapes) {
        long reps = sh.lanes == 1 ? (words >= (1L << 16) ? 2 : 10) : (words >= (1L << 18) ? 10 : 100);
        CHECK(hipMemset(bar, 0, 64));
        hipLaunchKernelGGL(PullCopy, dim3(sh.blocks), dim3(sh.lanes < 64 ? 64 : sh.lanes), 0, 0,
                           onB, local ? localSrc : onA, words, reps, bar, sh.lanes, ticks);
        CHECK(hipDeviceSynchronize());
        unsigned long long t = 0;
        CHECK(hipMemcpy(&t, ticks, 8, hipMemcpyDeviceToHost));
        double seconds = (double)t / (rate * 1000.0) / reps;
        std::printf(" %10.2f", words * 8 / seconds / 1e9);
      }
      std::printf("\n");
    }
  }
}

// ---- Experiment 3: the cost of a poll.
// One wave polls `word` `polls` times; mode 0 relaxed, 1 acquire, both at
// system scope. Lane 0 alone polls, as a process's lane does.
__global__ void Poller(long *word, long polls, int mode, unsigned long long *ticks,
                       long *stop) {
  if (threadIdx.x != 0) return;
  long seen = 0;
  unsigned long long start = wall_clock64();
  // With `stop`, polls until it is set, looking at it (host memory) only
  // every 4096 polls so that the word polled is nearly all of the traffic.
  for (long i = 0;; ++i) {
    if (!stop && i >= polls) break;
    if (stop && (i & 4095) == 0 &&
        __scoped_atomic_load_n(stop, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) break;
    if (mode == 2)
      seen += __scoped_atomic_load_n(word, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
    else if (mode == 1)
      seen += __scoped_atomic_load_n(word, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    else
      seen += __scoped_atomic_load_n(word, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
  if (seen == 42) word[1] = seen;
  *ticks = wall_clock64() - start;
}

// Reads 4 MiB of B's memory `reps` times, with all units, device-timed.
__global__ void Reread(const uint4 *from, long n, long reps, unsigned *bar,
                       unsigned long long *ticks, uint4 *sink) {
  long stride = (long)gridDim.x * blockDim.x;
  long me = (long)blockIdx.x * blockDim.x + threadIdx.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  unsigned long long start = wall_clock64();
  for (long r = 0; r < reps; ++r)
    for (long i = me; i < n; i += stride) {
      uint4 v = from[i];
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if (acc.x == 0x12345678u) sink[0] = acc;
  __syncthreads();
  if (threadIdx.x == 0) __scoped_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
  if (me == 0) {
    while (__scoped_atomic_load_n(bar, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < gridDim.x) {}
    *ticks = wall_clock64() - start;
  }
}

static void Poll(int a, int b) {
  (void)a;
  CHECK(hipSetDevice(b));
  long *fine, *host, *stop;
  CHECK(hipExtMallocWithFlags((void **)&fine, 64, hipDeviceMallocFinegrained));
  CHECK(hipMemset(fine, 0, 64));
  CHECK(hipHostMalloc((void **)&host, 64, hipHostMallocCoherent)); host[0] = 0; host[1] = 0;
  CHECK(hipHostMalloc((void **)&stop, 64, hipHostMallocCoherent));
  unsigned long long *ticks, *ticks2; unsigned *bar; uint4 *buf, *sink;
  CHECK(hipMalloc(&ticks, 64)); CHECK(hipMalloc(&ticks2, 64)); CHECK(hipMalloc(&bar, 64));
  const long n = (4L << 20) / 16;
  CHECK(hipMalloc(&buf, n * 16)); CHECK(hipMemset(buf, 5, n * 16)); CHECK(hipMalloc(&sink, 64));
  int rate = 0, units = 0;
  CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, b));
  CHECK(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, b));
  CHECK(hipDeviceSynchronize());
  hipStream_t s1, s2;
  CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  struct Case { const char *name; long *word; int mode; };
  long *coarse;
  CHECK(hipMalloc(&coarse, 64)); CHECK(hipMemset(coarse, 0, 64));
  const Case cases[] = {{"coarse device word, device acquire", coarse, 2},
                        {"fine-grained device word, relaxed", fine, 0},
                        {"fine-grained device word, acquire", fine, 1},
                        {"host word, relaxed", host, 0},
                        {"host word, acquire", host, 1}};
  const long polls = 200000;
  for (const Case &c : cases) {
    hipLaunchKernelGGL(Poller, dim3(1), dim3(64), 0, s1, c.word, polls, c.mode, ticks,
                       (long *)nullptr);
    CHECK(hipStreamSynchronize(s1));
    unsigned long long t = 0;
    CHECK(hipMemcpy(&t, ticks, 8, hipMemcpyDeviceToHost));
    std::printf("poll on %d, %-36s %8.1f ns a poll\n", b, c.name,
                (double)t / (rate * 1e3) / polls * 1e9);
  }
  const long reps = 400;
  auto reread = [&](const char *name, const Case *c) {
    *stop = 0;
    if (c)
      hipLaunchKernelGGL(Poller, dim3(1), dim3(64), 0, s1, c->word, 0, c->mode, ticks2, stop);
    CHECK(hipMemset(bar, 0, 64));
    hipLaunchKernelGGL(Reread, dim3(units * 4), dim3(256), 0, s2, buf, n, reps, bar, ticks, sink);
    CHECK(hipStreamSynchronize(s2));
    *stop = 1;
    CHECK(hipStreamSynchronize(s1));
    unsigned long long t = 0;
    CHECK(hipMemcpy(&t, ticks, 8, hipMemcpyDeviceToHost));
    double seconds = (double)t / (rate * 1e3);
    std::printf("reread 4 MiB on %d x%ld, %-40s %8.1f GB/s\n", b, reps, name,
                n * 16.0 * reps / seconds / 1e9);
  };
  reread("no poller", nullptr);
  for (const Case &c : cases) reread(c.name, &c);
}

// ---- Experiment 4: push against pull, one wave each side.
// Push: the sender stores the payload into the receiver's fine-grained
// buffer, then its flag. Pull: the sender stores it into its own coarse
// buffer, releases at system scope, sets the receiver's flag; the receiver
// acquires at system scope and loads from the sender's buffer.
__global__ void PingPong(unsigned long *myOut, unsigned long *theirIn, unsigned long *myIn,
                         unsigned long *theirOut, long *myFlag, long *theirFlag, long words,
                         long rounds, int first, int pull, unsigned long *sink, long *stuck) {
  long me = threadIdx.x;
  unsigned long acc = 0;
  for (long i = 1; i <= rounds; ++i) {
    for (int turn = 0; turn < 2; ++turn) {
      bool sending = (turn == 0) == (first != 0);
      if (sending) {
        unsigned long *dst = pull ? myOut : theirIn;
        for (long w = me; w < words; w += blockDim.x) dst[w] = (unsigned long)i + w;
        __syncthreads();
        if (me == 0) {
          __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
          __scoped_atomic_store_n(theirFlag, i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        }
      } else {
        if (me == 0)
          WAIT(__scoped_atomic_load_n(myFlag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) == i);
        __syncthreads();
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,
                                     pull ? __MEMORY_SCOPE_SYSTEM : __MEMORY_SCOPE_DEVICE);
        const unsigned long *src = pull ? theirOut : myIn;
        for (long w = me; w < words; w += blockDim.x) acc += src[w];
        __syncthreads();
      }
    }
  }
  if (acc == 42) *sink = acc;
}

static void Push(int a, int b) {
  Peers(a, b);
  const long most = 8192;  // words, 64 KiB
  unsigned long *outA, *inA, *outB, *inB, *sink; long *flagA, *flagB, *stuck;
  CHECK(hipHostMalloc((void **)&stuck, 64, hipHostMallocCoherent));
  CHECK(hipSetDevice(a));
  CHECK(hipMalloc(&outA, most * 8));
  CHECK(hipExtMallocWithFlags((void **)&inA, most * 8, hipDeviceMallocFinegrained));
  CHECK(hipExtMallocWithFlags((void **)&flagA, 64, hipDeviceMallocFinegrained));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipDeviceSynchronize());
  CHECK(hipSetDevice(b));
  CHECK(hipMalloc(&outB, most * 8));
  CHECK(hipExtMallocWithFlags((void **)&inB, most * 8, hipDeviceMallocFinegrained));
  CHECK(hipExtMallocWithFlags((void **)&flagB, 64, hipDeviceMallocFinegrained));
  CHECK(hipDeviceSynchronize());
  hipStream_t sa, sb;
  CHECK(hipSetDevice(a)); CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
  CHECK(hipSetDevice(b)); CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
  std::printf("ping-pong %d <-> %d, one wave each, one way, us (launches included)\n"
              "      bytes       push       pull\n", a, b);
  for (long words = 1; words <= most; words *= 8) {
    std::printf("%11ld", words * 8);
    for (int pull = 0; pull <= 1; ++pull) {
      long rounds = 20000;
      *stuck = 0;
      CHECK(hipSetDevice(a)); CHECK(hipMemset(flagA, 0, 64)); CHECK(hipDeviceSynchronize());
      CHECK(hipSetDevice(b)); CHECK(hipMemset(flagB, 0, 64)); CHECK(hipDeviceSynchronize());
      double start = Seconds();
      hipLaunchKernelGGL(PingPong, dim3(1), dim3(64), 0, sb, outB, inA, inB, outA, flagB, flagA,
                         words, rounds, 0, pull, sink, stuck);
      CHECK(hipSetDevice(a));
      hipLaunchKernelGGL(PingPong, dim3(1), dim3(64), 0, sa, outA, inB, inA, outB, flagA, flagB,
                         words, rounds, 1, pull, sink, stuck);
      CHECK(hipStreamSynchronize(sa));
      CHECK(hipSetDevice(b)); CHECK(hipStreamSynchronize(sb));
      double t = Seconds() - start;
      if (*stuck) std::printf("      STUCK");
      else std::printf(" %10.2f", t / rounds / 2 * 1e6);
    }
    std::printf("\n");
  }
}

// Experiment 5 (peer slot). The slots, in B's fine-grained memory, as
// place.h's PeerQueueSlot: a sequence and the message's words.
constexpr int kSlots = 16;
constexpr int kSlotWords = 64;
struct Slot { unsigned long sequence; unsigned long words[kSlotWords]; };

__device__ unsigned long Word(long i, long k) {
  return (unsigned long)i * 0x9E3779B97F4A7C15ul + (unsigned long)k;
}

// mode: 0 relaxed, no wait; 1 release at device scope; 2 at system scope.
__global__ void SlotSender(Slot *slots, long messages, long words, int mode,
                           unsigned long *dirty, int lines,
                           unsigned long long *ticks) {
  if (threadIdx.x != 0) return;
  unsigned long long t0 = __builtin_readsteadycounter();
  for (long i = 0; i < messages; ++i) {
    for (int l = 0; l < lines; ++l) dirty[l * 16] = i;
    Slot &slot = slots[i % kSlots];
    while (__scoped_atomic_load_n(&slot.sequence, __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_SYSTEM) != (unsigned long)i)
      __builtin_amdgcn_s_sleep(1);
    for (long k = 0; k < words; ++k)
      __scoped_atomic_store_n(&slot.words[k], Word(i, k), __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);
    if (mode == 0)
      __scoped_atomic_store_n(&slot.sequence, i + 1, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);
    else if (mode == 1)
      __scoped_atomic_store_n(&slot.sequence, i + 1, __ATOMIC_RELEASE,
                              __MEMORY_SCOPE_DEVICE);
    else
      __scoped_atomic_store_n(&slot.sequence, i + 1, __ATOMIC_RELEASE,
                              __MEMORY_SCOPE_SYSTEM);
  }
  *ticks = __builtin_readsteadycounter() - t0;
}

// Other teams on A, writing `words` of A's memory over and over until
// *stop.
__global__ void SlotNoise(unsigned long *buffer, long words, long *stop) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long step = (long)gridDim.x * blockDim.x;
  for (unsigned long r = 1;; ++r) {
    for (long k = i; k < words; k += step) buffer[k] = r;
    if (__scoped_atomic_load_n(stop, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM))
      return;
  }
}

__global__ void SlotReceiver(Slot *slots, long messages, long words,
                             unsigned long long *bad) {
  if (threadIdx.x != 0) return;
  unsigned long long wrong = 0;
  for (long i = 0; i < messages; ++i) {
    Slot &slot = slots[i % kSlots];
    while (__scoped_atomic_load_n(&slot.sequence, __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_SYSTEM) != (unsigned long)i + 1)
      ;
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
    for (long k = 0; k < words; ++k)
      if (__scoped_atomic_load_n(&slot.words[k], __ATOMIC_RELAXED,
                                 __MEMORY_SCOPE_SYSTEM) != Word(i, k))
        ++wrong;
    __scoped_atomic_store_n(&slot.sequence, i + kSlots, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_DEVICE);
  }
  *bad = wrong;
}

static void Slots(int a, int b, long messages, long words) {
  if (words < 1 || words > kSlotWords) {
    std::fprintf(stderr, "words: 1 to %d\n", kSlotWords); std::exit(2);
  }
  Peers(a, b);
  Slot *slots; unsigned long long *bad, *ticks; unsigned long *dirty;
  CHECK(hipSetDevice(b));
  CHECK(hipExtMallocWithFlags((void **)&slots, sizeof(Slot) * kSlots,
                              hipDeviceMallocFinegrained));
  CHECK(hipMalloc((void **)&bad, 8));
  CHECK(hipSetDevice(a));
  CHECK(hipMalloc((void **)&ticks, 8));
  CHECK(hipMalloc((void **)&dirty, 64 * 128));
  const long noiseWords = (32l << 20) / 8;
  unsigned long *noise; long *stop;
  CHECK(hipMalloc((void **)&noise, noiseWords * 8));
  // In host memory, which the host sets as the kernels run: a copy on the
  // null stream would wait for SlotNoise, which waits for it.
  CHECK(hipHostMalloc((void **)&stop, 64, hipHostMallocCoherent));
  hipStream_t sa, sb, sn;
  CHECK(hipSetDevice(a)); CHECK(hipStreamCreate(&sa)); CHECK(hipStreamCreate(&sn));
  CHECK(hipSetDevice(b)); CHECK(hipStreamCreate(&sb));
  std::printf("%ld messages of %ld words, A=%d B=%d\n", messages, words, a, b);
  std::printf("%-36s %12s %12s\n", "publish", "us a message", "wrong words");
  const char *names[3] = {"relaxed, no wait", "release, device scope",
                          "release, system scope"};
  for (int load = 0; load < 3; ++load)
    for (int mode = 0; mode < 3; ++mode) {
      int lines = load == 1 ? 64 : 0;
      if (load == 2) {
        CHECK(hipSetDevice(a));
        __atomic_store_n(stop, 0, __ATOMIC_SEQ_CST);
        hipLaunchKernelGGL(SlotNoise, dim3(96), dim3(256), 0, sn, noise,
                           noiseWords, stop);
      }
      Slot init[kSlots];
      std::memset(init, 0, sizeof init);
      for (int i = 0; i < kSlots; ++i) init[i].sequence = i;
      CHECK(hipSetDevice(b));
      CHECK(hipMemcpy(slots, init, sizeof init, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(SlotReceiver, dim3(1), dim3(64), 0, sb, slots,
                         messages, words, bad);
      CHECK(hipSetDevice(a));
      hipLaunchKernelGGL(SlotSender, dim3(1), dim3(64), 0, sa, slots, messages,
                         words, mode, dirty, lines, ticks);
      CHECK(hipStreamSynchronize(sa));
      CHECK(hipSetDevice(b)); CHECK(hipStreamSynchronize(sb));
      if (load == 2) {
        CHECK(hipSetDevice(a));
        __atomic_store_n(stop, 1, __ATOMIC_SEQ_CST);
        CHECK(hipStreamSynchronize(sn));
      }
      unsigned long long t = 0, w = 0;
      CHECK(hipSetDevice(a)); CHECK(hipMemcpy(&t, ticks, 8, hipMemcpyDeviceToHost));
      CHECK(hipSetDevice(b)); CHECK(hipMemcpy(&w, bad, 8, hipMemcpyDeviceToHost));
      char name[64];
      std::snprintf(name, sizeof name, "%s%s", names[mode],
                    load == 1 ? ", 64 lines" : load == 2 ? ", other teams" : "");
      // The steady counter ticks at 25 MHz on gfx90a.
      std::printf("%-36s %12.3f %12llu\n", name, t * 0.04 / messages, w);
    }
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: see the file\n"); return 2; }
  if (!std::strcmp(argv[1], "info")) { Info(); return 0; }
  if (argc < 4) { std::fprintf(stderr, "give two devices\n"); return 2; }
  int a = std::atoi(argv[2]), b = std::atoi(argv[3]);
  if (!std::strcmp(argv[1], "bw")) Bandwidth(a, b);
  else if (!std::strcmp(argv[1], "lat")) {
    long rounds = argc > 4 ? std::atol(argv[4]) : 100000;
    Latency(a, b, rounds, true);
    Latency(a, b, rounds, false);
  } else if (!std::strcmp(argv[1], "coh"))
    Coherence(a, b, argc > 4 ? std::atol(argv[4]) : 4096,
              argc > 5 ? std::atol(argv[5]) : 200, argc > 6 ? std::atoi(argv[6]) : 1,
              argc > 7 && std::atoi(argv[7]) == 1);
  else if (!std::strcmp(argv[1], "pull"))
    Pull(a, b, argc > 4 ? std::atol(argv[4]) : 4096, argc > 5 ? std::atol(argv[5]) : 200);
  else if (!std::strcmp(argv[1], "pullbw")) PullBandwidth(a, b);
  else if (!std::strcmp(argv[1], "poll")) Poll(a, b);
  else if (!std::strcmp(argv[1], "push")) Push(a, b);
  else if (!std::strcmp(argv[1], "slot"))
    Slots(a, b, argc > 4 ? std::atol(argv[4]) : 1000000,
          argc > 5 ? std::atol(argv[5]) : 4);
  return 0;
}
