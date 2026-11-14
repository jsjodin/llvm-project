// The kernels of fine-memory-gpu.test.
#include <gpuintrin.h>
#include <stdint.h>

// Adds 1 to *counter `rounds` times at system scope, from every thread.
extern "C" __gpu_kernel void fine_count(uint64_t *counter, uint32_t rounds) {
  for (uint32_t i{0}; i < rounds; ++i)
    __scoped_atomic_fetch_add(counter, 1, __ATOMIC_RELAXED,
        __MEMORY_SCOPE_SYSTEM);
}

// One side of a ping-pong between two GPUs: `mine` is in this GPU's memory
// and written by the other's kernel, `theirs` in the other GPU's. The first
// side starts. A side that waits for about 2^24 polls sets *stuck and ends:
// a word written by a peer in coarse-grained memory is never seen.
extern "C" __gpu_kernel void fine_ping(uint64_t *mine, uint64_t *theirs,
    uint32_t rounds, uint32_t first, uint32_t *stuck) {
  if (__gpu_thread_id_x() != 0 || __gpu_block_id_x() != 0)
    return;
  for (uint64_t i{1}; i <= rounds; ++i) {
    if (first)
      __scoped_atomic_store_n(theirs, i, __ATOMIC_RELEASE,
          __MEMORY_SCOPE_SYSTEM);
    uint64_t polls{0};
    while (__scoped_atomic_load_n(mine, __ATOMIC_ACQUIRE,
               __MEMORY_SCOPE_SYSTEM) != i) {
      if (++polls > (uint64_t{1} << 24)) {
        __scoped_atomic_store_n(stuck, 1u, __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
        return;
      }
    }
    if (!first)
      __scoped_atomic_store_n(theirs, i, __ATOMIC_RELEASE,
          __MEMORY_SCOPE_SYSTEM);
  }
}
