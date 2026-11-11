#include <gpuintrin.h>
#include <stdint.h>

// Every work group arrives at a barrier across the grid, which only works if
// all of them are resident at once, as a cooperative launch promises. A group
// that waits too long gives up and reports it, so that a launch that breaks
// the promise fails the test rather than hanging the device.
extern "C" __gpu_kernel void cooperative_barrier(uint32_t *Arrived,
                                                 uint32_t *TimedOut) {
  if (__gpu_thread_id(0) != 0)
    return;
  uint32_t Groups = __gpu_num_blocks(0);
  __scoped_atomic_fetch_add(Arrived, 1u, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
  for (uint32_t I = 0; I < (1u << 20); ++I) {
    if (__scoped_atomic_load_n(Arrived, __ATOMIC_RELAXED,
                               __MEMORY_SCOPE_DEVICE) >= Groups)
      return;
#ifdef __AMDGPU__
    __builtin_amdgcn_s_sleep(2);
#endif
  }
  __scoped_atomic_fetch_add(TimedOut, 1u, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
}
