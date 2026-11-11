#include <gpuintrin.h>
#include <stdint.h>

// Sets this kernel's flag, Flags[Mine], and waits for its peer's, the other
// of Flags[0] and Flags[1], which the peer kernel sets: only kernels that run
// at once both see the other's. A kernel that waits too long gives up and
// reports it, so that kernels that run one after the other fail the test
// rather than hang the device.
extern "C" __gpu_kernel void wait_for_peer(uint32_t *Flags, uint32_t Mine,
                                           uint32_t *TimedOut) {
  if (__gpu_thread_id(0) != 0 || __gpu_block_id(0) != 0)
    return;
  __scoped_atomic_store_n(&Flags[Mine], 1u, __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_DEVICE);
  for (uint32_t I = 0; I < (1u << 20); ++I) {
    if (__scoped_atomic_load_n(&Flags[1 - Mine], __ATOMIC_RELAXED,
                               __MEMORY_SCOPE_DEVICE) != 0)
      return;
#ifdef __AMDGPU__
    __builtin_amdgcn_s_sleep(2);
#endif
  }
  __scoped_atomic_fetch_add(TimedOut, 1u, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
}
