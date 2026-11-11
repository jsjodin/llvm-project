#include <gpuintrin.h>
#include <stdint.h>

// A recursion that the compiler cannot bound, so that the kernel's stack
// size is not fixed: each call keeps a frame of its own.
[[clang::noinline]] static uint32_t sum(volatile uint32_t *Frames,
                                        uint32_t Depth) {
  volatile uint32_t Frame[16];
  for (uint32_t I = 0; I < 16; ++I)
    Frame[I] = Depth + I;
  if (Depth == 0)
    return Frame[0];
  uint32_t Rest = sum(Frames, Depth - 1);
  return Rest + Frame[15] - 15;
}

extern "C" __gpu_kernel void recursive(uint32_t *Out, uint32_t Depth) {
  Out[__gpu_num_threads_x() * __gpu_block_id_x() + __gpu_thread_id_x()] =
      sum(Out, Depth);
}
