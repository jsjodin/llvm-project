// The kernel of grid-barrier-gpu.test: rounds in which every thread writes its
// slot, crosses merk_device_grid_barrier, and checks a slot of the next team
// and of the team before it, written in the same round. A barrier that let a
// thread through early, or did not make the writes visible, leaves a slot of
// the round before. Then a second barrier, so that no slot is written for the
// next round before every thread has read it.
#include <gpuintrin.h>
#include <stdint.h>

struct MerkGridBarrier;
extern "C" void merk_device_grid_barrier(MerkGridBarrier *barrier);

// The image's kernel of GPU processes, merk_place, which the start object
// links in, starts kinds through merk_process_dispatch and runs their loops
// across teams through merk_process_loop and merk_team_loop, with their
// reductions through merk_process_loop_start and merk_process_loop_reduce,
// which the device's compile of a module makes; this image has no kinds.
extern "C" void merk_process_dispatch(uint64_t, uint64_t) {}
extern "C" void merk_process_loop(uint64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, uint64_t, uint64_t) {}
extern "C" void merk_team_loop(uint64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, uint64_t, uint64_t) {}
extern "C" void merk_process_loop_start(uint64_t, uint64_t *) {}
extern "C" void merk_process_loop_reduce(
    uint64_t, uint64_t *, const uint64_t *) {}

extern "C" __gpu_kernel void grid_barrier_test(MerkGridBarrier *barrier,
    uint64_t *slots, uint32_t rounds, uint32_t *errors) {
  uint64_t teams{__gpu_num_blocks_x()};
  uint64_t threads{__gpu_num_threads_x()};
  uint64_t team{__gpu_block_id_x()};
  uint64_t thread{__gpu_thread_id_x()};
  uint64_t next{(team + 1) % teams * threads + thread};
  uint64_t before{
      (team + teams - 1) % teams * threads + (threads - 1 - thread)};
  for (uint64_t round{1}; round <= rounds; ++round) {
    slots[team * threads + thread] = round;
    merk_device_grid_barrier(barrier);
    if (slots[next] != round || slots[before] != round)
      __scoped_atomic_fetch_add(
          errors, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    merk_device_grid_barrier(barrier);
  }
}
