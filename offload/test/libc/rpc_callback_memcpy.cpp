// RUN: %libomptarget-compilexx-generic
// RUN: %libomptarget-run-generic | %fcheck-generic
// REQUIRES: libc
// REQUIRES: amdgcn-amd-amdhsa

// An RPC callback copies the memory of the kernel that called it, with
// omp_target_memcpy, while the kernel waits for it, and the kernel copies a
// result back when it ends. The kernel's copy back waits for the kernel; the
// callback's copies must not wait behind it.

#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>

// CHECK: PASS

// If the RPC headers are not present, just pass the test.
#if !__has_include(<../../libc/shared/rpc.h>)
int main() { printf("PASS\n"); }
#else

#include <../../libc/shared/rpc.h>
#include <../../libc/shared/rpc_dispatch.h>

[[gnu::weak]] rpc::Client client asm("__llvm_rpc_client");
#pragma omp declare target to(client) device_type(nohost)

constexpr uint32_t DOUBLE_OPCODE = 1;
constexpr int N = 64;

// Doubles the N words at a device address: copies them to the host and back.
int double_words(uintptr_t address) {
  uint64_t words[N];
  int device = omp_get_default_device();
  int host = omp_get_initial_device();
  if (omp_target_memcpy(words, reinterpret_cast<void *>(address),
                        sizeof(words), 0, 0, host, device))
    return 1;
  for (int i = 0; i < N; ++i)
    words[i] *= 2;
  return omp_target_memcpy(reinterpret_cast<void *>(address), words,
                           sizeof(words), 0, 0, device, host);
}

#pragma omp begin declare variant match(device = {kind(gpu)})
int double_words(uintptr_t address) {
  return rpc::dispatch<DOUBLE_OPCODE, double_words>(client, address);
}
#pragma omp end declare variant

static uint32_t handleOpcodes(void *raw, uint32_t numLanes) {
  rpc::Server::Port &Port = *reinterpret_cast<rpc::Server::Port *>(raw);
  if (Port.get_opcode() != DOUBLE_OPCODE)
    return rpc::RPC_UNHANDLED_OPCODE;
  if (numLanes == 1)
    rpc::invoke<1>(Port, double_words);
  else if (numLanes == 32)
    rpc::invoke<32>(Port, double_words);
  else if (numLanes == 64)
    rpc::invoke<64>(Port, double_words);
  else
    return rpc::RPC_ERROR;
  return rpc::RPC_SUCCESS;
}

extern "C" void __tgt_register_rpc_callback(unsigned (*callback)(void *,
                                                                 unsigned));

[[gnu::constructor]] void register_callback() {
  __tgt_register_rpc_callback(&handleOpcodes);
}

int main() {
  uint64_t *words = static_cast<uint64_t *>(
      omp_target_alloc(N * sizeof(uint64_t), omp_get_default_device()));
  int failures = -1;
#pragma omp target is_device_ptr(words) map(from : failures)
  {
    failures = 0;
    for (int i = 0; i < N; ++i)
      words[i] = i;
    // More than one round, so that the copies use each copy engine.
    for (int round = 0; round < 4; ++round)
      failures += double_words(reinterpret_cast<uintptr_t>(words));
    for (int i = 0; i < N; ++i)
      failures += words[i] != 16 * static_cast<uint64_t>(i);
  }
  omp_target_free(words, omp_get_default_device());
  assert(failures == 0);
  printf("PASS\n");
}

#endif
