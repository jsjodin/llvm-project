// Host compilation (x86 host, AMDGPU offload target): no address space on allocas.
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas in private address space, addrspacecast for map info.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// A whole fixed-size array is mapped with omp.map.bounds describing the full
// extent [0, N-1], so the runtime maps the entire buffer.
void target_map_array(void) {
  int a[256];
  // CIR-HOST: cir.func{{.*}}@target_map_array
  // CIR-HOST: %[[A_ALLOCA:.*]] = cir.alloca "a" align(16) : !cir.ptr<!cir.array<!s32i x 256>>
  // CIR-HOST: %[[LB:.*]] = cir.const #cir.int<0> : !s64i
  // CIR-HOST: %[[LB_I:.*]] = cir.builtin_int_cast %[[LB]] : !s64i -> i64
  // CIR-HOST: %[[UB:.*]] = cir.const #cir.int<255> : !s64i
  // CIR-HOST: %[[UB_I:.*]] = cir.builtin_int_cast %[[UB]] : !s64i -> i64
  // CIR-HOST: %[[EX:.*]] = cir.const #cir.int<256> : !s64i
  // CIR-HOST: %[[EX_I:.*]] = cir.builtin_int_cast %[[EX]] : !s64i -> i64
  // CIR-HOST: %[[ST:.*]] = cir.const #cir.int<1> : !s64i
  // CIR-HOST: %[[ST_I:.*]] = cir.builtin_int_cast %[[ST]] : !s64i -> i64
  // CIR-HOST: %[[SI:.*]] = cir.const #cir.int<0> : !s64i
  // CIR-HOST: %[[SI_I:.*]] = cir.builtin_int_cast %[[SI]] : !s64i -> i64
  // CIR-HOST: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB_I]] : i64) upper_bound(%[[UB_I]] : i64) extent(%[[EX_I]] : i64) stride(%[[ST_I]] : i64) start_idx(%[[SI_I]] : i64)
  // CIR-HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[A_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 256>>, !cir.array<!s32i x 256>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 256>> {name = "a"}
  // CIR-HOST: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!cir.array<!s32i x 256>>) {
  // CIR-HOST: cir.get_element %[[ARG]]

  // CIR-DEVICE: cir.func{{.*}}@target_map_array
  // CIR-DEVICE: %[[A_ALLOCA:.*]] = cir.alloca "a" align(4) : !cir.ptr<!cir.array<!s32i x 256>, target_address_space(5)>
  // CIR-DEVICE: %[[CAST:.*]] = cir.cast address_space %[[A_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 256>, target_address_space(5)> -> !cir.ptr<!cir.array<!s32i x 256>>
  // Bounds are host-only metadata; the device map.info carries no bounds.
  // CIR-DEVICE-NOT: omp.map.bounds
  // CIR-DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!cir.array<!s32i x 256>>, !cir.array<!s32i x 256>) map_clauses(tofrom) capture(ByRef) -> !cir.ptr<!cir.array<!s32i x 256>> {name = "a"}
  // CIR-DEVICE: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!cir.array<!s32i x 256>>) {
#pragma omp target map(tofrom : a)
  {
    a[0] = 42;
  }
}
