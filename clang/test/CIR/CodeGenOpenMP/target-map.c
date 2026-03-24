// Host compilation (x86 host, AMDGPU offload target): no address space on allocas.
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST

// Device compilation (AMDGPU): allocas in private address space, addrspacecast for map info.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE

void use(int);

void target_map_to(int x) {
  // HOST: cir.func{{.*}}@target_map_to
  // HOST: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // HOST-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // HOST-NEXT: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // HOST-NEXT: cir.call @use(%[[LOAD]])
  // HOST-NEXT: omp.terminator
  // HOST-NEXT: }

  // DEVICE: cir.func{{.*}}@target_map_to
  // DEVICE: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, addrspace(offload_private)>, ["x", init]
  // DEVICE: %[[CAST:.*]] = cir.cast(address_space, %[[X_ALLOCA]] : !cir.ptr<!s32i, addrspace(offload_private)>), !cir.ptr<!s32i>
  // DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // DEVICE-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // DEVICE: omp.terminator
  // DEVICE-NEXT: }
#pragma omp target map(to : x)
  {
    use(x);
  }
}

void target_map_from(int x) {
  // HOST: cir.func{{.*}}@target_map_from
  // HOST: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // HOST-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // HOST-NEXT: %[[C42:.*]] = cir.const #cir.int<42> : !s32i
  // HOST-NEXT: cir.store align(4) %[[C42]], %[[ARG]]
  // HOST-NEXT: omp.terminator
  // HOST-NEXT: }

  // DEVICE: cir.func{{.*}}@target_map_from
  // DEVICE: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, addrspace(offload_private)>, ["x", init]
  // DEVICE: %[[CAST:.*]] = cir.cast(address_space, %[[X_ALLOCA]] : !cir.ptr<!s32i, addrspace(offload_private)>), !cir.ptr<!s32i>
  // DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // DEVICE-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // DEVICE: omp.terminator
  // DEVICE-NEXT: }
#pragma omp target map(from : x)
  {
    x = 42;
  }
}

void target_map_tofrom(int x) {
  // HOST: cir.func{{.*}}@target_map_tofrom
  // HOST: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // HOST-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // HOST: omp.terminator
  // HOST-NEXT: }

  // DEVICE: cir.func{{.*}}@target_map_tofrom
  // DEVICE: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, addrspace(offload_private)>, ["x", init]
  // DEVICE: %[[CAST:.*]] = cir.cast(address_space, %[[X_ALLOCA]] : !cir.ptr<!s32i, addrspace(offload_private)>), !cir.ptr<!s32i>
  // DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // DEVICE-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // DEVICE: omp.terminator
  // DEVICE-NEXT: }
#pragma omp target map(tofrom : x)
  {
    x = x + 1;
  }
}

void target_map_multiple(int a, int b) {
  // HOST: cir.func{{.*}}@target_map_multiple
  // HOST-DAG: %[[A_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
  // HOST-DAG: %[[B_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
  // HOST: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "a"}
  // HOST-NEXT: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "b"}
  // HOST-NEXT: omp.target map_entries(%[[MAP_A]] -> %[[ARG_A:.*]], %[[MAP_B]] -> %[[ARG_B:.*]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // HOST: omp.terminator
  // HOST-NEXT: }

  // DEVICE: cir.func{{.*}}@target_map_multiple
  // DEVICE-DAG: %[[A_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, addrspace(offload_private)>, ["a", init]
  // DEVICE-DAG: %[[B_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, addrspace(offload_private)>, ["b", init]
  // DEVICE: %[[CAST_A:.*]] = cir.cast(address_space, %[[A_ALLOCA]] : !cir.ptr<!s32i, addrspace(offload_private)>), !cir.ptr<!s32i>
  // DEVICE: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[CAST_A]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "a"}
  // DEVICE: %[[CAST_B:.*]] = cir.cast(address_space, %[[B_ALLOCA]] : !cir.ptr<!s32i, addrspace(offload_private)>), !cir.ptr<!s32i>
  // DEVICE: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[CAST_B]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "b"}
  // DEVICE: omp.target map_entries(%[[MAP_A]] -> %[[ARG_A:.*]], %[[MAP_B]] -> %[[ARG_B:.*]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // DEVICE: omp.terminator
  // DEVICE-NEXT: }
#pragma omp target map(to : a) map(from : b)
  {
    b = a;
  }
}

// Test implicit mapping: variables used in target region without explicit map clause.
void target_implicit_map(int x) {
  // HOST: cir.func{{.*}}@target_implicit_map
  // HOST: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(implicit, tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // HOST-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // HOST-NEXT: %[[C42:.*]] = cir.const #cir.int<42> : !s32i
  // HOST-NEXT: cir.store align(4) %[[C42]], %[[ARG]]
  // HOST-NEXT: omp.terminator
  // HOST-NEXT: }

  // DEVICE: cir.func{{.*}}@target_implicit_map
  // DEVICE: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, addrspace(offload_private)>, ["x", init]
  // DEVICE: %[[CAST:.*]] = cir.cast(address_space, %[[X_ALLOCA]] : !cir.ptr<!s32i, addrspace(offload_private)>), !cir.ptr<!s32i>
  // DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(implicit, tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // DEVICE-NEXT: omp.target map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // DEVICE: omp.terminator
  // DEVICE-NEXT: }
#pragma omp target
  {
    x = 42;
  }
}
