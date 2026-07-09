// Host compilation (x86 host, AMDGPU offload target): no address space on allocas.
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas in private address space, addrspacecast for map info.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

#define N 10000

// A pointer array section map(p[0:N]) maps the pointed-to data. The map operand
// is the loaded pointer value with omp.map.bounds describing the section, and
// the region body accesses the data through a local slot holding the (device)
// data pointer.
void write_section(int *p) {
  // CIR-HOST: cir.func{{.*}}@write_section
  // CIR-HOST: %[[P_ALLOCA:.*]] = cir.alloca "p" align(8) init : !cir.ptr<!cir.ptr<!s32i>>
  // CIR-HOST: %[[P_VAL:.*]] = cir.load %[[P_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR-HOST: %[[BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}} : i64) upper_bound({{.*}} : i64) extent({{.*}} : i64) stride({{.*}} : i64) start_idx({{.*}} : i64)
  // CIR-HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[P_VAL]] : !cir.ptr<!s32i>, !s32i) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !cir.ptr<!s32i> {name = "p"}
  // CIR-HOST: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // The region materializes a local slot holding the mapped data pointer.
  // CIR-HOST: %[[SLOT:.*]] = cir.alloca "p" align(8) : !cir.ptr<!cir.ptr<!s32i>>
  // CIR-HOST: cir.store align(8) %[[ARG]], %[[SLOT]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR-HOST: cir.load align(8) %[[SLOT]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>

  // CIR-DEVICE: cir.func{{.*}}@write_section
  // CIR-DEVICE: %[[P_VAL:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!s32i>, target_address_space(5)>, !cir.ptr<!s32i>
  // Bounds are host-only metadata; the device map.info carries no bounds.
  // CIR-DEVICE-NOT: omp.map.bounds
  // CIR-DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[P_VAL]] : !cir.ptr<!s32i>, !s32i) map_clauses(tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "p"}
  // CIR-DEVICE: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
#pragma omp target teams distribute parallel for map(tofrom : p[0 : N])
  for (int i = 0; i < N; i = i + 1)
    p[i] = i;
}
