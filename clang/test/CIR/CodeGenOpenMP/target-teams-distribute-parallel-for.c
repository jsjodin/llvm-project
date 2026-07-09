// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU).
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void during(int);

// The combined `target teams distribute parallel for` directive decomposes into
// `target`, `teams`, `parallel`, `distribute` and `for` leaves. The composite
// `distribute parallel for` tail lowers to a composite omp.parallel wrapping the
// composite omp.distribute + omp.wsloop stack, nested inside omp.teams inside
// omp.target. This is a target SPMD construct, so the omp.target is marked
// kernel_type(spmd) and combined, its loop trip count is evaluated on the host
// and forwarded through host_eval block arguments, and the omp.loop_nest bounds
// reference those block arguments. The `teams` leaf also carries omp.combined;
// the `parallel` leaf and the two loop wrappers carry omp.composite.
void target_teams_distribute_parallel_for() {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute_parallel_for
// CIR-HOST: %[[LB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[UB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[STEP:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%[[LB]] -> %[[ALB:.*]], %[[UB]] -> %[[AUB:.*]], %[[STEP]] -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-HOST: omp.teams {
// CIR-HOST: omp.parallel {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.wsloop {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ALB]]) to (%[[AUB]]) step (%[[ASTEP]]) {
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: } {omp.composite}
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute_parallel_for
// CIR-DEVICE: omp.target kernel_type(spmd) host_eval(%{{.*}} -> %[[ALB:.*]], %{{.*}} -> %[[AUB:.*]], %{{.*}} -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: omp.parallel {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.wsloop {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[ALB]]) to (%[[AUB]]) step (%[[ASTEP]]) {
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: }
