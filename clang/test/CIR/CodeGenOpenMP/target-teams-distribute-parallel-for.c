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
// omp.target. The `teams` leaf carries omp.combined; the `parallel` leaf and the
// two loop wrappers carry omp.composite. The omp.target stays generic and
// unmarked (its combined/SPMD metadata is deferred with host_eval), so the loop
// trip count is evaluated in-region.
void target_teams_distribute_parallel_for() {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute_parallel_for
// CIR-HOST: omp.target kernel_type(generic) {
// CIR-HOST: omp.teams {
// CIR-HOST: omp.parallel {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.wsloop {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
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
// CIR-HOST: }

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute_parallel_for
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: omp.parallel {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.wsloop {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
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
