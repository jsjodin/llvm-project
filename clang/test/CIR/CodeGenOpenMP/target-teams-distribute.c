// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU).
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void during(int);

// The combined 'target teams distribute' directive decomposes into 'target',
// 'teams' and 'distribute' leaves and lowers to an omp.distribute +
// omp.loop_nest inside omp.teams inside omp.target. Although this construct runs
// with the generic kernel_type, its distribute trip count must still be
// evaluated on the host and forwarded through host_eval block arguments (see
// TargetOp::hasHostEvalTripCount), so the omp.loop_nest bounds reference those
// block arguments. Both the 'target' and (non-innermost) 'teams' leaves carry
// the omp.combined marker.
void target_teams_distribute() {
#pragma omp target teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute
// CIR-HOST: %[[LB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[UB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[STEP:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: omp.target kernel_type(generic) host_eval(%[[LB]] -> %[[ALB:.*]], %[[UB]] -> %[[AUB:.*]], %[[STEP]] -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-HOST: omp.teams {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ALB]]) to (%[[AUB]]) step (%[[ASTEP]]) {
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: }
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute
// CIR-DEVICE: omp.target kernel_type(generic) host_eval(%{{.*}} -> %[[ALB:.*]], %{{.*}} -> %[[AUB:.*]], %{{.*}} -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[ALB]]) to (%[[AUB]]) step (%[[ASTEP]]) {
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: }
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
