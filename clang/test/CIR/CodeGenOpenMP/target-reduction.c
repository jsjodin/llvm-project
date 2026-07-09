// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU).
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

#define N 256

// A simple built-in `+` reduction on a scalar integer and a scalar float. The
// reduction is carried by the `teams` and (composite) `wsloop` leaves, and each
// leaf rebinds the reduction variable to its own private block argument. The
// `parallel` and `distribute` leaves carry no reduction.
int reduction_add(int *a) {
  int sum = 0;
  float fsum = 0.0f;
#pragma omp target teams distribute parallel for \
    map(tofrom : a[0:N]) map(tofrom : sum, fsum) \
    reduction(+ : sum) reduction(+ : fsum)
  for (int i = 0; i < N; ++i) {
    sum += a[i];
    fsum += (float)a[i];
  }
  return sum;
}

// CIR-HOST-LABEL: cir.func{{.*}}@reduction_add
// The reduction variables are mapped into the target region ...
// CIR-HOST: omp.target
// CIR-HOST-SAME: map_entries({{.*}}%[[SUM_MAP:[0-9a-z_]+]] -> %[[SUM_A0:[0-9a-z_]+]], %[[FSUM_MAP:[0-9a-z_]+]] -> %[[FSUM_A0:[0-9a-z_]+]] : {{.*}}!cir.ptr<!s32i>, !cir.ptr<!cir.float>)
// ... then the teams leaf carries the reduction on the mapped accumulators.
// CIR-HOST: omp.teams reduction(@add_reduction_i32 %[[SUM_A0]] -> %[[SUM_A1:[0-9a-z_]+]], @add_reduction_f32 %[[FSUM_A0]] -> %[[FSUM_A1:[0-9a-z_]+]] : !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
// The parallel and distribute leaves carry no reduction.
// CIR-HOST-NEXT: omp.parallel {
// CIR-HOST: omp.distribute {
// The composite wsloop carries the reduction on the teams-private copies.
// CIR-HOST: omp.wsloop reduction(@add_reduction_i32 %[[SUM_A1]] -> %[[SUM_A2:[0-9a-z_]+]], @add_reduction_f32 %[[FSUM_A1]] -> %[[FSUM_A2:[0-9a-z_]+]] : !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
// CIR-HOST: omp.loop_nest
// The loop body reads and updates the innermost (wsloop-private) copies.
// CIR-HOST: cir.load {{.*}}%[[SUM_A2]]
// CIR-HOST: cir.add
// CIR-HOST: cir.store {{.*}}%[[SUM_A2]]
// CIR-HOST: cir.load {{.*}}%[[FSUM_A2]]
// CIR-HOST: cir.fadd
// CIR-HOST: cir.store {{.*}}%[[FSUM_A2]]
// CIR-HOST: {omp.composite}
// CIR-HOST: {omp.composite}

// The declare_reduction ops implement `+` by value: init yields the additive
// identity and the combiner yields the sum of its two value arguments.
// CIR-HOST: omp.declare_reduction @add_reduction_i32 : !s32i init {
// CIR-HOST: ^bb0(%[[I_MOLD:.*]]: !s32i{{.*}}):
// CIR-HOST:   %[[I_ID:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST:   omp.yield(%[[I_ID]] : !s32i)
// CIR-HOST: } combiner {
// CIR-HOST: ^bb0(%[[I_L:.*]]: !s32i{{.*}}, %[[I_R:.*]]: !s32i{{.*}}):
// CIR-HOST:   %[[I_SUM:.*]] = cir.add %[[I_L]], %[[I_R]] : !s32i
// CIR-HOST:   omp.yield(%[[I_SUM]] : !s32i)
// CIR-HOST: }
// CIR-HOST: omp.declare_reduction @add_reduction_f32 : !cir.float init {
// CIR-HOST: ^bb0(%[[F_MOLD:.*]]: !cir.float{{.*}}):
// CIR-HOST:   %[[F_ID:.*]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR-HOST:   omp.yield(%[[F_ID]] : !cir.float)
// CIR-HOST: } combiner {
// CIR-HOST: ^bb0(%[[F_L:.*]]: !cir.float{{.*}}, %[[F_R:.*]]: !cir.float{{.*}}):
// CIR-HOST:   %[[F_SUM:.*]] = cir.fadd %[[F_L]], %[[F_R]] : !cir.float
// CIR-HOST:   omp.yield(%[[F_SUM]] : !cir.float)
// CIR-HOST: }

// CIR-DEVICE-LABEL: cir.func{{.*}}@reduction_add
// CIR-DEVICE: omp.teams reduction(@add_reduction_i32 {{.*}}, @add_reduction_f32 {{.*}} : !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
// CIR-DEVICE: omp.wsloop reduction(@add_reduction_i32 {{.*}}, @add_reduction_f32 {{.*}} : !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
// CIR-DEVICE: omp.declare_reduction @add_reduction_i32 : !s32i init {
// CIR-DEVICE: omp.declare_reduction @add_reduction_f32 : !cir.float init {
