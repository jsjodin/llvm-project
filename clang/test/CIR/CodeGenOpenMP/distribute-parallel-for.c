// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);

// `distribute parallel for` is a composite construct. It lowers to a composite
// omp.parallel wrapping the composite omp.distribute + omp.wsloop stack around a
// single shared omp.loop_nest. Each loop wrapper (plus the enclosing parallel)
// carries the omp.composite marker required by the verifier. Here it is closely
// nested in a standalone `teams` region, so the omp.teams is not marked.
void distribute_parallel_for() {
  // CHECK: cir.func{{.*}}@distribute_parallel_for
#pragma omp teams
  {
#pragma omp distribute parallel for
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }

  // CHECK: omp.teams {
  // CHECK: omp.parallel {

  // The induction variable alloca and loop bounds are emitted inside the
  // composite omp.parallel, before the omp.distribute (they are CIR ops, so the
  // composite-parallel verifier permits them alongside the sole omp.distribute).
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[C0:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CHECK: %[[C10:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CHECK: %[[C1:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32

  // CHECK: omp.distribute {
  // CHECK-NEXT: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: } {omp.composite}
  // CHECK: } {omp.composite}
  // CHECK: omp.terminator
  // CHECK: } {omp.composite}
  // CHECK: omp.terminator
  // CHECK: }
}

// The combined `teams distribute parallel for` directive nests the composite
// `distribute parallel for` stack inside an omp.teams. The `teams` leaf is not
// the innermost leaf, so it carries the omp.combined marker, while the loop
// wrappers and their enclosing omp.parallel are marked omp.composite.
void teams_distribute_parallel_for() {
  // CHECK: cir.func{{.*}}@teams_distribute_parallel_for
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CHECK: omp.teams {
  // CHECK: omp.parallel {
  // CHECK: omp.distribute {
  // CHECK-NEXT: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: } {omp.composite}
  // CHECK: } {omp.composite}
  // CHECK: omp.terminator
  // CHECK: } {omp.composite}
  // CHECK: omp.terminator
  // CHECK: } {omp.combined}
}
