// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void during(int);

// A `distribute` construct closely nested in a `teams` region lowers to an
// omp.distribute + omp.loop_nest inside the omp.teams, sharing the canonical
// loop lowering with the worksharing loop.
void standalone_distribute() {
  // CHECK: cir.func{{.*}}@standalone_distribute
#pragma omp teams
  {
#pragma omp distribute
    for (int i = 0; i < 10; i++) {
      during(i);
    }
  }

  // CHECK: omp.teams {

  // The induction variable alloca is emitted before the distribute op.
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>

  // CHECK: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[C0:.*]] = cir.builtin_int_cast %[[C0_CIR]] : !s32i -> i32
  // CHECK: %[[C10:.*]] = cir.builtin_int_cast %[[C10_CIR]] : !s32i -> i32
  // CHECK: %[[C1:.*]] = cir.builtin_int_cast %[[C1_CIR]] : !s32i -> i32

  // CHECK: omp.distribute {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// The combined `teams distribute` directive decomposes into a `teams` leaf and
// a `distribute` leaf, lowering to an omp.distribute nested inside an omp.teams.
// The `teams` leaf is not the innermost leaf, so it carries the omp.combined
// marker (unlike the standalone `teams` above).
void teams_distribute() {
  // CHECK: cir.func{{.*}}@teams_distribute
#pragma omp teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }

  // CHECK: omp.teams {
  // CHECK: omp.distribute {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
  // CHECK: cir.call @{{.*}}during
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: } {omp.combined}
}
