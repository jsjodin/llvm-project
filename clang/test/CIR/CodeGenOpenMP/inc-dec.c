// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

// Pre/post increment and decrement, and compound assignment, are ordinary
// scalar updates under OpenMP. (The only OpenMP-specific behavior is the
// lastprivate(conditional:) runtime notification, which is not yet supported.)

void inc_dec(void) {
  int x = 0;
  // CHECK-LABEL: cir.func{{.*}}@inc_dec
  // CHECK: omp.parallel {
  // CHECK: %[[V0:.*]] = cir.load {{.*}} %[[X:.*]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[INC:.*]] = cir.inc nsw %[[V0]] : !s32i
  // CHECK: cir.store {{.*}} %[[INC]], %[[X]]
  // CHECK: %[[V1:.*]] = cir.load {{.*}} %[[X]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[DEC:.*]] = cir.dec nsw %[[V1]] : !s32i
  // CHECK: cir.store {{.*}} %[[DEC]], %[[X]]
  // CHECK: %[[C3:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK: %[[V2:.*]] = cir.load {{.*}} %[[X]] : !cir.ptr<!s32i>, !s32i
  // CHECK: %[[ADD:.*]] = cir.add nsw %[[V2]], %[[C3]] : !s32i
  // CHECK: cir.store {{.*}} %[[ADD]], %[[X]]
  // CHECK: omp.terminator
#pragma omp parallel
  {
    x++;
    --x;
    x += 3;
  }
}
