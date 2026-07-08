// RUN: %clang_cc1 -fopenmp -emit-cir -fclangir %s -o - | FileCheck %s

void before(int);
void during(int);
void after(int);

void emit_simple_for() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_simple_for
  int j = 5;
  before(j);
  // CHECK: cir.call @{{.*}}before
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
        during(j);
    }
  }
  // CHECK: omp.parallel {

  // The induction variable alloca is emitted before the wsloop.
  // CHECK: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>

  // CIR constants for the loop bounds.
  // CHECK: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i

  // Bounds are cast to builtin integers for omp.loop_nest.
  // CHECK: %[[C0:.*]] = cir.builtin_int_cast %[[C0_CIR]] : !s32i -> i32
  // CHECK: %[[C10:.*]] = cir.builtin_int_cast %[[C10_CIR]] : !s32i -> i32
  // CHECK: %[[C1:.*]] = cir.builtin_int_cast %[[C1_CIR]] : !s32i -> i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[C0]]) to (%[[C10]]) step (%[[C1]]) {

  // The induction variable block arg is cast back to a CIR integer and stored.
  // CHECK: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV_CIR]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // during(j)
  // CHECK: cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }
  after(j);
  // CHECK: cir.call @{{.*}}after
}

void emit_for_with_vars() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_with_vars
  int j = 5;
  before(j);
  // CHECK: cir.call @{{.*}}before
#pragma omp parallel
  {
    int lb = 1;
    long ub = 10;
    short step = 1;
#pragma omp for
    for (int i = 0; i < ub; i=i+step) {
        during(j);
    }
  }

  // CHECK: omp.parallel {

  // Local variable allocas.
  // CHECK: %[[LB:.*]] = cir.alloca "lb" align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[UB:.*]] = cir.alloca "ub" align(8) init : !cir.ptr<!s64i>
  // CHECK: %[[STEP:.*]] = cir.alloca "step" align(2) init : !cir.ptr<!s16i>

  // Induction variable alloca (emitted before the wsloop).
  // CHECK: %[[I2_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>

  // The variable bounds are reduced to CIR integers of the loop variable type
  // and then cast to builtin integers for omp.loop_nest.
  // CHECK: %[[LB0:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CHECK: %[[UBSTD:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CHECK: %[[STEPSTD:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV2:.*]]) : i32 = (%[[LB0]]) to (%[[UBSTD]]) step (%[[STEPSTD]]) {

  // store induction variable block arg into alloca
  // CHECK: %[[IV2_CIR:.*]] = cir.builtin_int_cast %[[IV2]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV2_CIR]], %[[I2_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // during(j)
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }

  after(j);
  // CHECK: cir.call @{{.*}}after
}

void emit_for_with_induction_var() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_with_induction_var
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // induction variable alloca
  // CHECK: %[[IV_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>

  // CIR constants
  // CHECK: %[[IC0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[IC10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[IC1_CIR:.*]] = cir.const #cir.int<1> : !s32i

  // conversion to builtin integer
  // CHECK: %[[IC0:.*]] = cir.builtin_int_cast %[[IC0_CIR]] : !s32i -> i32
  // CHECK: %[[IC10:.*]] = cir.builtin_int_cast %[[IC10_CIR]] : !s32i -> i32
  // CHECK: %[[IC1:.*]] = cir.builtin_int_cast %[[IC1_CIR]] : !s32i -> i32

  // omp loop
  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%[[IV3:.*]]) : i32 = (%[[IC0]]) to (%[[IC10]]) step (%[[IC1]]) {

  // store induction variable into alloca
  // CHECK: %[[IV3_CIR:.*]] = cir.builtin_int_cast %[[IV3]] : i32 -> !s32i
  // CHECK: cir.store align(4) %[[IV3_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>

  // during(i) - loads the induction variable from the alloca
  // CHECK: %[[I_VAL:.*]] = cir.load align(4) %[[IV_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.call @{{.*}}during(%[[I_VAL]])

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }

  // CHECK: omp.terminator
  // CHECK: }
}

// Test inclusive upper bound (i <= 9)
void emit_for_inclusive_bound() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_inclusive_bound
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i <= 9; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: %[[INC_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[INC0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[INC9_CIR:.*]] = cir.const #cir.int<9> : !s32i
  // CHECK: %[[INC1_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[INC_C0:.*]] = cir.builtin_int_cast %[[INC0_CIR]] : !s32i -> i32
  // CHECK: %[[INC_C9:.*]] = cir.builtin_int_cast %[[INC9_CIR]] : !s32i -> i32
  // CHECK: %[[INC_C1:.*]] = cir.builtin_int_cast %[[INC1_CIR]] : !s32i -> i32

  // CHECK: omp.wsloop {
  // inclusive = true
  // CHECK-NEXT: omp.loop_nest (%[[INC_IV:.*]]) : i32 = (%[[INC_C0]]) to (%[[INC_C9]]) inclusive step (%[[INC_C1]]) {

  // CHECK: cir.builtin_int_cast %[[INC_IV]] : i32 -> !s32i
  // CHECK: cir.store
  // CHECK: cir.call @{{.*}}during

  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test reversed comparison (10 > i)
void emit_for_reversed_cmp() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_reversed_cmp
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; 10 > i; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: %[[REV_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[REV0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[REV10_CIR:.*]] = cir.const #cir.int<10> : !s32i
  // CHECK: %[[REV1_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[REV_C0:.*]] = cir.builtin_int_cast %[[REV0_CIR]] : !s32i -> i32
  // CHECK: %[[REV_C10:.*]] = cir.builtin_int_cast %[[REV10_CIR]] : !s32i -> i32
  // CHECK: %[[REV_C1:.*]] = cir.builtin_int_cast %[[REV1_CIR]] : !s32i -> i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[REV_C0]]) to (%[[REV_C10]]) step (%[[REV_C1]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test reversed inclusive comparison (9 >= i)
void emit_for_reversed_inclusive_cmp() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_reversed_inclusive_cmp
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; 9 >= i; i++) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: %[[RI_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[RI0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[RI9_CIR:.*]] = cir.const #cir.int<9> : !s32i
  // CHECK: %[[RI1_CIR:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK: %[[RI_C0:.*]] = cir.builtin_int_cast %[[RI0_CIR]] : !s32i -> i32
  // CHECK: %[[RI_C9:.*]] = cir.builtin_int_cast %[[RI9_CIR]] : !s32i -> i32
  // CHECK: %[[RI_C1:.*]] = cir.builtin_int_cast %[[RI1_CIR]] : !s32i -> i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[RI_C0]]) to (%[[RI_C9]]) inclusive step (%[[RI_C1]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test compound assignment step (i += 2)
void emit_for_compound_step() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_compound_step
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 20; i += 2) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // CHECK: %[[CS_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
  // CHECK: %[[CS0_CIR:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: %[[CS20_CIR:.*]] = cir.const #cir.int<20> : !s32i
  // CHECK: %[[CS2_CIR:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK: %[[CS_C0:.*]] = cir.builtin_int_cast %[[CS0_CIR]] : !s32i -> i32
  // CHECK: %[[CS_C20:.*]] = cir.builtin_int_cast %[[CS20_CIR]] : !s32i -> i32
  // CHECK: %[[CS_C2:.*]] = cir.builtin_int_cast %[[CS2_CIR]] : !s32i -> i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[CS_C0]]) to (%[[CS_C20]]) step (%[[CS_C2]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}

// Test commuted step expression (i = step + i)
void emit_for_commuted_step() {
  // CHECK: cir.func{{.*}}@{{.*}}emit_for_commuted_step
  short step = 3;
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 30; i = step + i) {
        during(i);
    }
  }
  // CHECK: omp.parallel {

  // Induction variable alloca (emitted before the wsloop).
  // CHECK: %[[CM_ALLOCA:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>

  // The bounds (with the variable step reduced to the loop variable type) are
  // cast to builtin integers for omp.loop_nest.
  // CHECK: %[[CM_C0:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CHECK: %[[CM_C30:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
  // CHECK: %[[CM_STEP:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32

  // CHECK: omp.wsloop {
  // CHECK-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%[[CM_C0]]) to (%[[CM_C30]]) step (%[[CM_STEP]]) {
  // CHECK: omp.yield
  // CHECK: }
  // CHECK: }
  // CHECK: omp.terminator
  // CHECK: }
}
