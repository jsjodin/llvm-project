// merk.aggregate (design document, Lowering, "Collecting solutions"): it
// prints as it is parsed, and outlining makes its step a continuation that
// keeps the accumulators in an environment in the caller's frame, and its
// generator a procedure that gives its solutions to it.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @ag attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @ag.digit_1_0() -> i64 attributes {determinism = #merk.determinism<multi>} {
    merk.disj {
      %three = merk.constant 3 : i64
      merk.succeed %three : i64
    }, {
      %one = merk.constant 1 : i64
      merk.succeed %one : i64
    }, {
      %two = merk.constant 2 : i64
      merk.succeed %two : i64
    }
  }

  // The sum of the solutions, each times a value from outside, and their
  // count: two accumulators, a step that always goes on.
  // CHECK-LABEL: merk.proc @ag.sum_3_0(%arg0: i64) -> (i64, i64)
  // CHECK-NEXT: %[[ZERO:.*]] = merk.constant 0 : i64
  // CHECK-NEXT: %[[R:.*]]:2 = merk.aggregate(%[[ZERO]], %[[ZERO]] : i64, i64) -> i64, i64 {
  // CHECK-NEXT: merk.nondet_call @ag.digit_1_0() {
  // CHECK-NEXT: ^bb0(%[[X:.*]]: i64):
  // CHECK-NEXT: merk.solution %[[X]] : i64
  // CHECK-NEXT: }
  // CHECK-NEXT: } step {
  // CHECK-NEXT: ^bb0(%[[Y:.*]]: i64, %[[S:.*]]: i64, %[[N:.*]]: i64):
  // CHECK: merk.yield %{{.*}}, %{{.*}}, %{{.*}} : i1, i64, i64
  // CHECK-NEXT: }
  // CHECK-NEXT: merk.succeed %[[R]]#0, %[[R]]#1 : i64, i64
  merk.proc @ag.sum_3_0(%k: i64) -> (i64, i64) attributes {determinism = #merk.determinism<det>} {
    %zero = merk.constant 0 : i64
    %r:2 = merk.aggregate(%zero, %zero : i64, i64) -> i64, i64 {
      merk.nondet_call @ag.digit_1_0() {
      ^bb0(%x: i64):
        merk.solution %x : i64
      }
    } step {
    ^bb0(%y: i64, %s: i64, %n: i64):
      %go = arith.constant true
      %ky = arith.muli %k, %y : i64
      %s1 = arith.addi %s, %ky : i64
      %one = merk.constant 1 : i64
      %n1 = arith.addi %n, %one : i64
      merk.yield %go, %s1, %n1 : i1, i64, i64
    }
    merk.succeed %r#0, %r#1 : i64, i64
  }

  // The first solution that is at least a value from outside: a step that
  // stops.
  // CHECK-LABEL: merk.proc @ag.until_2_0(%arg0: i64) -> i64
  // CHECK: merk.yield %{{.*}}, %{{.*}} : i1, i64
  merk.proc @ag.until_2_0(%min: i64) -> i64 attributes {determinism = #merk.determinism<det>} {
    %zero = merk.constant 0 : i64
    %r = merk.aggregate(%zero : i64) -> i64 {
      merk.nondet_call @ag.digit_1_0() {
      ^bb0(%x: i64):
        merk.solution %x : i64
      }
    } step {
    ^bb0(%y: i64, %last: i64):
      %small = arith.cmpi slt, %y, %min : i64
      merk.yield %small, %y : i1, i64
    }
    merk.succeed %r : i64
  }
}

// The environment, in the caller's frame, holds the value from outside and
// then the accumulators, which start as the operands; the generator is
// called in the condition of an if-then-else, and either way the results
// are read from their slots after it.
// LLVM-LABEL: define tailcc { i64, i64 } @ag.sum_3_0(i64 %0)
// LLVM-NEXT: %[[ENV:.*]] = alloca { i64, i64, i64 }
// LLVM: %[[K:.*]] = getelementptr { i64, i64, i64 }, ptr %[[ENV]], i32 0, i32 0
// LLVM-NEXT: store i64 %0, ptr %[[K]]
// LLVM-NEXT: %[[S0:.*]] = getelementptr { i64, i64, i64 }, ptr %[[ENV]], i32 0, i32 1
// LLVM-NEXT: store i64 0, ptr %[[S0]]
// LLVM-NEXT: %[[N0:.*]] = getelementptr { i64, i64, i64 }, ptr %[[ENV]], i32 0, i32 2
// LLVM-NEXT: store i64 0, ptr %[[N0]]
// LLVM: call tailcc i1 @ag.sum_3_0.cont.1(ptr %[[ENV]])
// LLVM: %[[S:.*]] = getelementptr { i64, i64, i64 }, ptr %[[ENV]], i32 0, i32 1
// LLVM-NEXT: load i64, ptr %[[S]]
// LLVM-NEXT: %[[N:.*]] = getelementptr { i64, i64, i64 }, ptr %[[ENV]], i32 0, i32 2
// LLVM-NEXT: load i64, ptr %[[N]]
// The generator gives its solutions to the step.
// LLVM-LABEL: define internal tailcc i1 @ag.sum_3_0.cont.1(ptr %0)
// LLVM: musttail call tailcc i1 @ag.digit_1_0(ptr @ag.sum_3_0.cont.0, ptr %0)
// The step reads the accumulators from their slots and writes the new ones
// back; it asks to stop if it says not to go on.
// LLVM-LABEL: define internal tailcc i1 @ag.sum_3_0.cont.0(ptr %0, i64 %1)
// LLVM: getelementptr { i64, i64, i64 }, ptr %0, i32 0, i32 1
// LLVM-NEXT: %[[S:.*]] = load i64
// LLVM-NEXT: getelementptr { i64, i64, i64 }, ptr %0, i32 0, i32 2
// LLVM-NEXT: %[[N:.*]] = load i64
// LLVM: %[[S1:.*]] = add i64 %[[S]],
// LLVM-NEXT: %[[N1:.*]] = add i64 %[[N]], 1
// LLVM-NEXT: %[[SS:.*]] = getelementptr { i64, i64, i64 }, ptr %0, i32 0, i32 1
// LLVM-NEXT: store i64 %[[S1]], ptr %[[SS]]
// LLVM-NEXT: %[[NS:.*]] = getelementptr { i64, i64, i64 }, ptr %0, i32 0, i32 2
// LLVM-NEXT: store i64 %[[N1]], ptr %[[NS]]
// LLVM-NEXT: br i1 false,
// LLVM-LABEL: define internal tailcc i1 @ag.until_2_0.cont.0(ptr %0, i64 %1)
// LLVM: %[[SMALL:.*]] = icmp slt i64 %1,
// LLVM: store i64 %1, ptr
// LLVM-NEXT: %[[STOP:.*]] = xor i1 %[[SMALL]], true
// LLVM-NEXT: br i1 %[[STOP]],
