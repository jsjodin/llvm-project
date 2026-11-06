// Nondeterministic code (design document, Lowering, "Nondeterministic
// code"): the operations print as they are parsed, in regions and after
// outlining, and lower to procedures that take continuations.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @nd attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // A disjunction in merk.each gives its solutions to the continuation.
  // CHECK-LABEL: merk.proc @nd.digit_1_0() -> i64 attributes {determinism = #merk.determinism<multi>} {
  // CHECK-NEXT: merk.each {
  // CHECK-NEXT: merk.disj {
  // CHECK: merk.solution %{{.*}} : i64
  // CHECK-NEXT: }, {
  // CHECK: merk.solution %{{.*}} : i64
  // CHECK-NEXT: }
  // CHECK-NEXT: } then {
  // CHECK-NEXT: ^bb0(%[[X:.*]]: i64):
  // CHECK-NEXT: merk.succeed %[[X]] : i64
  merk.proc @nd.digit_1_0() -> i64 attributes {determinism = #merk.determinism<multi>} {
    merk.each {
      merk.disj {
        %one = merk.constant 1 : i64
        merk.solution %one : i64
      }, {
        %two = merk.constant 2 : i64
        merk.solution %two : i64
      }
    } then {
    ^bb0(%x: i64):
      merk.succeed %x : i64
    }
  }

  // The first solution of a nondet call, whose continuation uses a value
  // from before it.
  // CHECK-LABEL: merk.proc @nd.over_1_0(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<cc_nondet>} {
  // CHECK-NEXT: %[[FIRST:.*]] = merk.commit -> i64 {
  // CHECK-NEXT: merk.nondet_call @nd.digit_1_0() {
  // CHECK-NEXT: ^bb0(%[[X:.*]]: i64):
  // CHECK-NEXT: %[[GT:.*]] = arith.cmpi sgt, %[[X]], %arg0 : i64
  // CHECK-NEXT: merk.guard %[[GT]]
  // CHECK-NEXT: merk.solution %[[X]] : i64
  // CHECK: merk.succeed %[[FIRST]] : i64
  merk.proc @nd.over_1_0(%min: i64) -> i64 attributes {determinism = #merk.determinism<cc_nondet>} {
    %first = merk.commit -> i64 {
      merk.nondet_call @nd.digit_1_0() {
      ^bb0(%x: i64):
        %gt = arith.cmpi sgt, %x, %min : i64
        merk.guard %gt
        merk.solution %x : i64
      }
    }
    merk.succeed %first : i64
  }

  // The operations that outlining makes.
  // CHECK-LABEL: merk.proc @nd.give_0_0(%arg0: !merk.cont, %arg1: !merk.env)
  // CHECK-NEXT: %[[ONE:.*]] = merk.constant 1 : i64
  // CHECK-NEXT: merk.call_cont %arg0(%arg1, %[[ONE]]) : i64
  // CHECK-NEXT: merk.succeed
  merk.proc @nd.give_0_0(%k: !merk.cont, %e: !merk.env) attributes {determinism = #merk.determinism<semidet>} {
    %one = merk.constant 1 : i64
    merk.call_cont %k(%e, %one) : i64
    merk.succeed
  }
  // CHECK-LABEL: merk.proc @nd.store_0_0(%arg0: !merk.env, %arg1: i64)
  // CHECK-NEXT: merk.env_set %arg0[1], %arg1 : i64 layout [i64, i64]
  merk.proc @nd.store_0_0(%e: !merk.env, %x: i64) attributes {determinism = #merk.determinism<semidet>} {
    merk.env_set %e[1], %x : i64 layout [i64, i64]
    merk.succeed
  }
  // CHECK-LABEL: merk.proc @nd.first_1_0(%arg0: i64) -> i64
  // CHECK-NEXT: %[[ENV:.*]] = merk.env_new(%arg0 : i64) layout [i64, i64]
  // CHECK-NEXT: %[[K:.*]] = merk.cont_ref @nd.store_0_0
  // CHECK-NEXT: merk.call @nd.give_0_0(%[[K]], %[[ENV]]) : (!merk.cont, !merk.env) -> ()
  // CHECK-NEXT: %[[X:.*]] = merk.env_get %[[ENV]][1] : i64 layout [i64, i64]
  // CHECK-NEXT: merk.succeed %[[X]] : i64
  merk.proc @nd.first_1_0(%y: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
    %e = merk.env_new(%y : i64) layout [i64, i64]
    %k = merk.cont_ref @nd.store_0_0
    merk.call @nd.give_0_0(%k, %e) : (!merk.cont, !merk.env) -> ()
    %x = merk.env_get %e[1] : i64 layout [i64, i64]
    merk.succeed %x : i64
  }
}

// After outlining, a multi procedure takes a continuation and its
// environment, and returns whether the continuation asked it to stop; a
// branch whose continuation does not stop goes to the next.
// LLVM-LABEL: define tailcc i1 @nd.digit_1_0(ptr %0, ptr %1)
// LLVM: %[[STOP:.*]] = call tailcc i1 %0(ptr %1, i64 1)
// LLVM-NEXT: br i1 %[[STOP]], label %{{.*}}, label %[[NEXT:.*]]
// LLVM: [[NEXT]]:
// LLVM-NEXT: musttail call tailcc i1 %0(ptr %1, i64 2)
// The commit's generator stores its first solution in an environment in the
// caller's frame, and the continuation of the nondet call reads the value
// it uses from its own.
// LLVM-LABEL: define tailcc { i1, i64 } @nd.over_1_0(i64 %0)
// LLVM: %[[ENV:.*]] = alloca { i64, i64 }
// LLVM: call tailcc i1 @nd.over_1_0.cont.0(ptr %[[ENV]])
// LLVM-LABEL: define internal tailcc i1 @nd.over_1_0.cont.0(ptr %0)
// LLVM: call tailcc i1 @nd.digit_1_0(ptr @nd.over_1_0.cont.0.cont.0, ptr %{{.*}})
// LLVM-LABEL: define internal tailcc i1 @nd.over_1_0.cont.0.cont.0(ptr %0, i64 %1)
// LLVM: icmp sgt i64 %1,
// A call that passes on the continuation it was given is a tail call; one
// that passes an environment in its own frame is not.
// LLVM-LABEL: define tailcc i1 @nd.give_0_0(ptr %0, ptr %1)
// LLVM-NEXT: musttail call tailcc i1 %0(ptr %1, i64 1)
// LLVM-LABEL: define tailcc { i1, i64 } @nd.first_1_0(i64 %0)
// LLVM: %[[ENV:.*]] = alloca { i64, i64 }
// LLVM: call tailcc i1 @nd.give_0_0(ptr @nd.store_0_0, ptr %[[ENV]])
