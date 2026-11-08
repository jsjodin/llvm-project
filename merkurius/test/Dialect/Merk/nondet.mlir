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

  // An if-then-else whose condition has more than one solution: the then
  // part runs for each, and the else part only if there are none.
  // CHECK-LABEL: merk.proc @nd.big_1_0(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<multi>} {
  // CHECK-NEXT: merk.soft_ite {
  // CHECK-NEXT: merk.nondet_call @nd.digit_1_0() {
  // CHECK: merk.solution %{{.*}} : i64
  // CHECK: } then {
  // CHECK-NEXT: ^bb0(%[[X:.*]]: i64):
  // CHECK-NEXT: merk.succeed %[[X]] : i64
  // CHECK-NEXT: } else {
  // CHECK-NEXT: %[[ZERO:.*]] = merk.constant 0 : i64
  // CHECK-NEXT: merk.succeed %[[ZERO]] : i64
  merk.proc @nd.big_1_0(%min: i64) -> i64 attributes {determinism = #merk.determinism<multi>} {
    merk.soft_ite {
      merk.nondet_call @nd.digit_1_0() {
      ^bb0(%x: i64):
        %gt = arith.cmpi sgt, %x, %min : i64
        merk.guard %gt
        merk.solution %x : i64
      }
    } then {
    ^bb0(%x: i64):
      merk.succeed %x : i64
    } else {
      %zero = merk.constant 0 : i64
      merk.succeed %zero : i64
    }
  }

  // A higher-order call of a nondet closure, whose continuation only gives
  // its solution, passes on the caller's continuation.
  // CHECK-LABEL: merk.proc @nd.call_1_0(%arg0: !merk.closure, %arg1: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
  // CHECK-NEXT: merk.nondet_call_closure %arg0(%arg1 : !merk.word) {
  // CHECK-NEXT: ^bb0(%[[Y:.*]]: !merk.word):
  // CHECK-NEXT: merk.succeed %[[Y]] : !merk.word
  merk.proc @nd.call_1_0(%p: !merk.closure, %x: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
    merk.nondet_call_closure %p(%x : !merk.word) {
    ^bb0(%y: !merk.word):
      merk.succeed %y : !merk.word
    }
  }

  // A call of a nondet method, whose continuation only gives its solution,
  // passes on the caller's continuation to the method's procedure in the
  // instance's table.
  // CHECK-LABEL: merk.proc @nd.method_1_0(%arg0: !merk.typeclass_info, %arg1: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
  // CHECK-NEXT: merk.nondet_call_method %arg0[1] (%arg1 : !merk.word) {
  // CHECK-NEXT: ^bb0(%[[Y:.*]]: !merk.word):
  // CHECK-NEXT: merk.succeed %[[Y]] : !merk.word
  merk.proc @nd.method_1_0(%d: !merk.typeclass_info, %x: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<nondet>} {
    merk.nondet_call_method %d[1](%x : !merk.word) {
    ^bb0(%y: !merk.word):
      merk.succeed %y : !merk.word
    }
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
// The then part of the soft cut sets the slot after the values it uses on
// entry; the else part runs when the condition has no more solutions, if
// the slot is not set.
// LLVM-LABEL: define tailcc i1 @nd.big_1_0(i64 %0, ptr %1, ptr %2)
// LLVM: %[[THEN:.*]] = alloca { ptr, ptr, i1 }
// LLVM: %[[FOUND:.*]] = getelementptr { ptr, ptr, i1 }, ptr %[[THEN]], i32 0, i32 2
// LLVM-NEXT: store i1 false, ptr %[[FOUND]]
// LLVM: call tailcc i1 @nd.digit_1_0(ptr @nd.big_1_0.cont.1, ptr %{{.*}})
// LLVM: load i1, ptr
// LLVM-NEXT: xor i1 %{{.*}}, true
// LLVM: musttail call tailcc i1 %1(ptr %2, i64 0)
// LLVM-LABEL: define internal tailcc i1 @nd.big_1_0.cont.0(ptr %0, i64 %1)
// LLVM: %[[SET:.*]] = getelementptr { ptr, ptr, i1 }, ptr %0, i32 0, i32 2
// LLVM-NEXT: store i1 true, ptr %[[SET]]
// LLVM-LABEL: define tailcc i1 @nd.call_1_0(ptr addrspace(1) %0, i64 %1, ptr %2, ptr %3)
// LLVM: %[[CODE:.*]] = load ptr, ptr addrspace(1) %0
// LLVM-NEXT: musttail call tailcc i1 %[[CODE]](ptr addrspace(1) %0, i64 %1, ptr %2, ptr %3)
// A call that passes on the continuation it was given is a tail call; one
// that passes an environment in its own frame is not.
// LLVM-LABEL: define tailcc i1 @nd.method_1_0(ptr addrspace(1) %0, i64 %1, ptr %2, ptr %3)
// LLVM: %[[TABLE:.*]] = load ptr, ptr addrspace(1) %0
// LLVM-NEXT: %[[SLOT:.*]] = getelementptr i8, ptr %[[TABLE]], i32 16
// LLVM-NEXT: %[[METHOD:.*]] = load ptr, ptr %[[SLOT]]
// LLVM-NEXT: musttail call tailcc i1 %[[METHOD]](ptr addrspace(1) %0, i64 %1, ptr %2, ptr %3)
// LLVM-LABEL: define tailcc i1 @nd.give_0_0(ptr %0, ptr %1)
// LLVM: musttail call tailcc i1 %0(ptr %1, i64 1)
// LLVM-LABEL: define tailcc { i1, i64 } @nd.first_1_0(i64 %0)
// LLVM: %[[ENV:.*]] = alloca { i64, i64 }
// LLVM: call tailcc i1 @nd.give_0_0(ptr @nd.store_0_0, ptr %[[ENV]])
