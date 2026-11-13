// Where FreeAtLastUse (Frees.cpp) frees a GPU process's cells and arrays:
// after the last use of what lives with them, in each branch where that is
// an if-then-else, and before a tail call; not what is an argument's,
// returned, or given to a foreign call that may keep it, nor what a tail
// call is given. A received message, in its message(...) cell, is freed
// through the cell. A callee's summary says whether its result is a new
// block, and whether it keeps an argument.
//
// RUN: merkc -fc1 -fdebug-dump-frees %s -o - | FileCheck %s

module @frees attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "frees.msg" [#merk.constructor<"data"(i64, !merk.array<i64>)>]
  merk.type_def "frees.received" [#merk.constructor<"message"(!merk.term<"frees.msg">)>, #merk.constructor<"timeout">]
  merk.type_def "frees.pair" [#merk.constructor<"pair"(i64, !merk.array<i64>)>]
  merk.type_def "frees.wrap" [#merk.constructor<"wrap"(!merk.array<i64>)>]

  // An array made and read is freed after its last read, before the tail
  // call of the next state, which it is not given.
  // CHECK-LABEL: merk.proc @frees.state_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK-NEXT: %[[S:.*]] = merk.array_size %[[A]]
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK-NEXT: merk.call @frees.state_1_0(%[[S]])
  merk.proc @frees.state_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.array_size %1 : <i64>
    merk.call @frees.state_1_0(%2) : (i64) -> ()
    merk.succeed
  }

  // What a tail call is given is not freed: the call's return would no
  // longer end the procedure.
  // CHECK-LABEL: merk.proc @frees.pass_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.pass_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.call @frees.reads_1_0(%1) : (!merk.array<i64>) -> ()
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.reads_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.reads_1_0(%arg0: !merk.array<i64>) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.array_size %arg0 : <i64>
    merk.succeed
  }

  // Returned, kept by a foreign call the pass does not know, or an
  // argument's: not freed.
  // CHECK-LABEL: merk.proc @frees.kept_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.kept_1_0(%arg0: !merk.array<i64>) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 4 : i64
    %1 = merk.array_new %0, %0 : i64 -> <i64>
    merk.foreign_call @keep(%1) : (!merk.array<i64>) -> ()
    %2 = merk.array_new %0, %0 : i64 -> <i64>
    %3 = merk.construct #merk.functor<"pair", 2>(%0, %arg0 : i64, !merk.array<i64>) -> !merk.term<"frees.pair">
    %4:2 = merk.deconstruct #merk.functor<"pair", 2> %3 : !merk.term<"frees.pair"> -> i64, !merk.array<i64>
    merk.succeed %2 : !merk.array<i64>
  }

  // Used in the then part only: freed there after its last use, and at the
  // start of the else part.
  // CHECK-LABEL: merk.proc @frees.branches_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK: } then {
  // CHECK-NEXT: merk.array_size %[[A]]
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK-NEXT: merk.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK-NEXT: merk.yield
  merk.proc @frees.branches_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.ite {
      %2 = arith.cmpi sgt, %arg0, %0 : i64
      merk.guard %2
      merk.yield
    } then {
      %2 = merk.array_size %1 : <i64>
      merk.yield
    } else {
      merk.yield
    }
    merk.succeed
  }

  // A received message, with its array: the message is freed through its
  // message(...) cell, where it is one, after the last use of its array,
  // and then the cell.
  // CHECK-LABEL: merk.proc @frees.receive_1_0
  // CHECK: %[[R:.*]] = merk.ite -> !merk.term<"frees.received">
  // CHECK: merk.array_size
  // CHECK-NEXT: merk.ite {
  // CHECK-NEXT: merk.has_functor #merk.functor<"message", 1> %[[R]]
  // CHECK-NEXT: merk.guard
  // CHECK-NEXT: %[[F:.*]] = merk.deconstruct #merk.functor<"message", 1> %[[R]]
  // CHECK-NEXT: merk.yield %[[F]]
  // CHECK-NEXT: } then {
  // CHECK-NEXT: ^bb0(%[[M:.*]]: !merk.term<"frees.msg">):
  // CHECK-NEXT: merk.free %[[M]] : !merk.term<"frees.msg">
  // CHECK: merk.free %[[R]] : !merk.term<"frees.received">
  merk.proc @frees.receive_1_0(%arg0: !merk.term<"frees.msg">) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant -1 : i64
    %1 = merk.type_info "frees.msg/0"
    %2 = builtin.unrealized_conversion_cast %1 : !merk.type_info to !llvm.ptr
    %3:2 = merk.foreign_call @merk_device_receive_term(%arg0, %0, %2) can_fail : (!merk.term<"frees.msg">, i64, !llvm.ptr) -> (i1, !merk.word)
    %4 = merk.ite -> !merk.term<"frees.received"> {
      merk.guard %3#0
      merk.yield
    } then {
      %5 = merk.from_word %3#1 : !merk.term<"frees.msg">
      %6 = merk.construct #merk.functor<"message", 1>(%5 : !merk.term<"frees.msg">) -> !merk.term<"frees.received">
      merk.yield %6 : !merk.term<"frees.received">
    } else {
      %5 = merk.construct #merk.functor<"timeout", 0> -> !merk.term<"frees.received">
      merk.yield %5 : !merk.term<"frees.received">
    }
    merk.ite {
      %5 = merk.has_functor #merk.functor<"message", 1> %4 : !merk.term<"frees.received">
      merk.guard %5
      %6 = merk.deconstruct #merk.functor<"message", 1> %4 : !merk.term<"frees.received"> -> !merk.term<"frees.msg">
      %7:2 = merk.deconstruct #merk.functor<"data", 2> %6 : !merk.term<"frees.msg"> -> i64, !merk.array<i64>
      merk.yield %7#1 : !merk.array<i64>
    } then {
    ^bb0(%arg1: !merk.array<i64>):
      %5 = merk.array_size %arg1 : <i64>
      merk.yield
    } else {
      merk.yield
    }
    merk.succeed
  }

  // A callee's new array is the caller's to free; an array that the callee
  // updates and returns is the same block as the one it was given, freed
  // once.
  // CHECK-LABEL: merk.proc @frees.calls_1_0
  // CHECK: %[[N:.*]] = merk.call @frees.make_1_0
  // CHECK-NEXT: %[[U:.*]] = merk.call @frees.update_1_0(%[[N]])
  // CHECK-NEXT: merk.array_size %[[U]]
  // CHECK-NEXT: merk.free %[[N]] : !merk.array<i64>
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.calls_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.call @frees.make_1_0(%arg0) : (i64) -> !merk.array<i64>
    %1 = merk.call @frees.update_1_0(%0) : (!merk.array<i64>) -> !merk.array<i64>
    %2 = merk.array_size %1 : <i64>
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.make_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.make_1_0(%arg0: i64) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.succeed %1 : !merk.array<i64>
  }

  // CHECK-LABEL: merk.proc @frees.update_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.update_1_0(%arg0: !merk.array<i64>) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_set %arg0[%0], %0 : <i64>, i64
    merk.succeed %1 : !merk.array<i64>
  }

  // An array given to a callee that keeps it is not freed.
  // CHECK-LABEL: merk.proc @frees.gives_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.gives_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.call @frees.make_1_0(%arg0) : (i64) -> !merk.array<i64>
    merk.call @frees.keeps_1_0(%0) : (!merk.array<i64>) -> ()
    %1 = merk.array_size %0 : <i64>
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.keeps_1_0
  merk.proc @frees.keeps_1_0(%arg0: !merk.array<i64>) attributes {determinism = #merk.determinism<det>} {
    merk.foreign_call @keep(%arg0) : (!merk.array<i64>) -> ()
    merk.succeed
  }

  // An array whose word a loop across the process's teams reads is freed
  // after the loop's call, which returns when the loop has ended.
  // CHECK-LABEL: merk.proc @frees.loop_1_0
  // CHECK: merk.foreign_call @merk_device_process_loop
  // CHECK-NEXT: merk.free %{{.*}} : !merk.array<i64>
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.loop_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.foreign_call @merk_device_loop_values() : () -> !llvm.ptr
    %3 = merk.to_word %1 : !merk.array<i64>
    %4 = merk.from_word %3 : i64
    %5 = llvm.getelementptr %2[0] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %4, %5 : i64, !llvm.ptr
    merk.foreign_call @merk_device_process_loop(%0, %0, %0, %arg0, %0) : (i64, i64, i64, i64, i64) -> ()
    merk.succeed
  }

  // A part that fails gives the result nothing: the other part's new
  // block is the result's.
  // CHECK-LABEL: merk.proc @frees.fails_1_0
  // CHECK: %[[R:.*]] = merk.switch
  // CHECK: merk.fail
  // CHECK: merk.array_size %[[R]]
  // CHECK-NEXT: merk.free %[[R]] : !merk.array<i64>
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.fails_1_0(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
    %0 = merk.constant 0 : i64
    %1 = merk.switch %arg0 : i64 -> !merk.array<i64> cases [[0], [1]] {
      %3 = merk.array_new %arg0, %0 : i64 -> <i64>
      merk.yield %3 : !merk.array<i64>
    }, {
      merk.fail
    }
    %2 = merk.array_size %1 : <i64>
    merk.succeed %2 : i64
  }

  // A term of a type whose one constructor has one field is that field,
  // with no cell: the array is freed once.
  // CHECK-LABEL: merk.proc @frees.wrapped_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK: merk.array_size
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.wrapped_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.construct #merk.functor<"wrap", 1>(%1 : !merk.array<i64>) -> !merk.term<"frees.wrap">
    %3 = merk.deconstruct #merk.functor<"wrap", 1> %2 : !merk.term<"frees.wrap"> -> !merk.array<i64>
    %4 = merk.array_size %3 : <i64>
    merk.succeed
  }

  // The new block of a branch's result, from each part, is freed by the
  // result, once.
  // CHECK-LABEL: merk.proc @frees.made_1_0
  // CHECK: %[[R:.*]] = merk.ite -> !merk.array<i64>
  // CHECK: merk.array_size %[[R]]
  // CHECK-NEXT: merk.free %[[R]] : !merk.array<i64>
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.made_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.ite -> !merk.array<i64> {
      %2 = arith.cmpi sgt, %arg0, %0 : i64
      merk.guard %2
      merk.yield
    } then {
      %2 = merk.array_new %arg0, %0 : i64 -> <i64>
      merk.yield %2 : !merk.array<i64>
    } else {
      %2 = merk.array_new %0, %0 : i64 -> <i64>
      merk.yield %2 : !merk.array<i64>
    }
    %2 = merk.array_size %1 : <i64>
    merk.succeed
  }
}
