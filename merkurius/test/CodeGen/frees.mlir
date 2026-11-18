// Where FreeAtLastUse (Frees.cpp) frees a GPU process's cells and arrays:
// after the last use of what lives with them, in each branch where that is
// an if-then-else, and before a tail call; not what is an argument's,
// returned, or given to a foreign call that may keep it, nor what a tail
// call that may come back to the procedure is given. A received message, in its message(...) cell, is freed
// through the cell. A callee's summary says whether its result is a new
// block, and whether it keeps an argument. A send that is the last use of
// the arrays its message holds moves them; one that is not copies them,
// and they are freed after it. An array taken by mode di is freed by the
// callee when each call gives it one of the caller's own, and a resize that
// clobbers its array frees the old one.
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

  // What a tail call of a procedure that does not call back is given is
  // freed after it: the call is no loop, and its frame costs one.
  // CHECK-LABEL: merk.proc @frees.pass_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK-NEXT: merk.call @frees.reads_1_0(%[[A]])
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.pass_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.call @frees.reads_1_0(%1) : (!merk.array<i64>) -> ()
    merk.succeed
  }

  // What a tail call that may come back to the procedure is given is not
  // freed: the call's return would no longer end the procedure, and each
  // round of the loop would keep a frame.
  // CHECK-LABEL: merk.proc @frees.again_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.again_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.call @frees.round_2_0(%arg0, %1) : (i64, !merk.array<i64>) -> ()
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.round_2_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.round_2_0(%arg0: i64, %arg1: !merk.array<i64>) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.array_size %arg1 : <i64>
    merk.call @frees.again_1_0(%0) : (i64) -> ()
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

  // An array sent as the whole message, through its pointer, as its last
  // use, moves: the send is marked so, and nothing frees the array.
  // CHECK-LABEL: merk.proc @frees.sends_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK-NEXT: %[[P:.*]] = builtin.unrealized_conversion_cast %[[A]]
  // CHECK-NEXT: merk.foreign_call @merk_device_send_array(%arg0, %{{.*}}, %[[P]]) moves
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.sends_1_0(%arg0: !merk.term<"frees.msg">, %arg1: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg1, %0 : i64 -> <i64>
    %2 = builtin.unrealized_conversion_cast %1 : !merk.array<i64> to !llvm.ptr
    merk.foreign_call @merk_device_send_array(%arg0, %0, %2) : (!merk.term<"frees.msg">, i64, !llvm.ptr) -> ()
    merk.succeed
  }

  // An array read after its send is copied by it, and freed after its last
  // read.
  // CHECK-LABEL: merk.proc @frees.sendsRead_1_0
  // CHECK: merk.foreign_call @merk_device_send_array(%arg0, %{{.*}}, %{{.*}}) :
  // CHECK-NEXT: merk.array_size %[[A:.*]] :
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  merk.proc @frees.sendsRead_1_0(%arg0: !merk.term<"frees.msg">, %arg1: i64) -> i64 attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg1, %0 : i64 -> <i64>
    %2 = builtin.unrealized_conversion_cast %1 : !merk.array<i64> to !llvm.ptr
    merk.foreign_call @merk_device_send_array(%arg0, %0, %2) : (!merk.term<"frees.msg">, i64, !llvm.ptr) -> ()
    %3 = merk.array_size %1 : <i64>
    merk.succeed %3 : i64
  }

  // A term sent as its last use moves the array it was constructed with,
  // which is not freed through its cell; the cell, which the send copies,
  // is.
  // CHECK-LABEL: merk.proc @frees.sendsTerm_1_0
  // CHECK: %[[C:.*]] = merk.construct #merk.functor<"data", 2>
  // CHECK: merk.foreign_call @merk_device_send_term(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) moves
  // CHECK-NEXT: merk.free %[[C]] : !merk.term<"frees.msg">
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.sendsTerm_1_0(%arg0: !merk.term<"frees.msg">, %arg1: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg1, %0 : i64 -> <i64>
    %2 = merk.construct #merk.functor<"data", 2>(%arg1, %1 : i64, !merk.array<i64>) -> !merk.term<"frees.msg">
    %3 = merk.to_word %2 : !merk.term<"frees.msg">
    %4 = merk.type_info "frees.msg/0"
    %5 = builtin.unrealized_conversion_cast %4 : !merk.type_info to !llvm.ptr
    merk.foreign_call @merk_device_send_term(%arg0, %0, %3, %5) : (!merk.term<"frees.msg">, i64, !merk.word, !llvm.ptr) -> ()
    merk.succeed
  }

  // An array that lives with a term but that the term does not hold is
  // freed after the send, which then copies the term's arrays.
  // CHECK-LABEL: merk.proc @frees.sendsOther_1_0
  // CHECK: merk.foreign_call @merk_device_send_term(%arg0, %{{.*}}, %{{.*}}, %{{.*}}) :
  // CHECK-DAG: merk.free %{{.*}} : !merk.array<i64>
  // CHECK-DAG: merk.free %{{.*}} : !merk.term<"frees.msg">
  // CHECK: merk.succeed
  merk.proc @frees.sendsOther_1_0(%arg0: !merk.term<"frees.msg">, %arg1: i64, %arg2: i1) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg1, %0 : i64 -> <i64>
    %2 = merk.array_new %arg1, %0 : i64 -> <i64>
    %3 = merk.ite -> !merk.array<i64> {
      merk.guard %arg2
      merk.yield
    } then {
      merk.yield %1 : !merk.array<i64>
    } else {
      merk.yield %2 : !merk.array<i64>
    }
    %4 = merk.construct #merk.functor<"data", 2>(%arg1, %3 : i64, !merk.array<i64>) -> !merk.term<"frees.msg">
    %5 = merk.to_word %4 : !merk.term<"frees.msg">
    %6 = merk.type_info "frees.msg/0"
    %7 = builtin.unrealized_conversion_cast %6 : !merk.type_info to !llvm.ptr
    merk.foreign_call @merk_device_send_term(%arg0, %0, %5, %7) : (!merk.term<"frees.msg">, i64, !merk.word, !llvm.ptr) -> ()
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

  // An array that a procedure takes by mode di (merk.clobbered) is its own
  // when each of its calls gives it a whole block of the caller's own: it
  // frees it after its last use, and the caller does not. So a pass that
  // makes the next array and gives it to a tail call of itself frees each.
  // CHECK-LABEL: merk.proc @frees.start_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.start_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.call @frees.pass_2_0(%arg0, %1) : (i64, !merk.array<i64>) -> ()
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.pass_2_0
  // CHECK: %[[S:.*]] = merk.array_size %arg1
  // CHECK-NEXT: merk.free %arg1 : !merk.array<i64>
  // CHECK: %[[N:.*]] = merk.array_new
  // CHECK-NOT: merk.free
  // CHECK: merk.call @frees.pass_2_0(%{{.*}}, %[[N]])
  merk.proc @frees.pass_2_0(%arg0: i64, %arg1: !merk.array<i64> {merk.clobbered}) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.array_size %arg1 : <i64>
    %1 = merk.constant 0 : i64
    merk.ite {
      %2 = arith.cmpi sgt, %arg0, %1 : i64
      merk.guard %2
      merk.yield
    } then {
      %2 = merk.constant 1 : i64
      %3 = arith.subi %arg0, %2 : i64
      %4 = merk.array_new %0, %1 : i64 -> <i64>
      merk.call @frees.pass_2_0(%3, %4) : (i64, !merk.array<i64>) -> ()
      merk.yield
    } else {
      merk.yield
    }
    merk.succeed
  }

  // A call that gives a di input what is not a whole block of the caller's
  // own, here an argument that it does not own, makes it the caller's for
  // every call: the callee frees nothing, and the other caller frees its
  // array after the call.
  // CHECK-LABEL: merk.proc @frees.passes_on_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.passes_on_1_0(%arg0: !merk.array<i64>) attributes {determinism = #merk.determinism<det>} {
    merk.call @frees.takes_1_0(%arg0) : (!merk.array<i64>) -> ()
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.gives_own_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK-NEXT: merk.call @frees.takes_1_0(%[[A]])
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  merk.proc @frees.gives_own_1_0(%arg0: i64) -> i64 attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.call @frees.takes_1_0(%1) : (!merk.array<i64>) -> ()
    %2 = arith.addi %arg0, %arg0 : i64
    merk.succeed %2 : i64
  }

  // CHECK-LABEL: merk.proc @frees.takes_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.takes_1_0(%arg0: !merk.array<i64> {merk.clobbered}) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.array_size %arg0 : <i64>
    merk.succeed
  }

  // A resize that clobbers its array (merk.clobbers) frees the old one
  // after it, when the old one's class is kept, here as a branch returns
  // it. The caller gave the array to a procedure that owns it and gives
  // back the result in its place, the same block or a new one: the caller
  // frees the result, not the array it gave.
  // CHECK-LABEL: merk.proc @frees.grows_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK-NEXT: %[[R:.*]] = merk.call @frees.room_2_0(%arg0, %[[A]])
  // CHECK-NEXT: merk.array_size %[[R]]
  // CHECK-NEXT: merk.free %[[R]] : !merk.array<i64>
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.grows_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.call @frees.room_2_0(%arg0, %1) : (i64, !merk.array<i64>) -> !merk.array<i64>
    %3 = merk.array_size %2 : <i64>
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.room_2_0
  // CHECK: %[[B:.*]] = merk.array_resize %arg1
  // CHECK-NEXT: merk.free %arg1 : !merk.array<i64>
  // CHECK-NEXT: merk.yield %[[B]]
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.room_2_0(%arg0: i64, %arg1: !merk.array<i64> {merk.clobbered}) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.array_size %arg1 : <i64>
    %1 = merk.ite -> !merk.array<i64> {
      %2 = arith.cmpi slt, %arg0, %0 : i64
      merk.guard %2
      merk.yield
    } then {
      merk.yield %arg1 : !merk.array<i64>
    } else {
      %2 = merk.constant 0 : i64
      %3 = arith.addi %0, %0 : i64
      %4 = merk.array_resize %arg1, %3, %2 : i64 {merk.clobbers} : <i64>
      merk.yield %4 : !merk.array<i64>
    }
    merk.succeed %1 : !merk.array<i64>
  }

  // A class that one path returns is freed on the others, after its last
  // use there.
  // CHECK-LABEL: merk.proc @frees.some_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK: } then {
  // CHECK-NEXT: merk.yield %[[A]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT: merk.array_size %[[A]]
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.some_1_0(%arg0: i64) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.ite -> !merk.array<i64> {
      %3 = arith.cmpi slt, %arg0, %0 : i64
      merk.guard %3
      merk.yield
    } then {
      merk.yield %1 : !merk.array<i64>
    } else {
      %3 = merk.array_size %1 : <i64>
      %4 = merk.array_new %3, %0 : i64 -> <i64>
      merk.yield %4 : !merk.array<i64>
    }
    merk.succeed %2 : !merk.array<i64>
  }

  // A message that a callee receives, in its message(...) cell, its caller
  // frees through the cell: the callee's summary says what its new result
  // holds.
  // CHECK-LABEL: merk.proc @frees.asks_1_0
  // CHECK: %[[R:.*]] = merk.call @frees.get_1_0(%arg0)
  // CHECK: merk.free %{{.*}} : !merk.term<"frees.msg">
  // CHECK: merk.free %[[R]] : !merk.term<"frees.received">
  merk.proc @frees.asks_1_0(%arg0: !merk.term<"frees.msg">) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.call @frees.get_1_0(%arg0) : (!merk.term<"frees.msg">) -> !merk.term<"frees.received">
    %1 = merk.has_functor #merk.functor<"message", 1> %0 : !merk.term<"frees.received">
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.get_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.get_1_0(%arg0: !merk.term<"frees.msg">) -> !merk.term<"frees.received"> attributes {determinism = #merk.determinism<det>} {
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
    merk.succeed %4 : !merk.term<"frees.received">
  }

  // A procedure that gives an owned array on to one that gives back a
  // block in its place gives that back in its place too: the first caller
  // frees it.
  // CHECK-LABEL: merk.proc @frees.chain_1_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK-NEXT: %[[R:.*]] = merk.call @frees.relay_2_0(%arg0, %[[A]])
  // CHECK-NEXT: merk.array_size %[[R]]
  // CHECK-NEXT: merk.free %[[R]] : !merk.array<i64>
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.chain_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.call @frees.relay_2_0(%arg0, %1) : (i64, !merk.array<i64>) -> !merk.array<i64>
    %3 = merk.array_size %2 : <i64>
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.relay_2_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.relay_2_0(%arg0: i64, %arg1: !merk.array<i64> {merk.clobbered}) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.call @frees.room_2_0(%arg0, %arg1) : (i64, !merk.array<i64>) -> !merk.array<i64>
    merk.succeed %0 : !merk.array<i64>
  }

  // A procedure that gives back either of two owned arrays, as one that
  // swaps them each round does, frees the other on the path that ends;
  // its caller frees the result, and neither array it gave.
  // CHECK-LABEL: merk.proc @frees.swaps_1_0
  // CHECK: %[[R:.*]] = merk.call @frees.rounds_3_0
  // CHECK-NEXT: merk.array_size %[[R]]
  // CHECK-NEXT: merk.free %[[R]] : !merk.array<i64>
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.swaps_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.array_new %arg0, %0 : i64 -> <i64>
    %3 = merk.call @frees.rounds_3_0(%arg0, %1, %2) : (i64, !merk.array<i64>, !merk.array<i64>) -> !merk.array<i64>
    %4 = merk.array_size %3 : <i64>
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.rounds_3_0
  // CHECK: } then {
  // CHECK-NEXT: merk.free %arg2 : !merk.array<i64>
  // CHECK-NEXT: merk.yield %arg1
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.rounds_3_0(%arg0: i64, %arg1: !merk.array<i64> {merk.clobbered}, %arg2: !merk.array<i64> {merk.clobbered}) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.ite -> !merk.array<i64> {
      %2 = arith.cmpi sle, %arg0, %0 : i64
      merk.guard %2
      merk.yield
    } then {
      merk.yield %arg1 : !merk.array<i64>
    } else {
      %2 = merk.constant 1 : i64
      %3 = arith.subi %arg0, %2 : i64
      %4 = merk.call @frees.rounds_3_0(%3, %arg2, %arg1) : (i64, !merk.array<i64>, !merk.array<i64>) -> !merk.array<i64>
      merk.yield %4 : !merk.array<i64>
    }
    merk.succeed %1 : !merk.array<i64>
  }

  // A resize frees the old array after it though the new one is given to
  // a procedure that owns it, whose the class is from then on.
  // CHECK-LABEL: merk.proc @frees.refills_1_0
  // CHECK-NOT: merk.free
  // CHECK: merk.succeed
  merk.proc @frees.refills_1_0(%arg0: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    merk.call @frees.refill_2_0(%arg0, %1) : (i64, !merk.array<i64>) -> ()
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.refill_2_0
  // CHECK: %[[B:.*]] = merk.array_resize %arg1
  // CHECK-NEXT: merk.free %arg1 : !merk.array<i64>
  // CHECK-NEXT: merk.call @frees.drops_1_0(%[[B]])
  // CHECK-NEXT: merk.succeed
  merk.proc @frees.refill_2_0(%arg0: i64, %arg1: !merk.array<i64> {merk.clobbered}) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_resize %arg1, %arg0, %0 : i64 {merk.clobbers} : <i64>
    merk.call @frees.drops_1_0(%1) : (!merk.array<i64>) -> ()
    merk.succeed
  }

  // CHECK-LABEL: merk.proc @frees.drops_1_0
  // CHECK: merk.array_size %arg0
  // CHECK-NEXT: merk.free %arg0 : !merk.array<i64>
  merk.proc @frees.drops_1_0(%arg0: !merk.array<i64> {merk.clobbered}) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.array_size %arg0 : <i64>
    merk.succeed
  }

  // A branch's result that nothing uses keeps nothing that it yields: the
  // part that reads the array and calls the next round frees it before the
  // call, and the other at its start.
  // CHECK-LABEL: merk.proc @frees.loops_2_0
  // CHECK: %[[A:.*]] = merk.array_new
  // CHECK: } then {
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK: } else {
  // CHECK-NEXT: merk.array_size %[[A]]
  // CHECK-NEXT: merk.free %[[A]] : !merk.array<i64>
  // CHECK: merk.call @frees.loops_2_0
  merk.proc @frees.loops_2_0(%arg0: i64, %arg1: i64) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 0 : i64
    %1 = merk.array_new %arg0, %0 : i64 -> <i64>
    %2 = merk.ite -> !merk.array<i64> {
      %3 = arith.cmpi sle, %arg1, %0 : i64
      merk.guard %3
      merk.yield
    } then {
      merk.yield %1 : !merk.array<i64>
    } else {
      %3 = merk.array_size %1 : <i64>
      %4 = merk.constant 1 : i64
      %5 = arith.subi %arg1, %4 : i64
      merk.call @frees.loops_2_0(%3, %5) : (i64, i64) -> ()
      merk.yield %1 : !merk.array<i64>
    }
    merk.succeed
  }
}
