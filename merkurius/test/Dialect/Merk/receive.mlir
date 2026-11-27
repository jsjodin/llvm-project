// merk.receive, a receive goal as desugaring made it, prints as it is
// parsed (docs/INPROGRESS.md, Concurrency in merk, step 1). Its lowering
// needs the process module's types, and is checked in
// test/CodeGen/receive-goal.m.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir

module @receives attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // A case's matcher: its captured values, then the message.
  merk.proc private @m.p$lambda0(%arg0: i64, %arg1: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<semidet>} {
    merk.succeed %arg1 : !merk.word
  }
  merk.proc private @m.p$lambda1(%arg0: !merk.word) -> !merk.word attributes {determinism = #merk.determinism<semidet>} {
    merk.succeed %arg0 : !merk.word
  }

  // CHECK-LABEL: merk.proc @m.p
  // CHECK: merk.receive %arg0 with %arg1
  // CHECK-SAME: matchers [@m.p$lambda0, @m.p$lambda1] counts [1, 0]
  // CHECK-SAME: on(%arg2, %arg3 : !merk.term<"process.mailbox(int)">, !merk.term<"process.mailbox(string)">)
  // CHECK-SAME: infos(%arg4, %arg5) captured(%arg6 : i64) captures {{\[\[\["int/0"\]\], \[\]\]}}
  merk.proc @m.p(%arg0: !merk.term<"process.timeout">, %arg1: !merk.type_info, %arg2: !merk.term<"process.mailbox(int)">, %arg3: !merk.term<"process.mailbox(string)">, %arg4: !merk.type_info, %arg5: !merk.type_info, %arg6: i64) -> !merk.term<"process.receive_result(string)"> attributes {determinism = #merk.determinism<det>} {
    %0 = merk.receive %arg0 with %arg1 matchers [@m.p$lambda0, @m.p$lambda1] counts [1, 0] on(%arg2, %arg3 : !merk.term<"process.mailbox(int)">, !merk.term<"process.mailbox(string)">) infos(%arg4, %arg5) captured(%arg6 : i64) captures [[["int/0"]], []] : !merk.term<"process.timeout"> -> !merk.term<"process.receive_result(string)">
    merk.succeed %0 : !merk.term<"process.receive_result(string)">
  }
}
