// merk.new_mailbox, merk.close_mailbox, merk.stop, merk.monitor and
// merk.demonitor print as they are parsed (docs/INPROGRESS.md, Concurrency
// in merk, step 2 (b)). Their lowering needs the process module's
// procedures, and is checked in test/CodeGen/mailbox-ops.m.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir

module @mailboxes attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // CHECK-LABEL: merk.proc @mailboxes(
  // CHECK: %[[BOX:.*]] = merk.new_mailbox with %{{.*}} : !merk.term<"process.mailbox(int)">
  // CHECK: %[[REF:.*]] = merk.monitor %[[BOX]], %{{.*}}, %{{.*}} with %{{.*}}, %{{.*}} : !merk.term<"process.mailbox(int)">, !merk.word -> !merk.term<"process.monitor_ref">
  // CHECK: merk.demonitor %[[REF]] : !merk.term<"process.monitor_ref">
  // CHECK: merk.stop %[[BOX]] with %{{.*}} : !merk.term<"process.mailbox(int)">
  // CHECK: merk.close_mailbox %[[BOX]] with %{{.*}} : !merk.term<"process.mailbox(int)">
  merk.proc @mailboxes(%i: !merk.type_info, %w: !merk.word, %f: !merk.closure) attributes {determinism = #merk.determinism<det>} {
    %box = merk.new_mailbox with %i : !merk.term<"process.mailbox(int)">
    %ref = merk.monitor %box, %w, %f with %i, %i : !merk.term<"process.mailbox(int)">, !merk.word -> !merk.term<"process.monitor_ref">
    merk.demonitor %ref : !merk.term<"process.monitor_ref">
    merk.stop %box with %i : !merk.term<"process.mailbox(int)">
    merk.close_mailbox %box with %i : !merk.term<"process.mailbox(int)">
    merk.succeed
  }
}
