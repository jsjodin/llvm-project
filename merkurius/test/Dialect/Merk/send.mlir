// merk.send, process.send/4, prints as it is parsed (docs/INPROGRESS.md,
// Concurrency in merk, step 2): the host's, with the message's type_info,
// and the device runtime's, with a key and a ring kind, and the type_info
// only for a copied term. Its lowering needs the process module's types,
// and is checked in test/CodeGen/send.m.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir

module @sends attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // CHECK-LABEL: merk.proc @send(
  // CHECK: merk.send %{{.*}} to %{{.*}} with %{{.*}} : i64, !merk.term<"process.mailbox(int)">
  // CHECK: merk.send %{{.*}} to %{{.*}} key 42 ring_kind 0 : i64, !merk.term<"process.mailbox(int)">
  // CHECK: merk.send %{{.*}} to %{{.*}} with %{{.*}} key -3 ring_kind 2 : !merk.word, !merk.word
  merk.proc @send(%m: i64, %w: !merk.word, %i: !merk.type_info, %b: !merk.term<"process.mailbox(int)">, %c: !merk.word) attributes {determinism = #merk.determinism<det>} {
    merk.send %m to %b with %i : i64, !merk.term<"process.mailbox(int)">
    merk.send %m to %b key 42 ring_kind 0 : i64, !merk.term<"process.mailbox(int)">
    merk.send %w to %c with %i key -3 ring_kind 2 : !merk.word, !merk.word
    merk.succeed
  }
}
