// merk.spawn prints as it is parsed (docs/INPROGRESS.md, Concurrency in
// merk, step 2 (c)): a known body with its captured values, a closure's
// unit body, at a place, with a stack's size, and a declared kind's. Its
// lowering needs the process module's procedures, and is checked in
// test/CodeGen/spawn-ops.m.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir

module @spawns attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // CHECK-LABEL: merk.proc @spawns(
  // CHECK: merk.spawn body @m.body$words(%{{.*}} : i64) with %{{.*}} captures {{\[\["int/0"\]\]}} : !merk.term<"process.mailbox(T0)">
  // CHECK: merk.spawn body unit(%{{.*}} : !merk.closure) at %{{.*}} : !merk.term<"process.place"> with %{{.*}} : !merk.term<"process.mailbox(T0)">
  // CHECK: merk.spawn body unit(%{{.*}} : !merk.closure) stack %{{.*}} with %{{.*}} : !merk.term<"process.mailbox(T0)">
  // CHECK: merk.spawn body @m.counter_4_0$words(%{{.*}}, %{{.*}} : i64, !merk.term<"process.mailbox(int)">) at %{{.*}} : !merk.term<"process.place"> kind "counter" on_gpu : !merk.term<"process.mailbox(int)">
  merk.proc @spawns(%n: i64, %c: !merk.closure, %p: !merk.term<"process.place">, %i: !merk.type_info, %r: !merk.term<"process.mailbox(int)">) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.spawn body @m.body$words(%n : i64) with %i captures [["int/0"]] : !merk.term<"process.mailbox(T0)">
    %1 = merk.spawn body unit(%c : !merk.closure) at %p : !merk.term<"process.place"> with %i : !merk.term<"process.mailbox(T0)">
    %2 = merk.spawn body unit(%c : !merk.closure) stack %n with %i : !merk.term<"process.mailbox(T0)">
    %3 = merk.spawn body @m.counter_4_0$words(%n, %r : i64, !merk.term<"process.mailbox(int)">) at %p : !merk.term<"process.place"> kind "counter" on_gpu : !merk.term<"process.mailbox(int)">
    merk.succeed
  }
}
