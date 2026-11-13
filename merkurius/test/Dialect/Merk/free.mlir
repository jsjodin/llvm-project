// merk.free (docs/INPROGRESS.md, Memory that a GPU process gives back)
// prints as it is parsed. In a GPU process's compile it calls
// merk_device_free: an array or a string by its pointer, a term's cell from
// its word without the tag, or null for a constant's word; a term of an
// enumeration has no cell and is not freed. Static cells and strings go into
// a section of their own in a GPU process's image, which merk_device_free
// leaves alone.
// On the host, which has a collector, it does nothing.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// The GPU process's compile is of the module with a device's triple, the
// procedures marked as roots of what it keeps.
// RUN: sed -e 's/llvm.target_triple = "x86_64-unknown-linux-gnu"/llvm.target_triple = "amdgpu10.30-amd-amdhsa", merk.process_kernels/' -e 's/determinism = #merk.determinism<det>}/determinism = #merk.determinism<det>, merk.process_root}/' %s > %t.gpu.mlir
// RUN: merkc -fc1 -emit-llvm %t.gpu.mlir -o - | FileCheck %s --check-prefix=GPU
// RUN: merkc -fc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST

module @free attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
  merk.type_def "free.colour" [#merk.constructor<"red">, #merk.constructor<"green">]

  // GPU: @merk.str.0 = private unnamed_addr constant [4 x i8] c"one\00", section "merk_static"
  // GPU-NEXT: @merk.term.0 = private unnamed_addr constant { i64, ptr } { i64 1, ptr null }, section "merk_static", align 8
  // HOST-NOT: merk_static

  // CHECK-LABEL: merk.proc @free.drop_4_0
  // CHECK: merk.free %arg0 : !merk.array<i64>
  // CHECK-NEXT: merk.free %arg1 : !merk.string
  // CHECK-NEXT: merk.free %arg2 : !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.free %arg3 : !merk.term<"free.colour">
  // GPU-LABEL: define {{.*}}@free.drop_4_0(ptr %0, ptr %1, ptr %2, i64 %3)
  // GPU: call void @merk_device_free(ptr %0)
  // GPU-NEXT: call void @merk_device_free(ptr %1)
  // GPU-NEXT: %[[W:.*]] = ptrtoint ptr %2 to i64
  // GPU-NEXT: %[[C:.*]] = icmp eq i64 %[[W]], 0
  // GPU-NEXT: %[[U:.*]] = and i64 %[[W]], -8
  // GPU-NEXT: %[[S:.*]] = select i1 %[[C]], i64 0, i64 %[[U]]
  // GPU-NEXT: %[[P:.*]] = inttoptr i64 %[[S]] to ptr
  // GPU-NEXT: call void @merk_device_free(ptr %[[P]])
  // GPU-NEXT: ret void
  // HOST-LABEL: define {{.*}}@free.drop_4_0(
  // HOST-NOT: merk_device_free
  // HOST: ret void
  merk.proc @free.drop_4_0(%arg0: !merk.array<i64>, %arg1: !merk.string, %arg2: !merk.term<"list.list(int)">, %arg3: !merk.term<"free.colour">) attributes {determinism = #merk.determinism<det>} {
    merk.free %arg0 : !merk.array<i64>
    merk.free %arg1 : !merk.string
    merk.free %arg2 : !merk.term<"list.list(int)">
    merk.free %arg3 : !merk.term<"free.colour">
    merk.succeed
  }

  // A ground term is a static cell.
  merk.proc @free.ground_1_0() -> (!merk.term<"list.list(int)">, !merk.string) attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant 1 : i64
    %1 = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
    %2 = merk.construct #merk.functor<"[|]", 2>(%0, %1 : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    %3 = merk.constant "one" : !merk.string
    merk.succeed %2, %3 : !merk.term<"list.list(int)">, !merk.string
  }
}
