// The array operations print as they are parsed, and lower to loads and
// stores of words after the array's size, or to calls of the runtime.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @arrays attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // CHECK-LABEL: merk.proc @arrays.new_2_0(%arg0: i64, %arg1: f64) -> (!merk.array<f64>, !merk.array<i32>)
  // CHECK-NEXT: %[[A:.*]] = merk.array_new %arg0, %arg1 : f64 -> <f64>
  // CHECK-NEXT: %[[E:.*]] = merk.array_new %arg0 -> <i32>
  // CHECK-NEXT: merk.succeed %[[A]], %[[E]]
  merk.proc @arrays.new_2_0(%n: i64, %x: f64) -> (!merk.array<f64>, !merk.array<i32>) attributes {determinism = #merk.determinism<det>} {
    %a = merk.array_new %n, %x : f64 -> !merk.array<f64>
    %e = merk.array_new %n -> !merk.array<i32>
    merk.succeed %a, %e : !merk.array<f64>, !merk.array<i32>
  }

  // CHECK-LABEL: merk.proc @arrays.get_3_0(%arg0: !merk.array<f64>, %arg1: i64) -> (i64, f64)
  // CHECK-NEXT: merk.check_index %arg0[%arg1], "array.lookup" : <f64>
  // CHECK-NEXT: %[[S:.*]] = merk.array_size %arg0 : <f64>
  // CHECK-NEXT: %[[X:.*]] = merk.array_get %arg0[%arg1] : <f64> -> f64
  merk.proc @arrays.get_3_0(%a: !merk.array<f64>, %i: i64) -> (i64, f64) attributes {determinism = #merk.determinism<det>} {
    merk.check_index %a[%i], "array.lookup" : !merk.array<f64>
    %s = merk.array_size %a : !merk.array<f64>
    %x = merk.array_get %a[%i] : !merk.array<f64> -> f64
    merk.succeed %s, %x : i64, f64
  }

  // CHECK-LABEL: merk.proc @arrays.update_5_0
  // CHECK-NEXT: %[[S:.*]] = merk.array_set %arg0[%arg1], %arg2 : <!merk.string>, !merk.string
  // CHECK-NEXT: %[[F:.*]] = merk.array_fill %[[S]][%arg1, %arg1], %arg2 : <!merk.string>, !merk.string
  // CHECK-NEXT: %[[C:.*]] = merk.array_copy %[[F]] : <!merk.string>
  // CHECK-NEXT: %[[R:.*]] = merk.array_resize %[[C]], %arg1, %arg2 : !merk.string : <!merk.string>
  // CHECK-NEXT: %[[K:.*]] = merk.array_resize %[[R]], %arg1 : <!merk.string>
  merk.proc @arrays.update_5_0(%a: !merk.array<!merk.string>, %i: i64, %x: !merk.string) -> !merk.array<!merk.string> attributes {determinism = #merk.determinism<det>} {
    %s = merk.array_set %a[%i], %x : !merk.array<!merk.string>, !merk.string
    %f = merk.array_fill %s[%i, %i], %x : !merk.array<!merk.string>, !merk.string
    %c = merk.array_copy %f : !merk.array<!merk.string>
    %r = merk.array_resize %c, %i, %x : !merk.string : !merk.array<!merk.string>
    %k = merk.array_resize %r, %i : !merk.array<!merk.string>
    merk.succeed %k : !merk.array<!merk.string>
  }
}

// An array is a pointer; an element is a word, a float's bits or a
// character extended.
// LLVM-LABEL: define tailcc { ptr, ptr } @arrays.new_2_0(i64 %0, double %1)
// LLVM: %[[W:.*]] = bitcast double %1 to i64
// LLVM-NEXT: call ptr @merk_array_new(i64 %0, i64 %[[W]])
// LLVM-NEXT: call ptr @merk_array_new(i64 %0, i64 0)
// The index is checked against the size, the first word, as unsigned.
// LLVM-LABEL: define tailcc { i64, double } @arrays.get_3_0(ptr %0, i64 %1)
// LLVM-NEXT: %[[SIZE:.*]] = load i64, ptr %0
// LLVM-NEXT: %[[IN:.*]] = icmp ult i64 %1, %[[SIZE]]
// LLVM-NEXT: br i1 %[[IN]], label %[[GET:.*]], label %[[OUT:.*]]
// LLVM: [[OUT]]:
// LLVM-NEXT: %[[MAX:.*]] = sub i64 %[[SIZE]], 1
// LLVM-NEXT: call void @merk_index_out_of_bounds(ptr @merk.str.{{[0-9]+}}, i64 %1, i64 %[[MAX]])
// LLVM-NEXT: unreachable
// LLVM: [[GET]]:
// LLVM: %[[SLOT:.*]] = add i64 %1, 1
// LLVM-NEXT: %[[P:.*]] = getelementptr i64, ptr %0, i64 %[[SLOT]]
// LLVM-NEXT: %[[X:.*]] = load i64, ptr %[[P]]
// LLVM-NEXT: bitcast i64 %[[X]] to double
// A set stores in place; the others are the runtime's.
// LLVM-LABEL: define tailcc ptr @arrays.update_5_0(ptr %0, i64 %1, ptr %2)
// LLVM: %[[W:.*]] = ptrtoint ptr %2 to i64
// LLVM-NEXT: store i64 %[[W]], ptr
// LLVM: call ptr @merk_array_fill(ptr %0, i64 %1, i64 %1, i64
// LLVM: call ptr @merk_array_copy(ptr
// LLVM: call ptr @merk_array_resize(ptr {{.*}}, i64 %1, i64
// LLVM: call ptr @merk_array_resize(ptr {{.*}}, i64 %1, i64 0)
// LLVM: declare void @merk_index_out_of_bounds(ptr, i64, i64) #[[NORETURN:[0-9]+]]
// LLVM: attributes #[[NORETURN]] = { noreturn nounwind }
