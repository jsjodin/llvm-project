// The array operations print as they are parsed, and lower to loads and
// stores after the array's header, or to calls of the runtime;
// merk.array_from_list's with the tag of [|] in the list's layout, and
// those that make arrays with their elements' width (runtime/array.h): 0
// for words, 3 for 4 bytes.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @arrays attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", merk.no_root_types} {
  merk.type_ctor "string/0" module "" name "string" kind "string" linkage "declaration" arity 0
  merk.type_ctor "array.array/1" module "array" name "array" kind "array" linkage "declaration" arity 1
  merk.type_ctor "float/0" module "" name "float" kind "float" linkage "declaration" arity 0
  merk.type_ctor "list.list/1" module "list" name "list" kind "term" linkage "declaration" arity 1
  merk.type_def "list.list(float)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(f64, !merk.term<"list.list(float)">)>]

  // CHECK-LABEL: merk.proc @arrays.from_list_2_0(%arg0: !merk.term<"list.list(float)">) -> !merk.array<f64>
  // CHECK-NEXT: %[[A:.*]] = merk.array_from_list %arg0 : <"list.list(float)"> -> <f64>
  merk.proc @arrays.from_list_2_0(%l: !merk.term<"list.list(float)">) -> !merk.array<f64> attributes {determinism = #merk.determinism<det>} {
    %a = merk.array_from_list %l : !merk.term<"list.list(float)"> -> !merk.array<f64>
    merk.succeed %a : !merk.array<f64>
  }

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

  // CHECK-LABEL: merk.proc @arrays.narrow_4_0
  // CHECK-NEXT: %[[N:.*]] = merk.array_new %arg1, %arg2 : !merk.word width of %arg0 -> <!merk.word>
  // CHECK-NEXT: %[[L:.*]] = merk.array_from_list %arg3 width of %arg0 : <"list.list(float)"> -> <!merk.word>
  // CHECK-NEXT: %[[W:.*]] = merk.array_words %[[N]] : <!merk.word>
  // CHECK-NEXT: %[[B:.*]] = merk.array_new %[[W]] -> <i8>
  // CHECK-NEXT: %[[X:.*]] = merk.array_get %[[B]][%arg1] : <i8> -> i8
  // CHECK-NEXT: %[[S:.*]] = merk.array_set %[[B]][%arg1], %[[X]] : <i8>, i8
  // CHECK-NEXT: %[[Y:.*]] = merk.array_get %[[N]][%arg1] : <!merk.word> -> !merk.word
  // CHECK-NEXT: %[[T:.*]] = merk.array_set %[[L]][%arg1], %[[Y]] : <!merk.word>, !merk.word
  merk.proc @arrays.narrow_4_0(%ti: !merk.type_info, %n: i64, %x: !merk.word, %l: !merk.term<"list.list(float)">) -> (!merk.array<i8>, !merk.array<!merk.word>) attributes {determinism = #merk.determinism<det>} {
    %a = merk.array_new %n, %x : !merk.word width of %ti -> !merk.array<!merk.word>
    %f = merk.array_from_list %l width of %ti : !merk.term<"list.list(float)"> -> !merk.array<!merk.word>
    %w = merk.array_words %a : !merk.array<!merk.word>
    %b = merk.array_new %w -> !merk.array<i8>
    %y = merk.array_get %b[%n] : !merk.array<i8> -> i8
    %c = merk.array_set %b[%n], %y : !merk.array<i8>, i8
    %z = merk.array_get %a[%n] : !merk.array<!merk.word> -> !merk.word
    %d = merk.array_set %f[%n], %z : !merk.array<!merk.word>, !merk.word
    merk.succeed %c, %d : !merk.array<i8>, !merk.array<!merk.word>
  }
}

// An array is a pointer; an element is a word, a float's bits or a
// character extended.
// LLVM-LABEL: define tailcc ptr addrspace(1) @arrays.from_list_2_0(ptr addrspace(1) %0)
// LLVM-NEXT: call void @merk_gc_poll()
// LLVM-NEXT: call ptr addrspace(1) @merk_array_from_list(ptr addrspace(1) %0, i64 1, i64 0)
// LLVM-LABEL: define tailcc { ptr addrspace(1), ptr addrspace(1) } @arrays.new_2_0(i64 %0, double %1)
// LLVM: %[[W:.*]] = bitcast double %1 to i64
// LLVM-NEXT: %[[P:.*]] = inttoptr i64 %[[W]] to ptr addrspace(1)
// LLVM-NEXT: call ptr addrspace(1) @merk_array_new(i64 %0, ptr addrspace(1) %[[P]], i64 0)
// LLVM-NEXT: call void @merk_gc_poll()
// LLVM-NEXT: call ptr addrspace(1) @merk_array_new(i64 %0, ptr addrspace(1) null, i64 3)
// The index is checked against the size, the header's low 56 bits, as
// unsigned.
// LLVM-LABEL: define tailcc { i64, double } @arrays.get_3_0(ptr addrspace(1) %0, i64 %1)
// LLVM-NEXT: %[[HEADER:.*]] = load i64, ptr addrspace(1) %0
// LLVM-NEXT: %[[SIZE:.*]] = and i64 %[[HEADER]], 72057594037927935
// LLVM-NEXT: %[[IN:.*]] = icmp ult i64 %1, %[[SIZE]]
// LLVM-NEXT: br i1 %[[IN]], label %[[GET:.*]], label %[[OUT:.*]]
// LLVM: [[OUT]]:
// LLVM-NEXT: %[[MAX:.*]] = sub i64 %[[SIZE]], 1
// LLVM-NEXT: call void @merk_index_out_of_bounds(ptr @merk.str.{{[0-9]+}}, i64 %1, i64 %[[MAX]])
// LLVM-NEXT: unreachable
// LLVM: [[GET]]:
// LLVM: %[[SLOT:.*]] = add i64 %1, 1
// LLVM-NEXT: %[[P:.*]] = getelementptr i64, ptr addrspace(1) %0, i64 %[[SLOT]]
// LLVM-NEXT: %[[X:.*]] = load i64, ptr addrspace(1) %[[P]]
// LLVM-NEXT: bitcast i64 %[[X]] to double
// A set stores in place; the others are the runtime's.
// LLVM-LABEL: define tailcc ptr addrspace(1) @arrays.update_5_0(ptr addrspace(1) %0, i64 %1, ptr addrspace(1) %2)
// LLVM: store ptr addrspace(1) %2, ptr addrspace(1)
// LLVM: call ptr addrspace(1) @merk_array_fill(ptr addrspace(1) %0, i64 %1, i64 %1, ptr addrspace(1)
// LLVM: call ptr addrspace(1) @merk_array_copy(ptr addrspace(1)
// LLVM: call ptr addrspace(1) @merk_array_resize(ptr addrspace(1) {{.*}}, i64 %1, ptr addrspace(1)
// LLVM: call ptr addrspace(1) @merk_array_resize(ptr addrspace(1) {{.*}}, i64 %1, ptr addrspace(1) null)
// Polymorphic arrays are made as wide as the runtime finds the type_info's
// type; an array's words come from its header; an i8 element is its byte,
// after the header; a word array's element, or a narrow one's, as the
// header says (runtime/array.h).
// LLVM-LABEL: define tailcc { ptr addrspace(1), ptr addrspace(1) } @arrays.narrow_4_0(
// LLVM: %[[WIDTH:.*]] = call i64 @merk_array_width(ptr %{{.*}})
// LLVM-NEXT: call ptr addrspace(1) @merk_array_new(i64 %1, ptr addrspace(1) %2, i64 %[[WIDTH]])
// LLVM: %[[WIDTH2:.*]] = call i64 @merk_array_width(ptr %{{.*}})
// LLVM-NEXT: call ptr addrspace(1) @merk_array_from_list(ptr addrspace(1) %3, i64 1, i64 %[[WIDTH2]])
// LLVM: %[[H:.*]] = load i64, ptr addrspace(1) %{{.*}}
// LLVM-NEXT: and i64 %[[H]], 72057594037927935
// LLVM-NEXT: lshr i64 %[[H]], 56
// LLVM: call ptr addrspace(1) @merk_array_new(i64 %{{.*}}, ptr addrspace(1) null, i64 1)
// LLVM: %[[E:.*]] = getelementptr i64, ptr addrspace(1) %[[B:.*]], i32 1
// LLVM-NEXT: %[[AT:.*]] = getelementptr i8, ptr addrspace(1) %[[E]], i64 %1
// LLVM-NEXT: %[[X:.*]] = load i8, ptr addrspace(1) %[[AT]]
// LLVM: store i8 %[[X]], ptr addrspace(1)
// LLVM: %[[H2:.*]] = load i64, ptr addrspace(1) %{{.*}}
// LLVM-NEXT: %[[CODE:.*]] = lshr i64 %[[H2]], 56
// LLVM-NEXT: %[[WORDS:.*]] = icmp eq i64 %[[CODE]], 0
// LLVM-NEXT: br i1 %[[WORDS]]
// LLVM: switch i64 %[[CODE]]
// LLVM: merk.array_load.exit:
// LLVM: merk.array_store.exit:
// LLVM-NOT: define {{.*}}@merk.array_
// LLVM: declare void @merk_index_out_of_bounds(ptr, i64, i64) #[[NORETURN:[0-9]+]]
// LLVM: attributes #[[NORETURN]] = { noreturn nounwind }
