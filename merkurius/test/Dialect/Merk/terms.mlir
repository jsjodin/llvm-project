// The term operations print as they are parsed, and lower by their types'
// layouts, chosen from the merk.type_defs (design document, "Type
// representation"): a type of constants only is a word, a type of one
// constructor of one argument is the argument, and any other a word of a
// constant shifted past a 3-bit primary tag or a pointer to a cell with
// its constructor's tag. Tag 0 is the constants', and the last tag is
// shared, with a secondary tag in the cell's first word, by the
// constructors with arguments that need more tags than are left.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @terms attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  // CHECK: merk.type_def "terms.colour" [#merk.constructor<"red">, #merk.constructor<"green">]
  // CHECK: merk.type_def "terms.wrap" [#merk.constructor<"wrap"(f64)>]
  // CHECK: merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
  merk.type_def "terms.colour" [#merk.constructor<"red">, #merk.constructor<"green">]
  merk.type_def "terms.wrap" [#merk.constructor<"wrap"(f64)>]
  merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
  // Eight constructors with arguments and a constant: tags 1 to 6, and 7
  // shared by g and h, with secondary tags 0 and 1.
  merk.type_def "terms.many" [#merk.constructor<"z">, #merk.constructor<"a"(i64)>, #merk.constructor<"b"(i64)>, #merk.constructor<"c"(i64)>, #merk.constructor<"d"(i64)>, #merk.constructor<"e"(i64)>, #merk.constructor<"f"(i64)>, #merk.constructor<"g"(i64)>, #merk.constructor<"h"(i64)>]

  // CHECK-LABEL: merk.proc @terms.cons_3_0(%arg0: i64, %arg1: !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
  // CHECK-NEXT: %[[L:.*]] = merk.construct #merk.functor<"[|]", 2>(%arg0, %arg1 : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.succeed %[[L]]
  // LLVM-LABEL: define tailcc ptr @terms.cons_3_0(i64 %0, ptr %1) {
  // LLVM-NEXT: %[[CELL:.*]] = call ptr @merk_alloc(i64 16)
  // LLVM-NEXT: store i64 %0, ptr %[[CELL]]
  // LLVM-NEXT: %[[T:.*]] = getelementptr i8, ptr %[[CELL]], i32 8
  // LLVM-NEXT: store ptr %1, ptr %[[T]]
  // LLVM-NEXT: %[[L:.*]] = getelementptr i8, ptr %[[CELL]], i32 1
  // LLVM-NEXT: ret ptr %[[L]]
  merk.proc @terms.cons_3_0(%h: i64, %t: !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)"> attributes {determinism = #merk.determinism<det>} {
    %l = merk.construct #merk.functor<"[|]", 2>(%h, %t : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    merk.succeed %l : !merk.term<"list.list(int)">
  }

  // CHECK-LABEL: merk.proc @terms.head_2_0(%arg0: !merk.term<"list.list(int)">) -> i64
  // CHECK-NEXT: %[[IS:.*]] = merk.has_functor #merk.functor<"[|]", 2> %arg0 : !merk.term<"list.list(int)">
  // CHECK-NEXT: merk.guard %[[IS]]
  // CHECK-NEXT: %[[F:.*]]:2 = merk.deconstruct #merk.functor<"[|]", 2> %arg0 : !merk.term<"list.list(int)"> -> i64, !merk.term<"list.list(int)">
  // LLVM-LABEL: define tailcc { i1, i64 } @terms.head_2_0(ptr %0) {
  // LLVM-NEXT: %[[W:.*]] = ptrtoint ptr %0 to i64
  // LLVM-NEXT: %[[TAG:.*]] = and i64 %[[W]], 7
  // LLVM-NEXT: %[[IS:.*]] = icmp eq i64 %[[TAG]], 1
  // LLVM-NEXT: br i1 %[[IS]]
  // LLVM: %[[HP:.*]] = getelementptr i8, ptr %0, i32 -1
  // LLVM-NEXT: load i64, ptr %[[HP]]
  // LLVM-NEXT: %[[TP:.*]] = getelementptr i8, ptr %0, i32 7
  // LLVM-NEXT: load ptr, ptr %[[TP]]
  merk.proc @terms.head_2_0(%l: !merk.term<"list.list(int)">) -> i64 attributes {determinism = #merk.determinism<semidet>} {
    %is = merk.has_functor #merk.functor<"[|]", 2> %l : !merk.term<"list.list(int)">
    merk.guard %is
    %h, %t = merk.deconstruct #merk.functor<"[|]", 2> %l : !merk.term<"list.list(int)"> -> i64, !merk.term<"list.list(int)">
    merk.succeed %h : i64
  }

  // A constant is its number shifted past the tag; a constant only type's
  // constant is its number.
  // CHECK-LABEL: merk.proc @terms.constants_2_0() -> (!merk.term<"list.list(int)">, !merk.term<"terms.colour">)
  // CHECK-NEXT: merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
  // LLVM-LABEL: define tailcc { ptr, i64 } @terms.constants_2_0() {
  // LLVM-NEXT: ret { ptr, i64 } { ptr null, i64 1 }
  merk.proc @terms.constants_2_0() -> (!merk.term<"list.list(int)">, !merk.term<"terms.colour">) attributes {determinism = #merk.determinism<det>} {
    %nil = merk.construct #merk.functor<"[]", 0> -> !merk.term<"list.list(int)">
    %green = merk.construct #merk.functor<"green", 0> -> !merk.term<"terms.colour">
    merk.succeed %nil, %green : !merk.term<"list.list(int)">, !merk.term<"terms.colour">
  }

  // A type of one constructor of one argument is the argument.
  // LLVM-LABEL: define tailcc double @terms.unwrap_2_0(double %0) {
  // LLVM-NEXT: ret double %0
  merk.proc @terms.unwrap_2_0(%w: !merk.term<"terms.wrap">) -> f64 attributes {determinism = #merk.determinism<det>} {
    %x = merk.deconstruct #merk.functor<"wrap", 1> %w : !merk.term<"terms.wrap"> -> f64
    merk.succeed %x : f64
  }

  // A switch on the primary tag, then on the secondary tag of the
  // constructors that share one; tag 0 is the only constant's.
  // CHECK-LABEL: merk.proc @terms.which_2_0(%arg0: !merk.term<"terms.many">) -> i64
  // CHECK-NEXT: merk.switch %arg0 : !merk.term<"terms.many"> -> i64 cases {{\[}}[#merk.functor<"z", 0>], [#merk.functor<"a", 1>, #merk.functor<"h", 1>]] can_fail {
  // LLVM-LABEL: define tailcc { i1, i64 } @terms.which_2_0(ptr %0) {
  // LLVM-NEXT: %[[W:.*]] = ptrtoint ptr %0 to i64
  // LLVM-NEXT: %[[TAG:.*]] = and i64 %[[W]], 7
  // LLVM-NEXT: ptrtoint ptr %0 to i64
  // LLVM-NEXT: switch i64 %[[TAG]], label %[[NO:.*]] [
  // LLVM-NEXT: i64 1, label %[[A:.*]]
  // LLVM-NEXT: i64 0, label %{{.*}}
  // LLVM-NEXT: i64 7, label %[[SHARED:.*]]
  // LLVM-NEXT: ]
  // LLVM: [[SHARED]]:
  // LLVM-NEXT: %[[SP:.*]] = getelementptr i8, ptr %0, i32 -7
  // LLVM-NEXT: %[[S:.*]] = load i64, ptr %[[SP]]
  // LLVM-NEXT: switch i64 %[[S]], label %[[NO]] [
  // LLVM-NEXT: i64 1, label %[[A]]
  // LLVM-NEXT: ]
  merk.proc @terms.which_2_0(%m: !merk.term<"terms.many">) -> i64 attributes {determinism = #merk.determinism<semidet>} {
    %r = merk.switch %m : !merk.term<"terms.many"> -> i64 cases [[#merk.functor<"z", 0>], [#merk.functor<"a", 1>, #merk.functor<"h", 1>]] can_fail {
      %zero = merk.constant 0 : i64
      merk.yield %zero : i64
    }, {
      %one = merk.constant 1 : i64
      merk.yield %one : i64
    }
    merk.succeed %r : i64
  }

  // A cell of a shared tag starts with its secondary tag.
  // LLVM-LABEL: define tailcc ptr @terms.h_2_0(i64 %0) {
  // LLVM-NEXT: %[[CELL:.*]] = call ptr @merk_alloc(i64 16)
  // LLVM-NEXT: store i64 1, ptr %[[CELL]]
  // LLVM-NEXT: %[[X:.*]] = getelementptr i8, ptr %[[CELL]], i32 8
  // LLVM-NEXT: store i64 %0, ptr %[[X]]
  // LLVM-NEXT: getelementptr i8, ptr %[[CELL]], i32 7
  merk.proc @terms.h_2_0(%x: i64) -> !merk.term<"terms.many"> attributes {determinism = #merk.determinism<det>} {
    %h = merk.construct #merk.functor<"h", 1>(%x : i64) -> !merk.term<"terms.many">
    merk.succeed %h : !merk.term<"terms.many">
  }
}
