// The merk operations print as they are parsed: -fc1 reads a .mlir file and
// writes it again with -emit-merk, the same the second time. It also lowers
// one to LLVM IR for the target the module records.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @ops attributes {merk.foreign_code = "#include <stdio.h>\0Along merk_foreign_ops_Dshow(const char *s, long n) { return printf(\22%s%ld\22, s, n); }\0Avoid merk_foreign_ops_Dtwo(long *a, long *b) { *a = 1; *b = 2; }\0Aint merk_foreign_ops_Dhalf(long x, long *h) { *h = x / 2; return x % 2 == 0; }\0A", llvm.target_triple = "x86_64-unknown-linux-gnu", merk.address_spaces = #merk.address_spaces<generic = 0, global = 1, shared = 3, constant = 4, private = 5>} {
  // CHECK: merk.address_spaces = #merk.address_spaces<generic = 0, global = 1, shared = 3, constant = 4, private = 5>

  // CHECK-LABEL: merk.proc private @ops.double_f1_0(%arg0: i64) -> i64
  // CHECK-SAME: attributes {determinism = #merk.determinism<det>}
  // CHECK-NEXT: merk.succeed %arg0 : i64
  merk.proc private @ops.double_f1_0(%x: i64) -> i64 attributes {determinism = #merk.determinism<det>} {
    merk.succeed %x : i64
  }

  // CHECK-LABEL: merk.proc private @ops.two_2_0(%arg0: i64) -> (i64, f64)
  // CHECK-SAME: attributes {determinism = #merk.determinism<cc_multi>}
  // CHECK-NEXT: %[[F:.*]] = merk.call @other.float_f1_0(%arg0) : (i64) -> f64
  // CHECK-NEXT: merk.succeed %arg0, %[[F]] : i64, f64
  merk.proc private @ops.two_2_0(%x: i64) -> (i64, f64) attributes {determinism = #merk.determinism<cc_multi>} {
    %f = merk.call @other.float_f1_0(%x) : (i64) -> f64
    merk.succeed %x, %f : i64, f64
  }

  // A procedure of another module is declared without a body.
  // CHECK-LABEL: merk.proc private @other.float_f1_0(i64) -> f64
  merk.proc private @other.float_f1_0(i64) -> f64 attributes {determinism = #merk.determinism<det>}

  // CHECK-LABEL: merk.proc @ops.use_1_0(%arg0: i64)
  // CHECK-NEXT: %[[D:.*]] = merk.call @ops.double_f1_0(%arg0) : (i64) -> i64
  // CHECK-NEXT: %{{.*}}:2 = merk.call @ops.two_2_0(%[[D]]) : (i64) -> (i64, f64)
  // CHECK-NEXT: merk.succeed
  merk.proc @ops.use_1_0(%x: i64) attributes {determinism = #merk.determinism<det>} {
    %d = merk.call @ops.double_f1_0(%x) : (i64) -> i64
    %a:2 = merk.call @ops.two_2_0(%d) : (i64) -> (i64, f64)
    merk.succeed
  }

  // Constants; a foreign_proc's C function, of the module's foreign code.
  // CHECK-LABEL: merk.proc @ops.main_2_0()
  // CHECK-NEXT: %[[N:.*]] = merk.constant 42 : i64
  // CHECK-NEXT: %{{.*}} = merk.constant 2.500000e+00 : f64
  // CHECK-NEXT: %{{.*}} = merk.constant 97 : i32
  // CHECK-NEXT: %[[S:.*]] = merk.constant "hi\0A" : !merk.string
  // CHECK-NEXT: %{{.*}} = merk.foreign_call @merk_foreign_ops_Dshow(%[[S]], %[[N]]) : (!merk.string, i64) -> i64
  // CHECK-NEXT: %{{.*}}:2 = merk.foreign_call @merk_foreign_ops_Dtwo() : () -> (i64, i64)
  // CHECK-NEXT: merk.succeed
  merk.proc @ops.main_2_0() attributes {determinism = #merk.determinism<det>} {
    %n = merk.constant 42 : i64
    %f = merk.constant 2.5 : f64
    %c = merk.constant 97 : i32
    %s = merk.constant "hi\0A" : !merk.string
    %r = merk.foreign_call @merk_foreign_ops_Dshow(%s, %n) : (!merk.string, i64) -> i64
    %t:2 = merk.foreign_call @merk_foreign_ops_Dtwo() : () -> (i64, i64)
    merk.succeed
  }

  // Branches: an if-then-else whose condition binds a value for the then
  // part, a switch, a negation, tests, failure and an exception.
  // CHECK-LABEL: merk.proc private @ops.branch_2_0(%arg0: i64) -> i64
  // CHECK-SAME: attributes {determinism = #merk.determinism<semidet>}
  // CHECK-NEXT: %[[Y:.*]] = merk.ite -> i64 {
  // CHECK-NEXT: %[[H:.*]]:2 = merk.foreign_call @merk_foreign_ops_Dhalf(%arg0) can_fail : (i64) -> (i1, i64)
  // CHECK-NEXT: merk.guard %[[H]]#0
  // CHECK-NEXT: merk.yield %[[H]]#1 : i64
  // CHECK-NEXT: } then {
  // CHECK-NEXT: ^bb0(%[[HALF:.*]]: i64):
  // CHECK-NEXT: merk.yield %[[HALF]] : i64
  // CHECK-NEXT: } else {
  // CHECK-NEXT: %[[Z:.*]] = merk.switch %arg0 : i64 -> i64 cases {{\[}}[1, 3], [5]] can_fail {
  // CHECK-NEXT: merk.yield %arg0 : i64
  // CHECK-NEXT: }, {
  // CHECK-NEXT: merk.fail
  // CHECK-NEXT: }
  // CHECK-NEXT: merk.yield %[[Z]] : i64
  // CHECK-NEXT: }
  // CHECK-NEXT: merk.not {
  // CHECK-NEXT: %[[EQ:.*]] = arith.cmpi eq, %[[Y]], %arg0 : i64
  // CHECK-NEXT: merk.guard %[[EQ]]
  // CHECK-NEXT: merk.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[NEGATIVE:.*]] = arith.cmpi slt, %arg0, %[[ZERO]] : i64
  // CHECK-NEXT: merk.throw_if %[[NEGATIVE]], "domain_error(\22ops\22)"
  // CHECK-NEXT: merk.succeed %[[Y]] : i64
  merk.proc private @ops.branch_2_0(%x: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
    %y = merk.ite -> i64 {
      %h:2 = merk.foreign_call @merk_foreign_ops_Dhalf(%x) can_fail : (i64) -> (i1, i64)
      merk.guard %h#0
      merk.yield %h#1 : i64
    } then {
    ^bb0(%half: i64):
      merk.yield %half : i64
    } else {
      %z = merk.switch %x : i64 -> i64 cases [[1, 3], [5]] can_fail {
        merk.yield %x : i64
      }, {
        merk.fail
      }
      merk.yield %z : i64
    }
    merk.not {
      %eq = arith.cmpi eq, %y, %x : i64
      merk.guard %eq
      merk.yield
    }
    %zero = arith.constant 0 : i64
    %negative = arith.cmpi slt, %x, %zero : i64
    merk.throw_if %negative, "domain_error(\22ops\22)"
    merk.succeed %y : i64
  }

  // A tail call: a call whose results are the procedure's.
  // CHECK-LABEL: merk.proc private @ops.again_2_0(%arg0: i64) -> i64
  // CHECK-NEXT: %[[R:.*]] = merk.call @ops.branch_2_0(%arg0) : (i64) -> i64
  // CHECK-NEXT: merk.succeed %[[R]] : i64
  merk.proc private @ops.again_2_0(%x: i64) -> i64 attributes {determinism = #merk.determinism<semidet>} {
    %r = merk.call @ops.branch_2_0(%x) : (i64) -> i64
    merk.succeed %r : i64
  }

  // CHECK: merk.entry @ops.main_2_0
  merk.entry @ops.main_2_0
}

// Procedures are tailcc; several outputs are returned as a struct.
// LLVM: target triple = "x86_64-unknown-linux-gnu"
// LLVM: @[[HI:merk.str.[0-9]+]] = private unnamed_addr constant [4 x i8] c"hi\0A\00"
// LLVM: define internal tailcc i64 @ops.double_f1_0(i64 %0)
// LLVM: define internal tailcc { i64, double } @ops.two_2_0(i64 %0)
// LLVM: %[[F:.*]] = call tailcc double @other.float_f1_0(i64 %0)
// LLVM: insertvalue { i64, double } {{.*}}, double %[[F]], 1
// LLVM: declare tailcc double @other.float_f1_0(i64)
// LLVM: define tailcc void @ops.use_1_0(i64 %0)
// LLVM: call tailcc { i64, double } @ops.two_2_0
// A string is a constant global; a C function returns its only output, or
// stores several through pointers. The module's C code is compiled and
// linked in.
// LLVM: define tailcc void @ops.main_2_0()
// LLVM: alloca i64
// LLVM: alloca i64
// LLVM: call i64 @merk_foreign_ops_Dshow(ptr @[[HI]], i64 42)
// LLVM: call void @merk_foreign_ops_Dtwo(ptr %{{.*}}, ptr %{{.*}})
// A procedure that can fail returns its success flag before its outputs.
// LLVM: define internal tailcc { i1, i64 } @ops.branch_2_0(i64 %0)
// LLVM: %[[H:.*]] = call i32 @merk_foreign_ops_Dhalf(i64 %0, ptr %{{.*}})
// LLVM: icmp ne i32 %[[H]], 0
// LLVM: switch i64 %0, label %[[FAIL:.*]] [
// LLVM-NEXT: i64 1, label %[[CASE:.*]]
// LLVM-NEXT: i64 3, label %[[CASE]]
// LLVM-NEXT: i64 5, label %{{.*}}
// LLVM-NEXT: ]
// LLVM: call void @merk_uncaught_exception(ptr @merk.str.{{[0-9]+}})
// LLVM-NEXT: unreachable
// LLVM: [[FAIL]]:
// LLVM-NEXT: ret { i1, i64 } { i1 false, i64 undef }
// LLVM: define internal tailcc { i1, i64 } @ops.again_2_0(i64 %0)
// LLVM-NEXT: %[[R:.*]] = musttail call tailcc { i1, i64 } @ops.branch_2_0(i64 %0)
// LLVM-NEXT: ret { i1, i64 } %[[R]]
// LLVM: define void @merk_main()
// LLVM-NEXT: call tailcc void @ops.main_2_0()
// LLVM: define internal i64 @merk_foreign_ops_Dshow(
// LLVM: define internal void @merk_foreign_ops_Dtwo(
