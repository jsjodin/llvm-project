// The merk operations print as they are parsed: -fc1 reads a .mlir file and
// writes it again with -emit-merk, the same the second time. It also lowers
// one to LLVM IR for the target the module records.
//
// RUN: merkc -fc1 -emit-merk %s -o %t.1.mlir
// RUN: FileCheck %s < %t.1.mlir
// RUN: merkc -fc1 -emit-merk %t.1.mlir -o %t.2.mlir
// RUN: diff %t.1.mlir %t.2.mlir
// RUN: merkc -fc1 -emit-llvm %s | FileCheck %s --check-prefix=LLVM

module @ops attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", merk.address_spaces = #merk.address_spaces<generic = 0, global = 1, shared = 3, constant = 4, private = 5>} {
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

  // CHECK-LABEL: merk.proc @ops.main_2_0()
  merk.proc @ops.main_2_0() attributes {determinism = #merk.determinism<det>} {
    merk.succeed
  }

  // CHECK: merk.entry @ops.main_2_0
  merk.entry @ops.main_2_0
}

// Procedures are tailcc; several outputs are returned as a struct.
// LLVM: target triple = "x86_64-unknown-linux-gnu"
// LLVM: define internal tailcc i64 @ops.double_f1_0(i64 %0)
// LLVM: define internal tailcc { i64, double } @ops.two_2_0(i64 %0)
// LLVM: %[[F:.*]] = call tailcc double @other.float_f1_0(i64 %0)
// LLVM: insertvalue { i64, double } {{.*}}, double %[[F]], 1
// LLVM: declare tailcc double @other.float_f1_0(i64)
// LLVM: define tailcc void @ops.use_1_0(i64 %0)
// LLVM: call tailcc { i64, double } @ops.two_2_0
// LLVM: define void @merk_main()
// LLVM-NEXT: call tailcc void @ops.main_2_0()
