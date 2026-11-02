// The merk verifiers. Each module is checked on its own.
//
// RUN: split-file --leading-lines %s %t
// RUN: not merkc -fc1 -emit-merk %t/outputs.mlir 2>&1 | FileCheck %s --check-prefix=OUTPUTS
// RUN: not merkc -fc1 -emit-merk %t/output-type.mlir 2>&1 | FileCheck %s --check-prefix=OUTPUT-TYPE
// RUN: not merkc -fc1 -emit-merk %t/callee.mlir 2>&1 | FileCheck %s --check-prefix=CALLEE
// RUN: not merkc -fc1 -emit-merk %t/call-type.mlir 2>&1 | FileCheck %s --check-prefix=CALL-TYPE
// RUN: not merkc -fc1 -emit-merk %t/entry.mlir 2>&1 | FileCheck %s --check-prefix=ENTRY
// RUN: not merkc -fc1 -emit-merk %t/entry-arguments.mlir 2>&1 | FileCheck %s --check-prefix=ENTRY-ARGUMENTS
// RUN: not merkc -fc1 -emit-merk %t/determinism.mlir 2>&1 | FileCheck %s --check-prefix=DETERMINISM

//--- outputs.mlir
// OUTPUTS: outputs.mlir:[[@LINE+3]]:5: error: 'merk.succeed' op has 0 outputs, but its procedure has 1
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() -> i64 attributes {determinism = #merk.determinism<det>} {
    merk.succeed
  }
}

//--- output-type.mlir
// OUTPUT-TYPE: output-type.mlir:[[@LINE+3]]:5: error: 'merk.succeed' op output of type 'i64' does not match the procedure's 'f64'
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%x: i64) -> f64 attributes {determinism = #merk.determinism<det>} {
    merk.succeed %x : i64
  }
}

//--- callee.mlir
// CALLEE: callee.mlir:[[@LINE+3]]:5: error: 'merk.call' op calls @nowhere, which is no procedure
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() attributes {determinism = #merk.determinism<det>} {
    merk.call @nowhere() : () -> ()
    merk.succeed
  }
}

//--- call-type.mlir
// CALL-TYPE: call-type.mlir:[[@LINE+4]]:5: error: 'merk.call' op does not match the type of @q
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @q(i64) attributes {determinism = #merk.determinism<det>}
  merk.proc @p() attributes {determinism = #merk.determinism<det>} {
    merk.call @q() : () -> ()
    merk.succeed
  }
}

//--- entry.mlir
// ENTRY: entry.mlir:[[@LINE+2]]:3: error: 'merk.entry' op names @nowhere, which is no procedure
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.entry @nowhere
}

//--- entry-arguments.mlir
// ENTRY-ARGUMENTS: entry-arguments.mlir:[[@LINE+5]]:3: error: 'merk.entry' op names @p, which has inputs or outputs
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%x: i64) attributes {determinism = #merk.determinism<det>} {
    merk.succeed
  }
  merk.entry @p
}

//--- determinism.mlir
// DETERMINISM: determinism.mlir:[[@LINE+2]]:62: error: expected ::merkurius::merk::Determinism to be one of: det, semidet, multi, nondet, cc_multi, cc_nondet, erroneous, failure
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() attributes {determinism = #merk.determinism<sometimes>} {
    merk.succeed
  }
}
