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
// RUN: not merkc -fc1 -emit-merk %t/constant.mlir 2>&1 | FileCheck %s --check-prefix=CONSTANT
// RUN: not merkc -fc1 -emit-llvm %t/foreign.mlir 2>&1 | FileCheck %s --check-prefix=FOREIGN
// RUN: not merkc -fc1 -emit-merk %t/switch-regions.mlir 2>&1 | FileCheck %s --check-prefix=SWITCH-REGIONS
// RUN: not merkc -fc1 -emit-merk %t/switch-case.mlir 2>&1 | FileCheck %s --check-prefix=SWITCH-CASE
// RUN: not merkc -fc1 -emit-merk %t/switch-constant.mlir 2>&1 | FileCheck %s --check-prefix=SWITCH-CONSTANT
// RUN: not merkc -fc1 -emit-merk %t/yield.mlir 2>&1 | FileCheck %s --check-prefix=YIELD
// RUN: not merkc -fc1 -emit-merk %t/array-get.mlir 2>&1 | FileCheck %s --check-prefix=ARRAY-GET
// RUN: not merkc -fc1 -emit-merk %t/array-set.mlir 2>&1 | FileCheck %s --check-prefix=ARRAY-SET
// RUN: not merkc -fc1 -emit-merk %t/construct.mlir 2>&1 | FileCheck %s --check-prefix=CONSTRUCT
// RUN: not merkc -fc1 -emit-merk %t/deconstruct.mlir 2>&1 | FileCheck %s --check-prefix=DECONSTRUCT
// RUN: not merkc -fc1 -emit-merk %t/switch-functor.mlir 2>&1 | FileCheck %s --check-prefix=SWITCH-FUNCTOR
// RUN: not merkc -fc1 -emit-llvm %t/type-def.mlir 2>&1 | FileCheck %s --check-prefix=TYPE-DEF
// RUN: not merkc -fc1 -emit-llvm %t/functor.mlir 2>&1 | FileCheck %s --check-prefix=FUNCTOR
// RUN: not merkc -fc1 -emit-merk %t/instance.mlir 2>&1 | FileCheck %s --check-prefix=INSTANCE
// RUN: not merkc -fc1 -emit-merk %t/nondet-callee.mlir 2>&1 | FileCheck %s --check-prefix=NONDET-CALLEE
// RUN: not merkc -fc1 -emit-merk %t/nondet-call-type.mlir 2>&1 | FileCheck %s --check-prefix=NONDET-CALL-TYPE
// RUN: not merkc -fc1 -emit-merk %t/solution.mlir 2>&1 | FileCheck %s --check-prefix=SOLUTION
// RUN: not merkc -fc1 -emit-merk %t/solution-type.mlir 2>&1 | FileCheck %s --check-prefix=SOLUTION-TYPE

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

//--- constant.mlir
// CONSTANT: constant.mlir:[[@LINE+3]]:10: error: 'merk.constant' op is not an integer, float or string constant
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() attributes {determinism = #merk.determinism<det>} {
    %0 = merk.constant "text" : i64
    merk.succeed
  }
}

//--- foreign.mlir
// Without #line directives, clang's errors are in the temporary C file.
// FOREIGN: merk-foreign-{{.*}}.c:1:1: error: unknown type name 'not_a_type'
// FOREIGN: merkc: error: clang could not compile the module's foreign code
module attributes {merk.foreign_code = "not_a_type f(void);\0A", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
}

//--- switch-regions.mlir
// SWITCH-REGIONS: switch-regions.mlir:[[@LINE+3]]:5: error: 'merk.switch' op has 2 cases but 1 regions
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%x: i64) attributes {determinism = #merk.determinism<semidet>} {
    merk.switch %x : i64 cases [[1], [2]] can_fail {
      merk.yield
    }
    merk.succeed
  }
}

//--- switch-case.mlir
// SWITCH-CASE: switch-case.mlir:[[@LINE+3]]:5: error: 'merk.switch' op has a case that is not a list of constants
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%x: i64) attributes {determinism = #merk.determinism<det>} {
    merk.switch %x : i64 cases [[]] {
      merk.yield
    }
    merk.succeed
  }
}

//--- switch-constant.mlir
// SWITCH-CONSTANT: switch-constant.mlir:[[@LINE+3]]:5: error: 'merk.switch' op has a case constant that is not of type 'i64'
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%x: i64) attributes {determinism = #merk.determinism<det>} {
    merk.switch %x : i64 cases [[1 : i32]] {
      merk.yield
    }
    merk.succeed
  }
}

//--- yield.mlir
// YIELD: yield.mlir:[[@LINE+3]]:5: error: 'merk.yield' op expects parent op to be one of 'merk.ite, merk.switch, merk.not'
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() attributes {determinism = #merk.determinism<det>} {
    merk.yield
  }
}

//--- array-get.mlir
// ARRAY-GET: array-get.mlir:[[@LINE+3]]:10: error: 'merk.array_get' op result of type 'f64' is not of the array's element type 'i64'
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%a: !merk.array<i64>, %i: i64) -> f64 attributes {determinism = #merk.determinism<det>} {
    %x = merk.array_get %a[%i] : !merk.array<i64> -> f64
    merk.succeed %x : f64
  }
}

//--- array-set.mlir
// ARRAY-SET: array-set.mlir:[[@LINE+3]]:10: error: 'merk.array_set' op element of type '!merk.string' is not of the array's element type 'i64'
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p(%a: !merk.array<i64>, %i: i64, %s: !merk.string) -> !merk.array<i64> attributes {determinism = #merk.determinism<det>} {
    %b = merk.array_set %a[%i], %s : !merk.array<i64>, !merk.string
    merk.succeed %b : !merk.array<i64>
  }
}

//--- construct.mlir
// CONSTRUCT: construct.mlir:[[@LINE+4]]:10: error: 'merk.construct' op has 1 arguments for a functor of arity 2
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "t" [#merk.constructor<"f"(i64, i64)>]
  merk.proc @p(%x: i64) -> !merk.term<"t"> attributes {determinism = #merk.determinism<det>} {
    %t = merk.construct #merk.functor<"f", 2>(%x : i64) -> !merk.term<"t">
    merk.succeed %t : !merk.term<"t">
  }
}

//--- deconstruct.mlir
// DECONSTRUCT: deconstruct.mlir:[[@LINE+4]]:10: error: 'merk.deconstruct' op has 1 results for a functor of arity 2
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "t" [#merk.constructor<"f"(i64, i64)>]
  merk.proc @p(%t: !merk.term<"t">) -> i64 attributes {determinism = #merk.determinism<det>} {
    %x = merk.deconstruct #merk.functor<"f", 2> %t : !merk.term<"t"> -> i64
    merk.succeed %x : i64
  }
}

//--- switch-functor.mlir
// SWITCH-FUNCTOR: switch-functor.mlir:[[@LINE+4]]:5: error: 'merk.switch' op has a case constant that is not a functor
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "t" [#merk.constructor<"f">, #merk.constructor<"g">]
  merk.proc @p(%t: !merk.term<"t">) attributes {determinism = #merk.determinism<semidet>} {
    merk.switch %t : !merk.term<"t"> cases [[0]] can_fail {
      merk.yield
    }
    merk.succeed
  }
}

//--- type-def.mlir
// TYPE-DEF: type-def.mlir:[[@LINE+3]]:10: error: 'merk.construct' op is on a type with no merk.type_def
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.proc @p() -> !merk.term<"t"> attributes {determinism = #merk.determinism<det>} {
    %t = merk.construct #merk.functor<"f", 0> -> !merk.term<"t">
    merk.succeed %t : !merk.term<"t">
  }
}

//--- functor.mlir
// FUNCTOR: functor.mlir:[[@LINE+4]]:11: error: 'merk.has_functor' op has a functor its type does not have
module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  merk.type_def "t" [#merk.constructor<"f">, #merk.constructor<"g">]
  merk.proc @p(%t: !merk.term<"t">) attributes {determinism = #merk.determinism<semidet>} {
    %is = merk.has_functor #merk.functor<"f", 1> %t : !merk.term<"t">
    merk.guard %is
    merk.succeed
  }
}

//--- instance.mlir
// INSTANCE: instance.mlir:[[@LINE+3]]:10: error: 'merk.typeclass_info' op names @nothing, which is no instance
module {
  merk.proc @p() -> !merk.typeclass_info attributes {determinism = #merk.determinism<det>} {
    %d = merk.typeclass_info @nothing
    merk.succeed %d : !merk.typeclass_info
  }
}

//--- nondet-callee.mlir
// NONDET-CALLEE: nondet-callee.mlir:[[@LINE+7]]:5: error: 'merk.nondet_call' op calls @q, which is not nondet or multi
module {
  merk.proc @q() -> i64 attributes {determinism = #merk.determinism<det>} {
    %one = merk.constant 1 : i64
    merk.succeed %one : i64
  }
  merk.proc @p() -> i64 attributes {determinism = #merk.determinism<nondet>} {
    merk.nondet_call @q() {
    ^bb0(%x: i64):
      merk.succeed %x : i64
    }
  }
}

//--- nondet-call-type.mlir
// NONDET-CALL-TYPE: nondet-call-type.mlir:[[@LINE+6]]:5: error: 'merk.nondet_call' op does not match the type of @q
module {
  merk.proc @q() -> i64 attributes {determinism = #merk.determinism<nondet>} {
    merk.fail
  }
  merk.proc @p() -> i64 attributes {determinism = #merk.determinism<nondet>} {
    merk.nondet_call @q() {
    ^bb0(%x: i32):
      %y = arith.extsi %x : i32 to i64
      merk.succeed %y : i64
    }
  }
}

//--- solution.mlir
// SOLUTION: solution.mlir:[[@LINE+3]]:5: error: 'merk.solution' op is not in the generator of merk.each or merk.commit
module {
  merk.proc @p() attributes {determinism = #merk.determinism<det>} {
    merk.solution
  }
}

//--- solution-type.mlir
// SOLUTION-TYPE: solution-type.mlir:[[@LINE+5]]:7: error: 'merk.solution' op does not match the results of merk.commit
module {
  merk.proc @p() -> i64 attributes {determinism = #merk.determinism<cc_multi>} {
    %r = merk.commit -> i64 {
      %one = merk.constant 1 : i32
      merk.solution %one : i32
    }
    merk.succeed %r : i64
  }
}
