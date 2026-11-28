% MANUAL: Syntax for explicit type quantifiers
% DESIGN: Sized floating point types
% The manual says universal quantifiers must precede existential ones, but
% mmc accepts some before all, and several quantifiers of each kind ("Where
% mmc and the manual disagree"). It is a template, run for float and
% float32.
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 %t.float.d/run_quantifier_order_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_quantifier_order_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1
% OUTPUT-FLOAT-NEXT:"two"
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 %t.float32.d/run_quantifier_order_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_quantifier_order_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1
% OUTPUT-FLOAT32-NEXT:"two"
% OUTPUT-FLOAT32-NOT:{{.}}

:- module run_quantifier_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@.
:- some [T2] all [T3] pred foo(T1, T2, T3).
:- mode foo(in, out, in) is det.
foo(_, 1, _).

:- all [T1] some [T2] all [T3] pred bar(T1, T2, T3).
:- mode bar(in, out, in) is det.
bar(_, "two", _).

main(!IO) :-
    foo(1, X, 'c'),
    io.write_line(X, !IO),
    bar(1.5@S@, Y, 'c'),
    io.write_line(Y, !IO).
