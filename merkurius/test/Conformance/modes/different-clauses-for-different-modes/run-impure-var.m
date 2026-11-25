% MANUAL: Different clauses for different modes
% Clauses for different modes need not be equivalent, as in the manual's
% var/1, so such a predicate must be declared impure.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:X is bound
% OUTPUT-NEXT:_ is a variable
% OUTPUT-NOT:{{.}}

:- module run_impure_var.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- impure pred var(T).
:- mode var(in) is failure.
:- mode var(free >> free) is det.
var(_::in) :- fail.
var(_::(free >> free)) :- true.

main(!IO) :-
    X = 1,
    promise_pure (
        ( if impure var(X) then
            io.write_string("X is a variable\n", !IO)
        else
            io.write_string("X is bound\n", !IO)
        ),
        ( if impure var(_ : int) then
            io.write_string("_ is a variable\n", !IO)
        else
            io.write_string("_ is bound\n", !IO)
        )
    ).
