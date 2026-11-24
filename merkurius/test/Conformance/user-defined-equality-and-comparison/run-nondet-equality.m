% MANUAL: User-defined equality and comparison
% The manual says an equality predicate is semidet or more specific, but
% mmc accepts a nondet one, whose call has no outputs and so commits to its
% first solution ("Where mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:equal
% OUTPUT-NEXT:different
% OUTPUT-NOT:{{.}}

:- module run_nondet_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- type parity
    --->    parity(int)
            where equality is same_parity.

:- pred same_parity(parity::in, parity::in) is nondet.
same_parity(A, B) :-
    promise_equivalent_solutions [X, Y] (
        A = parity(X),
        B = parity(Y)
    ),
    ( X mod 2 = Y mod 2
    ; X = Y
    ).

:- pred test(parity::in, parity::in, io::di, io::uo) is det.
test(A, B, !IO) :-
    ( if A = B then
        io.write_string("equal\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).

main(!IO) :-
    test(parity(1), parity(3), !IO),
    test(parity(1), parity(2), !IO).
