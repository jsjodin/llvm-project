% MANUAL: Unification expressions
% X @ Y in a head and in a body, examined for switches.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{f(1, 2), 1}
% OUTPUT-NEXT:{f(3, 4), f(3, 4)}
% OUTPUT-NOT:{{.}}

:- module run_unification_expressions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type t ---> f(int, int) ; g.
:- pred first(t::in, t::out, int::out) is semidet.
first(X @ f(A, _), X, A).

main(!IO) :-
    ( if first(f(1, 2), T, A) then io.write({T, A}, !IO) else true ), io.nl(!IO),
    Y = (Z @ f(3, 4)), io.write({Y, Z}, !IO), io.nl(!IO).
