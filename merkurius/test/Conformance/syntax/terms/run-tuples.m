% MANUAL: Terms
% Tuple terms, including one with a quoted name and one in a clause head.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{{[{][{]}}1, '2', "three"}, 7, {}}
% OUTPUT-NOT:{{.}}

:- module run_tuples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred first({int, string, character}::in, int::out) is det.
first({X, _, _}, X).

main(!IO) :-
    T = {1, '2', "three"},
    first({7, "x", 'c'}, F),
    io.write({T, F, {}}, !IO), io.nl(!IO).
