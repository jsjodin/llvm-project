% MANUAL: Termination analysis
% terminates, does_not_terminate and check_termination pragmas, which do
% not change what a program does.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:9
% OUTPUT-NOT:{{.}}

:- module run_termination_pragmas.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred p(int::in, int::out) is det.
:- pragma terminates(p/2).
p(N, N + 1).
:- pred q(int::in) is det.
:- pragma does_not_terminate(q/1).
q(_).
:- pred r(list(int)::in, int::out) is det.
:- pragma check_termination(r/2).
r([], 0).
r([X | Xs], X + S) :- r(Xs, S).

main(!IO) :- p(3, X), q(X), r([X, 2, 3], S), io.write_int(S, !IO), io.nl(!IO).
