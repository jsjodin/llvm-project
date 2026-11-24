% MANUAL: Existential class constraints
% The manual's declaration with existential constraints c1 and c2 on T2 and
% a universal constraint c3 on T1: the callee satisfies the first, the
% caller the last, and the caller may use c1's and c2's methods.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"c3 s, c1 int, c2 int"
% OUTPUT-NOT:{{.}}

:- module run_constraints.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module string.

:- typeclass c1(T) where [ func name1(T) = string ].
:- typeclass c2(T) where [ func name2(T) = string ].
:- typeclass c3(T) where [ func name3(T) = string ].

:- instance c1(int) where [ name1(_) = "c1 int" ].
:- instance c2(int) where [ name2(_) = "c2 int" ].
:- instance c3(string) where [ name3(S) = "c3 " ++ S ].

    % Here `c1(T2)' and `c2(T2)' are existential constraints,
    % and `c3(T1)' is a universal constraint,
:- all [T1] some [T2] ((pred p(T1, T2) => (c1(T2), c2(T2))) <= c3(T1)).
:- mode p(in, out) is det.
p(_, 5).

:- pred describe(T1::in, string::out) is det <= c3(T1).
describe(X, S) :-
    p(X, Y),
    S = name3(X) ++ ", " ++ name1(Y) ++ ", " ++ name2(Y).

main(!IO) :-
    describe("s", S),
    io.write_line(S, !IO).
