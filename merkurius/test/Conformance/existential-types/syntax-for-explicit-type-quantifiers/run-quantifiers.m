% MANUAL: Syntax for explicit type quantifiers
% The manual's declarations with explicit quantifiers: some on a pred and a
% func, a mix of existential and universal variables, and all before
% some.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NEXT:{7 - "n", [7], ["n"]}
% OUTPUT-NEXT:"middle"
% OUTPUT-NEXT:1.5
% OUTPUT-NOT:{{.}}

:- module run_quantifiers.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, pair.

    % Here the type variable `T' is existentially quantified
:- some [T] pred foo(T).
:- mode foo(out) is det.
foo(42).

    % Here the type variables `T1' and `T2' are existentially quantified.
    % (list(T2) in place of the manual's set(T2).)
:- some [T1, T2] func bar(int, list(T1), list(T2)) = pair(T1, T2).
:- mode bar(in, out, out) = out is det.
bar(N, [N], ["n"]) = N - "n".

    % Here the type variable `T2' is existentially quantified,
    % but the type variables `T1' and `T3' are universally quantified.
:- some [T2] pred foo3(T1, T2, T3).
:- mode foo3(in, out, in) is det.
foo3(_, "middle", _).

:- all [T3] some [T2] pred foo4(T1, T2, T3).
:- mode foo4(in, out, in) is det.
foo4(_, 1.5, _).

main(!IO) :-
    foo(X),
    io.write_line(X, !IO),
    P = bar(7, L1, L2),
    io.write_line({P, L1, L2}, !IO),
    foo3(1, Y, 'c'),
    io.write_line(Y, !IO),
    foo4(1, Z, 'c'),
    io.write_line(Z, !IO).
