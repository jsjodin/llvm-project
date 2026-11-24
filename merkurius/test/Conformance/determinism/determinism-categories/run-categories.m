% MANUAL: Determinism categories
% Each determinism category, declared on mode declarations and on pred and
% func declarations: det, semidet, multi, nondet, failure and erroneous,
% a predicate without arguments, and a function whose determinism is on
% its func declaration, which gives it the default mode.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 2]
% OUTPUT-NEXT:[{[], [1, 2]}, {[1], [2]}, {[1, 2], []}]
% OUTPUT-NEXT:yes
% OUTPUT-NEXT:[1, 2]
% OUTPUT-NEXT:never failed
% OUTPUT-NEXT:8
% OUTPUT-NEXT:odd
% OUTPUT-NOT:{{.}}

:- module run_categories.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, require, solutions, string.

:- pred app(list(T), list(T), list(T)).
:- mode app(in, in, out) is det.
:- mode app(out, out, in) is multi.
:- mode app(in, in, in) is semidet.
app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :-
    app(Xs, Ys, Zs).

:- pred small(int::out) is nondet.
small(X) :-
    ( X = 1 ; X = 2 ; X = 3 ),
    X < 3.

:- pred never is failure.
never :- fail.

:- pred stop(int::in) is erroneous.
stop(X) :-
    error("stop " ++ string.int_to_string(X)).

:- pred yes is det.
yes.

:- func double(int) = int is det.
double(X) = 2 * X.

:- func half(int) = int.
:- mode half(in) = out is semidet.
half(X) = X // 2 :-
    X mod 2 = 0.

main(!IO) :-
    app([1], [2], L),
    io.write_line(L, !IO),
    solutions((pred({A, B}::out) is multi :- app(A, B, [1, 2])), Splits),
    io.write_line(Splits, !IO),
    ( if app([1], [2], [1, 2]) then
        io.write_string("yes\n", !IO)
    else
        io.write_string("no\n", !IO)
    ),
    solutions(small, Smalls),
    io.write_line(Smalls, !IO),
    ( if never then
        stop(1)
    else
        io.write_string("never failed\n", !IO)
    ),
    yes,
    io.write_line(double(4), !IO),
    ( if H = half(5) then
        io.write_line(H, !IO)
    else
        io.write_string("odd\n", !IO)
    ).
