% MANUAL: Predicate and function type declarations
% Predicates and functions declare their argument types, polymorphic ones
% with type variables that each call instantiates. A declaration may give
% the type through a type qualification, as the manual's foldl_pred and
% foldl_func do. A predicate and a function may share a name and arity.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:upper
% OUTPUT-NEXT:5
% OUTPUT-NEXT:member
% OUTPUT-NEXT:3
% OUTPUT-NEXT:2
% OUTPUT-NEXT:3
% OUTPUT-NEXT:5
% OUTPUT-NEXT:11
% OUTPUT-NEXT:20
% OUTPUT-NEXT:30
% OUTPUT-NOT:{{.}}

:- module run_declarations.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type foldl_pred(T, U) == pred(T, U, U).
:- type foldl_func(T, U) == (func(T, U) = U).

:- pred is_all_uppercase(string::in) is semidet.
is_all_uppercase(S) :- string.to_upper(S) = S.

:- func strlen(string) = int.
strlen(S) = string.length(S).

:- pred contains(T, list(T)).
:- mode contains(in, in) is semidet.
contains(X, [Y | Ys]) :- ( X = Y ; contains(X, Ys) ).

:- func count(list(T)) = int.
count([]) = 0.
count([_ | Xs]) = 1 + count(Xs).

:- pred p(int) : foldl_pred(T, U).
:- mode p(in, in, in, out) is det.
p(_, _, U, U).

:- func f(int) : foldl_func(T, int).
f(N, _, U) = N + U.

:- pred twice(int::in, int::out) is det.
twice(X, 2 * X).
:- func twice(int) = int.
twice(X) = 3 * X.

main(!IO) :-
    ( if is_all_uppercase("ABC") then
        io.write_string("upper\n", !IO)
    else
        io.write_string("not upper\n", !IO)
    ),
    io.write_line(strlen("hello"), !IO),
    ( if contains(2, [1, 2, 3]) then
        io.write_string("member\n", !IO)
    else
        io.write_string("not member\n", !IO)
    ),
    io.write_line(count([1, 2, 3]), !IO),
    io.write_line(count(["a", "b"]), !IO),
    io.write_line(count([[1], [], [2, 3]]), !IO),
    p(1, "ignored", 5, U),
    io.write_line(U, !IO),
    io.write_line(f(1, 'x', 10), !IO),
    twice(10, T),
    io.write_line(T, !IO),
    io.write_line(twice(10) : int, !IO).
