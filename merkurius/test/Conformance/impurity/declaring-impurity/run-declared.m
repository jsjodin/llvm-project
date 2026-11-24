% MANUAL: Declaring impurity
% Impure and semipure predicates and functions, declared so, written in
% Merkurius over other impure and semipure ones.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[8, 7, 21]
% OUTPUT-NOT:{{.}}

:- module run_declared.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pragma foreign_decl("C", local, "static long state = 0;").
:- impure pred set(int::in) is det.
:- pragma foreign_proc("C", set(X::in), [will_not_call_mercury],
    "state = X;").
:- semipure pred get(int::out) is det.
:- pragma foreign_proc("C", get(X::out),
    [promise_semipure, will_not_call_mercury], "X = state;").
:- impure func bump(int) = int.
:- pragma foreign_proc("C", bump(X::in) = (Y::out), [will_not_call_mercury],
    "state += X; Y = state;").
:- semipure func peek = int.
:- pragma foreign_proc("C", peek = (X::out),
    [promise_semipure, will_not_call_mercury], "X = state;").

:- impure pred add(int::in) is det.
add(X) :- semipure get(S), impure set(S + X).
:- semipure pred doubled(int::out) is det.
doubled(Y) :- semipure get(X), Y = 2 * X.
:- impure func add_get(int) = int.
add_get(X) = Y :- impure add(X), semipure get(Y).
:- semipure func tripled = int.
tripled = Y :- semipure X = peek, Y = 3 * X.

main(!IO) :-
    promise_pure (
        impure add(4), semipure doubled(A),
        impure B = add_get(3), semipure C = tripled
    ),
    io.write([A, B, C], !IO), io.nl(!IO).
