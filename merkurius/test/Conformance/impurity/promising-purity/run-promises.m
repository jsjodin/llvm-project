% MANUAL: Promising purity
% promise_pure and promise_semipure pragmas, and the promise_pure and
% promise_impure scopes.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[42, 42, 42]
% OUTPUT-NOT:{{.}}

:- module run_promises.
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

:- pred pure_get(int::out) is det.
:- pragma promise_pure(pure_get/1).
pure_get(X) :- semipure get(X).
:- pred scoped_get(int::out) is det.
scoped_get(X) :- promise_pure ( semipure get(X) ).
:- semipure pred semi(int::out) is det.
:- pragma promise_semipure(semi/1).
semi(X) :- semipure get(X).
:- impure pred imp_scope(int::in) is det.
imp_scope(X) :- promise_impure ( Y = X + 1 ), impure set(Y).

main(!IO) :-
    promise_pure ( impure imp_scope(41), semipure semi(A) ),
    pure_get(B), scoped_get(C),
    io.write([A, B, C], !IO), io.nl(!IO).
