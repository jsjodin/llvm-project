% MANUAL: Purity levels
% A semipure goal's result depends on the impure goals run before it, and
% a pure predicate promised pure hides that from its callers.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[5, 7]
% OUTPUT-NOT:{{.}}

:- module run_semipure_sees_state.
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

:- pred set_then_get(int::in, int::out) is det.
:- pragma promise_pure(set_then_get/2).
set_then_get(X, Y) :-
    semipure get(Before),
    impure set(X),
    semipure get(After),
    Y = After - Before.

main(!IO) :-
    set_then_get(5, A), set_then_get(12, B),
    io.write([A, B], !IO), io.nl(!IO).
