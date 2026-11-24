% MANUAL: Purity annotations on lambda expressions
% The manual's impure function and predicate lambda expressions, and a
% semipure one.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[0, 5, 107]
% OUTPUT-NOT:{{.}}

:- module run_lambdas.
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

:- impure func call_func(impure func(int) = int, int) = int.
call_func(F, X) = Y :- impure Y = impure_apply(F, X).
:- impure pred call_pred(impure pred(int, int)::in(pred(in, out) is det),
    int::in, int::out) is det.
call_pred(P, X, Y) :- impure P(X, Y).
:- semipure func call_semi(semipure func(int) = int, int) = int.
call_semi(F, X) = Y :- semipure Y = semipure_apply(F, X).

main(!IO) :-
    F = (impure func(X) = Y :- semipure get(Y), impure set(X)),
    P = (impure pred(X::in, Y::out) is det :- semipure get(Y), impure set(X)),
    S = (semipure func(X) = Y :- semipure Z = peek, Y = Z + X),
    promise_pure (
        impure A = call_func(F, 5),
        impure call_pred(P, 7, B),
        semipure C = call_semi(S, 100)
    ),
    io.write([A, B, C], !IO), io.nl(!IO).
