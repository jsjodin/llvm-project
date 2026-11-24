% MANUAL: Purity annotations on higher-order calls
% Calls of impure higher-order predicates, by higher-order syntax and by
% call/N, and the manual's map over an impure function with impure_apply.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NEXT:[12, 32, 62]
% OUTPUT-NOT:{{.}}

:- module run_calls.
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

:- func twice(impure pred(int)) = int.
:- mode twice(in(pred(out) is det)) = out is det.
:- pragma promise_pure(twice/1).
twice(ImpurePred) = X :-
    impure ImpurePred(X1),
    impure call(ImpurePred, X2),
    X = X1 + X2.
:- func imap(impure func(T1) = T2, list(T1)) = list(T2).
:- pragma promise_pure(imap/2).
imap(_ImpureFunc, []) = [].
imap(ImpureFunc, [X | Xs]) = [Y | Ys] :-
    impure Y = impure_apply(ImpureFunc, X),
    impure Ys = imap(ImpureFunc, Xs).
:- impure pred next(int::out) is det.
next(X) :- impure X = bump(1).

main(!IO) :-
    io.write_int(twice(next), !IO), io.nl(!IO),
    io.write(imap(bump, [10, 20, 30]), !IO), io.nl(!IO).
