% MANUAL: Purity annotations on lambda expressions
% A lambda expression whose body is impure must be annotated impure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:32:{{[0-9]+}}: error:

:- module reject_unannotated.
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

:- pred p(pred(int)::in(pred(out) is det), int::out) is det.
p(P, X) :- P(X).

main(!IO) :-
    p((pred(Y::out) is det :- impure Y = bump(1)), X), % ERROR
    io.write_int(X, !IO), io.nl(!IO).
