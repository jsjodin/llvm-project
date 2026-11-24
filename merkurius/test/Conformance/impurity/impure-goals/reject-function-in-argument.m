% MANUAL: Impure goals
% An impure function may be applied only in a unification with a variable
% annotated impure, not as an argument.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:33:{{[0-9]+}}: error:

:- module reject_function_in_argument.
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

:- pred q(int::in, int::out) is det.
q(X, X).
:- pred p(int::out) is det.
:- pragma promise_pure(p/1).
p(X) :- impure q(bump(1), X). % ERROR

main(!IO) :- p(X), io.write_int(X, !IO), io.nl(!IO).
