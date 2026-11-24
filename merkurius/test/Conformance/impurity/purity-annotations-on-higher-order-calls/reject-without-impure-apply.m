% MANUAL: Purity annotations on higher-order calls
% An impure higher-order function is applied with impure_apply, not by
% higher-order syntax.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:31:{{[0-9]+}}: error:

:- module reject_without_impure_apply.
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

:- func p(impure func(int) = int, int) = int.
:- pragma promise_pure(p/2).
p(F, X) = Y :- impure Y = F(X). % ERROR

main(!IO) :- io.write_int(p(bump, 1), !IO), io.nl(!IO).
