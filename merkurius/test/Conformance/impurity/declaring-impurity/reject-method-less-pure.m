% MANUAL: Declaring impurity
% An instance's method may not be less pure than the method's declaration.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:30:{{[0-9]+}}: error:

:- module reject_method_less_pure.
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

:- typeclass tc(T) where [ semipure pred m(T::in, int::out) is det ].
:- type c ---> c.
:- instance tc(c) where [ pred(m/2) is cm ]. % ERROR
:- impure pred cm(c::in, int::out) is det.
cm(_, X) :- impure X = bump(1).

main(!IO) :- promise_pure ( semipure m(c, X) ), io.write_int(X, !IO), io.nl(!IO).
