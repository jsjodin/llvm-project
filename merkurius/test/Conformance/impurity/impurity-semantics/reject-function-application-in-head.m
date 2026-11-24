% MANUAL: Impurity semantics
% The manual's example of a higher-order call ("Purity annotations on
% higher-order calls"): the head's X1 + X2 is a function call, which may
% not wait past the impure goals that bind its arguments, as mmc rejects it
% (design document, the register).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:34:{{[0-9]+}}: error:

:- module reject_function_application_in_head.
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

:- func foo(impure pred(int)) = int.
:- mode foo(in(pred(out) is det)) = out is det.
:- pragma promise_pure(foo/1).
foo(ImpurePred) = X1 + X2 :- % ERROR
    impure ImpurePred(X1),
    impure call(ImpurePred, X2).

:- impure pred next(int::out) is det.
next(X) :- impure X = bump(1).

main(!IO) :- io.write_int(foo(next), !IO), io.nl(!IO).
