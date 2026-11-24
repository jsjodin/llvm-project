% MANUAL: Purity annotations on higher-order types
% A pure closure is not of an impure higher-order type: there is no
% implicit conversion.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_no_conversion.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred p(impure pred(int)::in(pred(out) is det), int::out) is det.
:- pragma promise_pure(p/2).
p(P, X) :- impure P(X).

main(!IO) :-
    p((pred(Y::out) is det :- Y = 1), X), % ERROR
    io.write_int(X, !IO), io.nl(!IO).
