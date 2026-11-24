% MANUAL: User-defined equality and comparison
% So is a switch on such a value that can fail.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:

:- module reject_switch_can_fail.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t
    --->    a(int)
    ;       b(int)
            where equality is t_equals.

:- pred t_equals(t::in, t::in) is semidet.
t_equals(X, X).

:- pred is_a(t::in) is semidet.
is_a(T) :-
    T = a(_).                                       % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
